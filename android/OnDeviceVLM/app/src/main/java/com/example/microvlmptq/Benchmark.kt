package com.example.microvlmptq

import android.os.Debug
import android.os.SystemClock
import java.io.File
import kotlin.math.roundToLong

object Benchmark {
    data class Result(
        val variant: String,
        val sizeKb: Long,
        val iters: Int,
        val medianMs: Double,
        val meanMs: Double,
        val p95Ms: Double,
        val threadCpuMs: Long,
        val vmRssKb: Long,
        val captionCorrect: Boolean,
    ) {
        fun toCsvRow(): String =
            "$variant,$sizeKb,$iters," +
                "%.4f,%.4f,%.4f,$threadCpuMs,$vmRssKb,$captionCorrect"
                    .format(medianMs, meanMs, p95Ms)
    }

    /** Settled RSS (kB) via /proc/self/status. */
    fun vmRssKb(): Long {
        return try {
            File("/proc/self/status").readLines()
                .firstOrNull { it.startsWith("VmRSS:") }
                ?.substringAfter(":")
                ?.trim()
                ?.substringBefore(" kB")
                ?.toLong() ?: -1L
        } catch (e: Exception) {
            -1L
        }
    }

    /** Run N forwards with warm-up; returns timings plus rss/cpu readings. */
    fun runForwards(engine: VlmEngine, imgFloat: FloatArray, ids: IntArray, mask: IntArray, iters: Int): Result {
        val idsL = ids.map { it.toLong() }.toLongArray()
        val maskL = mask.map { it.toLong() }.toLongArray()
        // warm-up
        repeat(5) { engine.forward(imgFloat, idsL, maskL) }

        val times = LongArray(iters)
        var startCpu = SystemClock.currentThreadTimeMillis()
        for (i in 0 until iters) {
            val t0 = System.nanoTime()
            engine.forward(imgFloat, idsL, maskL)
            times[i] = System.nanoTime() - t0
        }
        val cpuMs = SystemClock.currentThreadTimeMillis() - startCpu

        val sorted = times.sortedArray()
        val median = sorted[iters / 2] / 1e6
        val mean = times.average() / 1e6
        val p95 = sorted[(iters * 0.95).toInt().coerceIn(0, iters - 1)] / 1e6
        Runtime.getRuntime().gc()
        return Result(engine.assetName, engine.modelSizeKb, iters, median, mean, p95, cpuMs,
            vmRssKb(), false)
    }

    fun dumpCsv(dir: File, rows: List<Result>): File {
        val header = "variant,size_kb,iters,median_ms,mean_ms,p95_ms,thread_cpu_ms,vm_rss_kb,caption_correct\n"
        val f = File(dir, "microvlm_ondevice_benchmark_${System.currentTimeMillis()}.csv")
        f.writeText(header + rows.joinToString("\n") { it.toCsvRow() })
        return f
    }
}