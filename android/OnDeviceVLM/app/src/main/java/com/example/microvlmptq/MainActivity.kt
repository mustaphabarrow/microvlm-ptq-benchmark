package com.example.microvlmptq

import android.os.Bundle
import android.os.Environment
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ImageView
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import org.json.JSONObject
import java.io.File
import java.util.Locale
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    private lateinit var tokenizer: Tokenizer
    private val engines = HashMap<String, VlmEngine>()

    private lateinit var spinner: Spinner
    private lateinit var sceneImage: ImageView
    private lateinit var trueCaption: TextView
    private lateinit var predCaption: TextView
    private lateinit var resultText: TextView
    private var currentScene: SceneGenerator.Scene? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val vocabRaw = assets.open("vocab.json").readBytes().toString(Charsets.UTF_8)
        tokenizer = Tokenizer.fromJson(JSONObject(vocabRaw))

        spinner = findViewById(R.id.variantSpinner)
        sceneImage = findViewById(R.id.sceneImage)
        trueCaption = findViewById(R.id.trueCaption)
        predCaption = findViewById(R.id.predCaption)
        resultText = findViewById(R.id.resultText)

        val variants = assets.list("").orEmpty()
            .filter { it.endsWith(".tflite") }
            .sorted()
        val defaultVariant = "microvlm_int8.tflite"
        spinner.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item, variants)
        val defaultIdx = variants.indexOf(defaultVariant).coerceAtLeast(0)
        spinner.setSelection(defaultIdx)

        findViewById<Button>(R.id.newSceneBtn).setOnClickListener { newScene() }
        findViewById<Button>(R.id.captionBtn).setOnClickListener { runCaption() }
        findViewById<Button>(R.id.benchBtn).setOnClickListener { runBenchmark() }
        findViewById<Button>(R.id.saveCsvBtn).setOnClickListener { exportCsv() }

        newScene()
    }

    private fun variant(): String = spinner.selectedItem?.toString() ?: "microvlm_int8.tflite"

    private fun engine(): VlmEngine {
        val v = variant()
        return engines.getOrPut(v) { VlmEngine(this, v, v) }
    }

    private fun newScene() {
        currentScene = SceneGenerator.randomScene(this)
        sceneImage.setImageBitmap(currentScene!!.bitmap)
        trueCaption.text = "true: ${currentScene!!.caption}"
        predCaption.text = "pred: -"
        resultText.text = ""
    }

    private fun runCaption() {
        val scene = currentScene ?: return
        predCaption.text = "pred: running..."
        thread(name = "caption") {
            try {
                val img = SceneGenerator.toFloatInput(scene.bitmap)
                val (idsBase) = tokenizer.encode("a large blue star at the right")
                val caption = engine().caption(img, idsBase.map { it.toLong() }.toLongArray())
                val steps = idsBase.size
                runOnUiThread {
                    predCaption.text = "pred: $caption  ($steps steps)"
                    val correct = caption == scene.caption
                    predCaption.append(if (correct) "  [CORRECT]" else "")
                }
            } catch (e: Exception) {
                runOnUiThread { predCaption.text = "pred: ERROR ${e.message}" }
            }
        }
    }

    private fun runBenchmark() {
        val scene = currentScene ?: return
        resultText.text = "benchmarking ${variant()}..."
        val v = variant()
        thread(name = "bench") {
            try {
                val eng = engine()
                val img = SceneGenerator.toFloatInput(scene.bitmap)
                val (ids, mask) = tokenizer.encode("a large blue star at the right")
                val res = Benchmark.runForwards(eng, img, ids, mask, iters = 30)
                val out = String.format(
                    Locale.US,
                    "variant  : %s\nmodel    : %d KB\niters    : %d\n" +
                        "median   : %.3f ms\nmean     : %.3f ms\np95      : %.3f ms\n" +
                        "threadCPU: %d ms\nVmRSS    : %d kB",
                    res.variant, res.sizeKb, res.iters, res.medianMs, res.meanMs,
                    res.p95Ms, res.threadCpuMs, res.vmRssKb)
                runOnUiThread { resultText.text = out }
            } catch (e: Exception) {
                val msg = e.message ?: "error"
                runOnUiThread { resultText.text = "benchmark failed: $msg" }
            }
        }
    }

    private fun exportCsv() {
        val scene = currentScene ?: return
        thread(name = "csv") {
            try {
                val dir = File(getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS), "microvlm")
                dir.mkdirs()
                val img = SceneGenerator.toFloatInput(scene.bitmap)
                val (ids, mask) = tokenizer.encode("a large blue star at the right")

                val rows = mutableListOf<Benchmark.Result>()
                for ((name, eng) in engines) {
                    val r = Benchmark.runForwards(eng, img, ids, mask, iters = 30)
                    rows += r.copy(variant = name)
                }
                for (asset in assets.list("").orEmpty().filter { it.endsWith(".tflite") }) {
                    try {
                        val eng = engineOf(asset)
                        val r = Benchmark.runForwards(eng, img, ids, mask, iters = 30)
                        rows += r.copy(variant = asset)
                    } catch (e: Exception) {
                        rows += Benchmark.Result(
                            variant = asset, sizeKb = -1, iters = 0, medianMs = -1.0,
                            meanMs = -1.0, p95Ms = -1.0, threadCpuMs = 0, vmRssKb = 0,
                            captionCorrect = false)
                    }
                }
                val f = Benchmark.dumpCsv(dir, rows)
                runOnUiThread {
                    resultText.text = "CSV saved:\n${f.absolutePath}\nrows=${rows.size}"
                    Toast.makeText(this, "CSV exported", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                runOnUiThread { resultText.text = "export failed: ${e.message}" }
            }
        }
    }

    private fun engineOf(asset: String): VlmEngine =
        engines.getOrPut(asset) { VlmEngine(this, asset, asset) }

    override fun onDestroy() {
        engines.values.forEach { it.close() }
        super.onDestroy()
    }
}// rebuild-touch $((Get-Date).Ticks)
// rebuild-touch
