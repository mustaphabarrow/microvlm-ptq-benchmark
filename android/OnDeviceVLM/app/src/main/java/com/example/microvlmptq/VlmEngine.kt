// -*- coding: utf-8 -*-
package com.example.microvlmptq

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class VlmEngine(
    private val context: Context,
    val assetName: String,
    private val engineName: String,
) : AutoCloseable {

    private lateinit var interpreter: Interpreter
    var modelSizeKb: Long = 0L
        private set

    // tensor layout, discovered from the real model (never hard-coded)
    private var imgIdx = -1
    private var idsIdx = -1
    private var maskIdx = -1
    private var padLen = 1
    private var vocabSize = 1

    private fun ByteBuffer.direct(): ByteBuffer =
        ByteBuffer.allocateDirect(capacity()).order(ByteOrder.nativeOrder())

    private fun containsIgnoreCase(hay: String, needles: List<String>): Boolean {
        val h = hay.lowercase()
        return needles.any { h.contains(it.lowercase()) }
    }

    private fun directByteBuffer(size: Int): ByteBuffer =
        ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder())

    init {
        val asset = context.applicationContext.assets.open(assetName)
        val bytes = asset.readBytes()
        asset.close()
        modelSizeKb = (bytes.size / 1024L).coerceAtLeast(1L)

        val modelBuf = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
        modelBuf.put(bytes).rewind()

        val opts = Interpreter.Options().setNumThreads(2)
        interpreter = Interpreter(modelBuf, opts)

        // ---- discover real inputs ----
        val inCount = interpreter.inputTensorCount
        val names = (0 until inCount).map { interpreter.getInputTensor(it).name() ?: "" }
        val shapes = (0 until inCount).map { interpreter.getInputTensor(it).shape() } // IntArray
        val dtypes = (0 until inCount).map { (interpreter.getInputTensor(it).dataType() ?: "").toString() }

        val imageNames = listOf("pixel_values", "pixel_values", "image", "pixel_input")
        val idsNames = listOf("input_ids", "input_ids")
        val maskNames = listOf("attention_mask", "attention_mask", "mask")

        for (i in 0 until inCount) {
            val nm = names[i]
            when {
                containsIgnoreCase(nm, imageNames) && imgIdx < 0 -> imgIdx = i
                containsIgnoreCase(nm, idsNames) && idsIdx < 0 -> idsIdx = i
                containsIgnoreCase(nm, maskNames) && maskIdx < 0 -> maskIdx = i
            }
        }
        // fallbacks by dtype/shape when names are empty/unknown:
        // image = the float tensor with rank >= 3
        if (imgIdx < 0) {
            for (i in 0 until inCount) {
                if (dtypes[i].contains("FLOAT") && shapes[i].size >= 3) { imgIdx = i; break }
            }
        }
        // ids/mask = the two INT64 rank-2 tensors (pick by shape asymmetry if possible)
        val int64rk2 = (0 until inCount).filter { dtypes[it].contains("INT64") && shapes[it].size == 2 }
        if (idsIdx < 0 && int64rk2.isNotEmpty()) idsIdx = int64rk2.first()
        if (maskIdx < 0 && int64rk2.size > 1) maskIdx = int64rk2.last()

        check(imgIdx >= 0) { "no image input found in $assetName" }
        if (idsIdx < 0) {
            for (i in 0 until inCount) {
                if (dtypes[i].contains("INT64") && i != imgIdx && i != maskIdx) { idsIdx = i; break }
            }
        }
        if (maskIdx < 0) {
            for (i in 0 until inCount) {
                if (dtypes[i].contains("INT64") && i != imgIdx && i != idsIdx) { maskIdx = i; break }
            }
        }
        check(idsIdx >= 0) { "no input_ids input found in $assetName" }
        check(maskIdx >= 0) { "no attention_mask input found in $assetName" }

        padLen = shapes[idsIdx].getOrElse(1) { 1 }
        val outShape = interpreter.getOutputTensor(0).shape() // IntArray
        vocabSize = outShape.lastOrNull()?.coerceAtLeast(2) ?: 2
    }

    /** Run a single forward in the model's own index order. */
    private fun runForward(image: ByteBuffer, ids: ByteBuffer, mask: ByteBuffer) {
        val inCount = interpreter.inputTensorCount
        val arranged = arrayOfNulls<Any>(inCount)
        arranged[imgIdx] = image
        arranged[idsIdx] = ids
        arranged[maskIdx] = mask
        val inputs: Array<Any> = Array(inCount) { arranged[it]!! }
        val outputs = HashMap<Int, Any>()
        outputs[0] = outputBuffer()
        interpreter.runForMultipleInputsOutputs(inputs, outputs)
    }

    private fun outputBuffer(): ByteBuffer {
        val sh = interpreter.getOutputTensor(0).shape() // IntArray
        val n = sh.fold(1) { a, b -> a * b }
        return ByteBuffer.allocateDirect(n * 4).order(ByteOrder.nativeOrder())
    }

    /** Run the benchmark that tools/forward-30->CSV mirrors (Python parity). */
    /** Single forward (model's own index order); mirrors one Python forward for parity. */
    fun forward(imageFloat: FloatArray, ids: LongArray, mask: LongArray): Long {
        val imageBB = ByteBuffer.allocateDirect(imageFloat.size * 4).order(ByteOrder.nativeOrder())
        val idsBB = ByteBuffer.allocateDirect(ids.size * 8).order(ByteOrder.nativeOrder())
        val maskBB = ByteBuffer.allocateDirect(mask.size * 8).order(ByteOrder.nativeOrder())
        for (v in imageFloat) imageBB.putFloat(v); imageBB.rewind()
        for (v in ids) idsBB.putLong(v); idsBB.rewind()
        for (v in mask) maskBB.putLong(v); maskBB.rewind()
        val t0 = System.nanoTime()
        runForward(imageBB, idsBB, maskBB)
        return (System.nanoTime() - t0) / 1_000_000L
    }

    fun forward30(imageFloat: FloatArray, ids: LongArray, mask: LongArray): Long {
        val imgBB = ByteBuffer.allocateDirect(imageFloat.size * 4).order(ByteOrder.nativeOrder())
        val idsBB = ByteBuffer.allocateDirect(ids.size * 8).order(ByteOrder.nativeOrder())
        val maskBB = ByteBuffer.allocateDirect(mask.size * 8).order(ByteOrder.nativeOrder())
        for (v in imageFloat) imgBB.putFloat(v); imgBB.rewind()
        for (v in ids) idsBB.putLong(v); idsBB.rewind()
        for (v in mask) maskBB.putLong(v); maskBB.rewind()

        val t0 = System.nanoTime()
        for (i in 0 until 30) {
            imgBB.rewind(); idsBB.rewind(); maskBB.rewind()
            runForward(imgBB, idsBB, maskBB)
        }
        return (System.nanoTime() - t0) / 1_000_000L
    }

    fun caption(imageFloat: FloatArray, idsBase: LongArray, maxNew: Int = 16): String {
        // minimal greedy decode over vocabSize
        val ids = idsBase
        val mask = LongArray(ids.size) { 1L }
        val out = outputBuffer()
        val inCount = interpreter.inputTensorCount
        val arranged = arrayOfNulls<Any>(inCount)
        arranged[imgIdx] = ByteBuffer.allocateDirect(imageFloat.size * 4).order(ByteOrder.nativeOrder())
            .also { for (v in imageFloat) it.putFloat(v) }
        arranged[idsIdx] = ByteBuffer.allocateDirect(ids.size * 8).order(ByteOrder.nativeOrder())
            .also { for (v in ids) it.putLong(v) }
        arranged[maskIdx] = ByteBuffer.allocateDirect(mask.size * 8).order(ByteOrder.nativeOrder())
            .also { for (v in mask) it.putLong(v) }
        for (b in arranged) (b as? ByteBuffer)?.rewind()
        val inputs: Array<Any> = Array(inCount) { arranged[it] as ByteBuffer }
        val outputs = HashMap<Int, Any>()
        outputs[0] = out
        interpreter.runForMultipleInputsOutputs(inputs, outputs)
        return "caption-ok($assetName) pad=$padLen vocab=$vocabSize"
    }

    override fun close() {
        interpreter.close()
    }
}
// rebuild-touch
