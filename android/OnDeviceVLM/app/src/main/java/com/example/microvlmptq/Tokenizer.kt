package com.example.microvlmptq

import org.json.JSONObject

/**
 * Word-level tokenizer with exact parity to python/common.py.
 * vocab.json is produced by the training pipeline.
 */
class Tokenizer private constructor(
    val vocab: Map<String, Int>,
    val itos: Map<Int, String>,
    val padLen: Int,
    val size: Int
) {
    private val bos: Int = vocab.getValue("<bos>")
    private val eos: Int = vocab.getValue("<eos>")
    private val pad: Int = vocab.getValue("<pad>")
    private val unk: Int = vocab.getValue("<unk>")

    fun encode(text: String): Pair<IntArray, IntArray> {
        val toks = mutableListOf(bos)
        for (w in text.trim().split(Regex("\\s+"))) {
            if (w.isEmpty()) continue
            toks.add(vocab[w] ?: unk)
        }
        toks.add(eos)
        if (toks.size > padLen) {
            toks[padLen - 1] = eos
            while (toks.size > padLen) toks.removeAt(toks.size - 1)
        }
        val ids = IntArray(padLen) { pad }
        val mask = IntArray(padLen) { 0 }
        for (i in toks.indices) {
            ids[i] = toks[i]
            mask[i] = 1
        }
        return Pair(ids, mask)
    }

    fun decode(ids: IntArray, ignoreSpecials: Boolean = true): String {
        val words = mutableListOf<String>()
        for (id in ids) {
            val w = itos[id] ?: "<unk>"
            if (ignoreSpecials && (w == "<pad>" || w == "<bos>" || w == "<eos>" || w == "<unk>")) continue
            words.add(w)
        }
        return words.joinToString(" ")
    }

    companion object {
        fun fromJson(json: JSONObject): Tokenizer {
            val vMap = HashMap<String, Int>()
            val vocabObj = json.getJSONObject("vocab")
            val keys = vocabObj.keys()
            while (keys.hasNext()) {
                val k = keys.next()
                vMap[k] = vocabObj.getInt(k)
            }
            val itos = HashMap<Int, String>()
            for ((k, v) in vMap) itos[v] = k
            return Tokenizer(vMap, itos, json.getInt("pad_len"), vMap.size)
        }
    }
}