// GptModel — the tiny character-level GPT of the "gpt" demo, re-implemented in pure Kotlin.
//
// Architecture (identical to tools/train_gpt.py, which trains it and writes the weights):
//   x_0[t]   = wte[token_t] + wpe[t]                                  (row 0 of the 3-D scene)
//   per layer: x += Wo · MultiHeadCausalAttention(LN1(x))             (attention: the light threads)
//              x += W2 · gelu_tanh(W1 · LN2(x) + b1) + b2             (MLP)
//                                                                    (rows 1, 2 of the scene)
//   logits[t] = LN_f(x[t]) · wteᵀ                                     (tied output head -> the bar fan)
// Attention per head h: a[i][j] = softmax_j( q_i·k_j / sqrt(d_head) ) over j <= i (causal mask).
// GELU uses the tanh approximation (PyTorch approximate="tanh"); LayerNorm eps = 1e-5.
//
// Everything is computed in Double from the float32 weights that ship, so the JVM test can compare
// against the PyTorch float64 forward of the same weights within 1e-4 (in practice ~1e-6).
//
// model.bin layout (little-endian, written by tools/train_gpt.py write_model):
//   "GPT1" | int32 version, V, ctx, d, nLayer, nHead, dff, nPca, vocabBytes | UTF-8 vocab (V chars)
//   float32: wte[V][d], wpe[ctx][d], per layer { ln1.w[d], ln1.b[d], qkv.w[3d][d], qkv.b[3d],
//            proj.w[d][d], proj.b[d], ln2.w[d], ln2.b[d], fc.w[dff][d], fc.b[dff], fc2.w[d][dff], fc2.b[d] },
//            lnf.w[d], lnf.b[d], per row r = 0..nLayer { mean[d], comps[nPca][d], std[nPca] }
//   int32 trainChars, valChars | float32 valPerplexity, unigramPerplexity (as printed by the script)

package com.bytedance.ailab.core.gpt

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.sqrt
import kotlin.math.tanh

class GptModel(bytes: ByteArray) {
    val vocab: String
    val V: Int; val ctx: Int; val d: Int; val nLayer: Int; val nHead: Int; val dff: Int; val nPca: Int
    val trainChars: Int; val valChars: Int
    /** Validation perplexity / unigram baseline as measured by the training script (PyTorch). */
    val reportedValPpl: Float; val reportedUnigramPpl: Float

    private val wte: DoubleArray; private val wpe: DoubleArray
    private class Layer(val ln1w: DoubleArray, val ln1b: DoubleArray, val qkvW: DoubleArray, val qkvB: DoubleArray,
                        val projW: DoubleArray, val projB: DoubleArray, val ln2w: DoubleArray, val ln2b: DoubleArray,
                        val fcW: DoubleArray, val fcB: DoubleArray, val fc2W: DoubleArray, val fc2B: DoubleArray)
    private val layers: List<Layer>
    private val lnfW: DoubleArray; private val lnfB: DoubleArray
    private val pcaMean: Array<DoubleArray>; private val pcaComp: Array<DoubleArray>; private val pcaStd: Array<DoubleArray>
    private val index = HashMap<Char, Int>()

    init {
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val magic = ByteArray(4).also { b.get(it) }
        require(String(magic, Charsets.US_ASCII) == "GPT1") { "not a GPT1 model file" }
        val version = b.int; require(version == 1) { "model.bin version $version" }
        V = b.int; ctx = b.int; d = b.int; nLayer = b.int; nHead = b.int; dff = b.int; nPca = b.int
        val vb = ByteArray(b.int).also { b.get(it) }
        vocab = String(vb, Charsets.UTF_8)
        require(vocab.length == V) { "vocab has ${vocab.length} chars, header says $V" }
        vocab.forEachIndexed { i, c -> index[c] = i }
        fun f(n: Int) = DoubleArray(n) { b.float.toDouble() }
        wte = f(V * d); wpe = f(ctx * d)
        layers = List(nLayer) {
            Layer(f(d), f(d), f(3 * d * d), f(3 * d), f(d * d), f(d), f(d), f(d), f(dff * d), f(dff), f(d * dff), f(d))
        }
        lnfW = f(d); lnfB = f(d)
        val m = ArrayList<DoubleArray>(); val c = ArrayList<DoubleArray>(); val s = ArrayList<DoubleArray>()
        repeat(nLayer + 1) { m += f(d); c += f(nPca * d); s += f(nPca) }
        pcaMean = m.toTypedArray(); pcaComp = c.toTypedArray(); pcaStd = s.toTypedArray()
        trainChars = b.int; valChars = b.int
        reportedValPpl = b.float; reportedUnigramPpl = b.float
        require(!b.hasRemaining()) { "trailing bytes in model.bin" }
    }

    /** Trainable parameters, counted like PyTorch's model.parameters() (head tied to wte). */
    val paramCount: Int get() = V * d + ctx * d + nLayer * (2 * d + 3 * d * d + 3 * d + d * d + d + 2 * d + dff * d + dff + d * dff + d) + 2 * d

    fun idOf(c: Char): Int = index[c] ?: -1
    fun charOf(id: Int): Char = vocab[id]
    fun encode(s: String): IntArray = IntArray(s.length) { idOf(s[it]).also { i -> require(i >= 0) { "'${s[it]}' not in vocab" } } }

    /**
     * Everything the scene shows, for one context of n <= ctx tokens.
     * rows[r][t] = residual stream after r layers (r = 0: embeddings) at position t;
     * att[l][h][i][j] = attention weight of query i on key j (exactly 0 for j > i);
     * logits[t] = next-token scores after position t (only the last position unless allLogits).
     */
    class Trace(val n: Int, val rows: Array<Array<DoubleArray>>, val att: Array<Array<Array<DoubleArray>>>, val logits: Array<DoubleArray?>) {
        val lastLogits: DoubleArray get() = logits[n - 1]!!
    }

    fun forward(ids: IntArray, allLogits: Boolean = false): Trace {
        val n = ids.size
        require(n in 1..ctx) { "context length $n not in 1..$ctx" }
        val hd = d / nHead
        val scale = 1.0 / sqrt(hd.toDouble())
        val rows = Array(nLayer + 1) { Array(n) { DoubleArray(d) } }
        val x = Array(n) { t -> DoubleArray(d) { k -> wte[ids[t] * d + k] + wpe[t * d + k] } }
        for (t in 0 until n) x[t].copyInto(rows[0][t])
        val att = Array(nLayer) { Array(nHead) { Array(n) { DoubleArray(n) } } }
        val h = DoubleArray(d); val qkv = Array(n) { DoubleArray(3 * d) }
        val y = DoubleArray(d); val mid = DoubleArray(dff)
        for ((li, L) in layers.withIndex()) {
            // attention sub-layer
            for (t in 0 until n) { layerNorm(x[t], L.ln1w, L.ln1b, h); linear(L.qkvW, L.qkvB, h, qkv[t], 3 * d, d) }
            val heads = Array(n) { DoubleArray(d) }
            for (hh in 0 until nHead) {
                val qo = hh * hd; val ko = d + hh * hd; val vo = 2 * d + hh * hd
                for (i in 0 until n) {
                    val a = att[li][hh][i]
                    var mx = Double.NEGATIVE_INFINITY
                    for (j in 0..i) {
                        var s = 0.0
                        for (k in 0 until hd) s += qkv[i][qo + k] * qkv[j][ko + k]
                        a[j] = s * scale; if (a[j] > mx) mx = a[j]
                    }
                    var z = 0.0
                    for (j in 0..i) { a[j] = exp(a[j] - mx); z += a[j] }
                    for (j in 0..i) a[j] /= z                      // j > i stays exactly 0: the causal mask
                    for (k in 0 until hd) {
                        var s = 0.0
                        for (j in 0..i) s += a[j] * qkv[j][vo + k]
                        heads[i][qo + k] = s
                    }
                }
            }
            for (t in 0 until n) { linear(L.projW, L.projB, heads[t], y, d, d); for (k in 0 until d) x[t][k] += y[k] }
            // MLP sub-layer
            for (t in 0 until n) {
                layerNorm(x[t], L.ln2w, L.ln2b, h)
                linear(L.fcW, L.fcB, h, mid, dff, d)
                for (k in 0 until dff) mid[k] = gelu(mid[k])
                linear(L.fc2W, L.fc2B, mid, y, d, dff)
                for (k in 0 until d) x[t][k] += y[k]
            }
            for (t in 0 until n) x[t].copyInto(rows[li + 1][t])
        }
        val logits = arrayOfNulls<DoubleArray>(n)
        for (t in (if (allLogits) 0 else n - 1) until n) {
            layerNorm(x[t], lnfW, lnfB, h)
            logits[t] = DoubleArray(V) { v -> var s = 0.0; val o = v * d; for (k in 0 until d) s += wte[o + k] * h[k]; s }
        }
        return Trace(n, rows, att, logits)
    }

    /** The first [nPca] principal coordinates of row [r]'s vector, in units of their training std. */
    fun pca(r: Int, v: DoubleArray, out: FloatArray) {
        val mean = pcaMean[r]; val comp = pcaComp[r]; val std = pcaStd[r]
        for (c in 0 until nPca) {
            var s = 0.0
            for (k in 0 until d) s += (v[k] - mean[k]) * comp[c * d + k]
            out[c] = (s / std[c]).toFloat()
        }
    }

    private fun linear(w: DoubleArray, b: DoubleArray, x: DoubleArray, out: DoubleArray, nOut: Int, nIn: Int) {
        for (o in 0 until nOut) {
            var s = b[o]; val r = o * nIn
            for (k in 0 until nIn) s += w[r + k] * x[k]
            out[o] = s
        }
    }

    private fun layerNorm(x: DoubleArray, w: DoubleArray, b: DoubleArray, out: DoubleArray) {
        var mu = 0.0; for (k in 0 until d) mu += x[k]; mu /= d
        var v = 0.0; for (k in 0 until d) { val e = x[k] - mu; v += e * e }; v /= d     // biased variance, as PyTorch
        val inv = 1.0 / sqrt(v + 1e-5)
        for (k in 0 until d) out[k] = (x[k] - mu) * inv * w[k] + b[k]
    }

    private fun gelu(x: Double) = 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
}
