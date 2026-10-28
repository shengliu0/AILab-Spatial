// Mlp — the 784 -> 64 -> 32 -> 10 multilayer perceptron behind the "神经网络 · 手写数字" demo, inference only.
//
//   h1 = ReLU(W1 x + b1)      x  = 28x28 pixels in [0,1], row-major, row 0 = top (MNIST layout)
//   h2 = ReLU(W2 h1 + b2)
//   z  = W3 h2 + b3           logits
//   p  = softmax(z / T)       T = temperature fitted on held-out data (Guo et al. 2017, "On Calibration of
//                             Modern Neural Networks") so the shown confidence is calibrated on synthetic digits
//
// Trained by tools/train_mlp.py (PyTorch, synthetic digits only: system fonts + pen-stroke templates, heavily
// augmented). Backpropagation: Rumelhart, Hinton & Williams, Nature 323 (1986). The presentation follows
// 3Blue1Brown, "But what is a neural network?" (2017).
//
// Weight file (assets/mlp/mlp_w.bin), little-endian float32, nothing else:
//   [784, 64, 32, 10]  layer sizes, as floats
//   W1 (64 x 784, row-major: W1[j * 784 + i] = weight from pixel i to neuron j), b1[64]
//   W2 (32 x 64), b2[32], W3 (10 x 32), b3[10]
//   act_scale1, act_scale2  99th percentile of h1 / h2 on held-out data (display brightness = a / scale)
//   temperature
// Pure Kotlin (no Android): checked against PyTorch's own outputs by MlpReferenceTest.

package com.bytedance.ailab.core.mlp

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp

class Mlp(
    val nIn: Int, val nH1: Int, val nH2: Int, val nOut: Int,
    val w1: FloatArray, val b1: FloatArray,
    val w2: FloatArray, val b2: FloatArray,
    val w3: FloatArray, val b3: FloatArray,
    /** Typical (99th-percentile) activation of each hidden layer: brightness = a / scale. */
    val actScale1: Float, val actScale2: Float,
    val temperature: Float,
) {
    /** All activations of one forward pass (reusable buffers). */
    class Act(nIn: Int, nH1: Int, nH2: Int, nOut: Int) {
        val x = FloatArray(nIn)
        val h1 = FloatArray(nH1)
        val h2 = FloatArray(nH2)
        val logits = FloatArray(nOut)
        val probs = FloatArray(nOut)
        val predicted: Int get() = argmax(probs)
        val confidence: Float get() = probs[predicted]
        /** Second most likely class. */
        val runnerUp: Int get() { val p = predicted; var b = -1; for (k in probs.indices) if (k != p && (b < 0 || probs[k] > probs[b])) b = k; return b }
        fun copyFrom(o: Act) {
            o.x.copyInto(x); o.h1.copyInto(h1); o.h2.copyInto(h2); o.logits.copyInto(logits); o.probs.copyInto(probs)
        }
    }

    fun newAct() = Act(nIn, nH1, nH2, nOut)

    /** Forward pass of [x] (copied into [out].x). Double accumulation; ~52 k multiply-adds. */
    fun forward(x: FloatArray, out: Act = newAct()): Act {
        require(x.size == nIn) { "input has ${x.size} values, expected $nIn" }
        if (x !== out.x) x.copyInto(out.x)
        dense(w1, b1, out.x, out.h1, nH1, nIn, relu = true)
        dense(w2, b2, out.h1, out.h2, nH2, nH1, relu = true)
        dense(w3, b3, out.h2, out.logits, nOut, nH2, relu = false)
        softmax(out.logits, temperature, out.probs)
        return out
    }

    /** Number of neurons with a > [frac] x the layer's display scale (frac 0 = strictly positive). */
    fun activeCount(a: FloatArray, scale: Float, frac: Float = 0f): Int = a.count { it > frac * scale && it > 0f }

    companion object {
        private fun dense(w: FloatArray, b: FloatArray, a: FloatArray, out: FloatArray, nOut: Int, nIn: Int, relu: Boolean) {
            for (j in 0 until nOut) {
                var s = b[j].toDouble()
                val row = j * nIn
                for (i in 0 until nIn) { val v = a[i]; if (v != 0f) s += w[row + i].toDouble() * v }
                out[j] = if (relu && s < 0.0) 0f else s.toFloat()
            }
        }

        /** p = softmax(z / t), numerically stable. */
        fun softmax(z: FloatArray, t: Float, out: FloatArray) {
            var m = Double.NEGATIVE_INFINITY
            for (v in z) m = maxOf(m, v / t.toDouble())
            var sum = 0.0
            val e = DoubleArray(z.size) { exp(z[it] / t.toDouble() - m) }
            for (v in e) sum += v
            for (k in z.indices) out[k] = (e[k] / sum).toFloat()
        }

        fun argmax(a: FloatArray): Int { var b = 0; for (k in 1 until a.size) if (a[k] > a[b]) b = k; return b }

        /** Parse assets/mlp/mlp_w.bin (layout in the file header comment). */
        fun load(bytes: ByteArray): Mlp {
            val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            fun take(n: Int) = FloatArray(n).also { bb.get(it) }
            val sz = take(4).map { it.toInt() }
            val (nIn, nH1, nH2, nOut) = sz
            require(nIn == 784 && nH1 > 0 && nH2 > 0 && nOut == 10) { "unexpected layer sizes $sz" }
            val w1 = take(nH1 * nIn); val b1 = take(nH1)
            val w2 = take(nH2 * nH1); val b2 = take(nH2)
            val w3 = take(nOut * nH2); val b3 = take(nOut)
            val tail = take(3)
            require(bb.remaining() == 0) { "weight file has ${bb.remaining()} trailing floats" }
            return Mlp(nIn, nH1, nH2, nOut, w1, b1, w2, b2, w3, b3, tail[0], tail[1], tail[2])
        }
    }
}

/** The shipped sample digits (pad strokes) and output-bar glyphs, parsed from the demo's assets. */
object DigitAssets {
    /**
     * assets/mlp/samples.bin, float32: [10], per digit [nStrokes], per stroke [nPts, x0, y0, ...] in pad units
     * (0..1, y down). Returns samples[digit][stroke] = (x0, y0, x1, y1, ...).
     */
    fun samples(bytes: ByteArray): List<List<FloatArray>> {
        val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val n = fb.get().toInt()
        return List(n) {
            val ns = fb.get().toInt()
            List(ns) { val np = fb.get().toInt(); FloatArray(np * 2).also { fb.get(it) } }
        }
    }

    /** assets/mlp/glyphs.bin: 10 x 48 x 48 uint8 alpha, row 0 = top. */
    const val GLYPH = 48
    fun glyphs(bytes: ByteArray): List<ByteArray> = List(10) { d -> bytes.copyOfRange(d * GLYPH * GLYPH, (d + 1) * GLYPH * GLYPH) }
}
