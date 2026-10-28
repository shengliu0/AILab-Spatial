// ResnetCore — the measured plain-vs-residual experiment (tools/train_resnet.py) as pure Kotlin data + a live
// forward pass of the shipped trained nets. No Android imports.
//
// The experiment (He, Zhang, Ren & Sun 2015, "Deep Residual Learning for Image Recognition", §4.1 "degradation"):
// MLPs of D = 4/8/16/32/56 weight layers, width 32, ReLU, no normalisation, two-spirals (2000 points), SGD
// momentum 0.9, lr 0.01, batch 100, 60 epochs, seeded. Residual block: h <- relu(h + W2 relu(W1 h + b1) + b2);
// plain block: h <- relu(W2 relu(W1 h + b1) + b2). Logged per epoch (0 = at init): full-batch training loss,
// accuracy and ||dL/dW_l|| (Frobenius) for every weight layer l (0 = stem next to the input, D-1 = head).
//
// File layouts (little-endian) are documented in tools/train_resnet.py.

package com.bytedance.ailab.core.resnet

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max

/** One recorded training run. [g] is [epoch][layer]. */
class Run(val residual: Boolean, val depth: Int, val loss: FloatArray, val acc: FloatArray, val g: Array<FloatArray>) {
    val epochs get() = loss.size - 1
    /** Linear interpolation of a per-epoch series at fractional epoch [e]. */
    fun at(s: FloatArray, e: Float): Float {
        val c = e.coerceIn(0f, epochs.toFloat()); val i = c.toInt().coerceAtMost(epochs - 1); val f = c - i
        return s[i] + (s[i + 1] - s[i]) * f
    }
    fun lossAt(e: Float) = at(loss, e)
    fun accAt(e: Float) = at(acc, e)
    /** Gradient norm of [layer] at fractional epoch [e], interpolated in log space. */
    fun gradAt(layer: Int, e: Float): Float {
        val c = e.coerceIn(0f, epochs.toFloat()); val i = c.toInt().coerceAtMost(epochs - 1); val f = c - i
        val a = log10(max(g[i][layer], 1e-12f)); val b = log10(max(g[i + 1][layer], 1e-12f))
        return Math.pow(10.0, (a + (b - a) * f).toDouble()).toFloat()
    }
}

class ResnetLogs(bytes: ByteArray) {
    val depths: IntArray
    val epochs: Int
    private val runs = HashMap<Pair<Boolean, Int>, Run>()

    init {
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val magic = String(bytes, 0, 4, Charsets.US_ASCII); require(magic == "RNL1") { "bad logs magic $magic" }
        b.position(4)
        val e = b.int; val k = b.int
        depths = IntArray(k) { b.int }; epochs = e - 1
        for (res in listOf(false, true)) for (d in depths) {
            val loss = FloatArray(e) { b.float }; val acc = FloatArray(e) { b.float }
            val g = Array(e) { FloatArray(d) { b.float } }
            runs[res to d] = Run(res, d, loss, acc, g)
        }
        require(!b.hasRemaining()) { "trailing bytes in logs" }
    }

    fun run(residual: Boolean, depth: Int): Run = runs[residual to depth] ?: error("no run ${if (residual) "residual" else "plain"} $depth")
    val all: Collection<Run> get() = runs.values

    companion object { const val PATH = "resnet/logs.bin" }
}

/** A trained net (weights.bin): live forward pass. [w] = per layer W[out][in] row-major, [b] = biases. */
class MlpNet(val residual: Boolean, val depth: Int, val sizes: IntArray, val w: Array<FloatArray>, val b: Array<FloatArray>) {
    private val width = sizes[1]
    private val h = FloatArray(width); private val t = FloatArray(width); private val f = FloatArray(width)

    private fun lin(l: Int, x: FloatArray, out: FloatArray) {
        val nin = sizes[l]; val nout = sizes[l + 1]; val W = w[l]; val B = b[l]
        for (j in 0 until nout) { var s = B[j]; val o = j * nin; for (i in 0 until nin) s += W[o + i] * x[i]; out[j] = s }
    }

    /** Logits (class 0, class 1) for the input point (x1, x2). Not thread-safe (scratch buffers). */
    fun forward(x1: Float, x2: Float, out: FloatArray = FloatArray(2)): FloatArray {
        val x = floatArrayOf(x1, x2)
        lin(0, x, h); for (j in 0 until width) h[j] = max(h[j], 0f)
        val nb = (depth - 2) / 2
        for (blk in 0 until nb) {
            lin(1 + 2 * blk, h, t); for (j in 0 until width) t[j] = max(t[j], 0f)
            lin(2 + 2 * blk, t, f)
            if (residual) for (j in 0 until width) h[j] = max(h[j] + f[j], 0f)
            else for (j in 0 until width) h[j] = max(f[j], 0f)
        }
        lin(depth - 1, h, out)
        return out
    }

    /** P(class 1) = softmax(logits)[1]. */
    fun p1(x1: Float, x2: Float, scratch: FloatArray = FloatArray(2)): Float {
        val o = forward(x1, x2, scratch); val d = (o[0] - o[1]).toDouble()
        return (1.0 / (1.0 + Math.exp(d))).toFloat()
    }

    companion object {
        const val PATH = "resnet/weights.bin"

        /** All nets in the file, keyed (residual, depth). */
        fun loadAll(bytes: ByteArray): Map<Pair<Boolean, Int>, MlpNet> {
            val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN); b.position(4)
            val k = b.int; val depths = IntArray(k) { b.int }; val width = b.int
            val out = LinkedHashMap<Pair<Boolean, Int>, MlpNet>()
            for (res in listOf(false, true)) for (d in depths) {
                val sizes = IntArray(d + 1) { if (it == 0) 2 else if (it == d) 2 else width }
                val w = arrayOfNulls<FloatArray>(d); val bb = arrayOfNulls<FloatArray>(d)
                for (l in 0 until d) { w[l] = FloatArray(sizes[l] * sizes[l + 1]) { b.float }; bb[l] = FloatArray(sizes[l + 1]) { b.float } }
                out[res to d] = MlpNet(res, d, sizes, w.requireNoNulls(), bb.requireNoNulls())
            }
            require(!b.hasRemaining()) { "trailing bytes in weights" }
            return out
        }
    }
}

/** PyTorch reference logits (tests.bin): inputs [x] (N x 2) and per (residual, depth) logits (N x 2). */
class ResnetRefs(bytes: ByteArray, depths: IntArray) {
    val x: FloatArray
    val n: Int
    val logits = LinkedHashMap<Pair<Boolean, Int>, FloatArray>()
    init {
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        require(String(bytes, 0, 4, Charsets.US_ASCII) == "RNT1"); b.position(4)
        n = b.int; x = FloatArray(n * 2) { b.float }
        for (res in listOf(false, true)) for (d in depths) logits[res to d] = FloatArray(n * 2) { b.float }
        require(!b.hasRemaining())
    }
    companion object { const val PATH = "resnet/tests.bin" }
}

/** The two-spirals training set, re-generated for display only (the tests compare shape, not RNG). */
object Spirals {
    /** [n] points per arm, noiseless: (x, y) on arm [cls] (0 or 1) — the same curve as tools/train_resnet.py. */
    fun arm(cls: Int, n: Int): FloatArray {
        val turns = 2.0; val tMax = turns * 2 * Math.PI + 0.5
        val out = FloatArray(n * 2)
        for (i in 0 until n) {
            val t = 0.5 + (tMax - 0.5) * i / (n - 1); val r = t / tMax; val a = t + Math.PI * cls
            out[i * 2] = (r * Math.cos(a)).toFloat(); out[i * 2 + 1] = (r * Math.sin(a)).toFloat()
        }
        return out
    }
    /** Chance-level cross-entropy for 2 balanced classes = ln 2. */
    val CHANCE_LOSS = ln(2.0).toFloat()
}
