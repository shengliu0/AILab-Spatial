// Net — a small fully connected network with hand-written backpropagation (Rumelhart, Hinton & Williams,
// "Learning representations by back-propagating errors", Nature 323, 1986). Pure Kotlin, double precision.
//
// Notation (one sample, layers 0..L, layer 0 = input):
//   z[l] = W[l-1] a[l-1] + b[l-1]          pre-activation of layer l
//   a[l] = f_l(z[l])                       f = tanh in the hidden layers, sigmoid at the output
//   loss  BCE   L = -(y ln p + (1-y) ln(1-p)),  p = a[L]            (default)
//         SSE   L = 1/2 sum (a[L] - y)^2                             (only for the textbook check)
// Backward (the chain rule, applied once per layer from the output back):
//   delta[L] = dL/dz[L]   = p - y                       (BCE + sigmoid: the sigmoid slope cancels)
//                         = (a - y) * f'(z)              (SSE)
//   delta[l] = f'(z[l]) * W[l]^T delta[l+1]             "my blame = my slope x the weighted blame downstream"
//   dL/dW[l][j,i] = delta[l+1][j] * a[l][i]             "a weight's gradient = (activity at its input) x (blame at its output)"
//   dL/db[l][j]   = delta[l+1][j]
// SGD: w <- w - lr * dL/dw.
// Checked by core/backprop tests: finite differences (every weight, rel. err < 1e-6), the textbook 2-2-2 example
// (Mazur 2015), a hand-computed 2-2-1 tanh/BCE step, and PyTorch autograd (tools/ref_backprop.py).

package com.bytedance.ailab.core.backprop

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.math.tanh

enum class Act {
    TANH, SIGMOID;
    fun f(z: Double): Double = when (this) { TANH -> tanh(z); SIGMOID -> sigmoid(z) }
    /** f'(z) written through the activation a = f(z) (tanh' = 1 - a^2, sigmoid' = a (1 - a)). */
    fun slopeFromA(a: Double): Double = when (this) { TANH -> 1.0 - a * a; SIGMOID -> a * (1.0 - a) }
}

enum class Loss { BCE, HALF_SSE }

fun sigmoid(z: Double): Double = if (z >= 0) 1.0 / (1.0 + exp(-z)) else exp(z).let { it / (1.0 + it) }

/** Deterministic, platform-independent RNG (SplitMix64, Steele et al. 2014): the same seed gives the same net everywhere. */
class SplitMix(seed: Long) {
    private var s = seed
    fun nextLong(): Long {
        s += 0x9E3779B97F4A7C15uL.toLong()
        var z = s
        z = (z xor (z ushr 30)) * 0xBF58476D1CE4E5B9uL.toLong()
        z = (z xor (z ushr 27)) * 0x94D049BB133111EBuL.toLong()
        return z xor (z ushr 31)
    }
    /** Uniform in [0, 1). */
    fun nextDouble(): Double = (nextLong() ushr 11) * (1.0 / (1L shl 53))
    fun nextInt(n: Int): Int = ((nextLong() ushr 33) % n).toInt()
    /** Standard normal (Box-Muller). */
    fun nextGaussian(): Double {
        val u = nextDouble().coerceAtLeast(1e-300); val v = nextDouble()
        return sqrt(-2.0 * ln(u)) * kotlin.math.cos(2.0 * Math.PI * v)
    }
}

class Net(
    val sizes: IntArray,
    val hidden: Act = Act.TANH,
    val out: Act = Act.SIGMOID,
    val loss: Loss = Loss.BCE,
) {
    /** Number of weight layers. */
    val nL = sizes.size - 1
    /** w[l][j * sizes[l] + i] = weight from neuron i of layer l to neuron j of layer l + 1. */
    val w: Array<DoubleArray> = Array(nL) { DoubleArray(sizes[it] * sizes[it + 1]) }
    val b: Array<DoubleArray> = Array(nL) { DoubleArray(sizes[it + 1]) }
    val paramCount get() = (0 until nL).sumOf { w[it].size + b[it].size }

    init { require(loss != Loss.BCE || (out == Act.SIGMOID && sizes.last() == 1)) { "BCE needs one sigmoid output" } }

    fun act(layer: Int): Act = if (layer == nL) out else hidden

    /** Xavier/Glorot uniform weights U(-sqrt(6/(in+out)), +sqrt(6/(in+out))), zero biases. */
    fun init(seed: Long): Net {
        val r = SplitMix(seed)
        for (l in 0 until nL) {
            val lim = sqrt(6.0 / (sizes[l] + sizes[l + 1]))
            for (k in w[l].indices) w[l][k] = (2.0 * r.nextDouble() - 1.0) * lim
            b[l].fill(0.0)
        }
        return this
    }

    fun copyFrom(o: Net) { for (l in 0 until nL) { o.w[l].copyInto(w[l]); o.b[l].copyInto(b[l]) } }

    /** Everything one sample leaves behind: activations, slopes, deltas and gradients (reused, no allocation). */
    inner class Trace {
        val a: Array<DoubleArray> = Array(nL + 1) { DoubleArray(sizes[it]) }
        /** delta[l] = dL/dz[l]; delta[0] (the inputs) is filled too, for display. */
        val delta: Array<DoubleArray> = Array(nL + 1) { DoubleArray(sizes[it]) }
        val gw: Array<DoubleArray> = Array(nL) { DoubleArray(w[it].size) }
        val gb: Array<DoubleArray> = Array(nL) { DoubleArray(b[it].size) }
        var lossValue = 0.0
        val output get() = a[nL][0]
    }
    fun newTrace() = Trace()

    /** Forward pass; returns the first output. */
    fun forward(x: DoubleArray, t: Trace): Double {
        x.copyInto(t.a[0], 0, 0, sizes[0])
        for (l in 0 until nL) {
            val nIn = sizes[l]; val nOut = sizes[l + 1]; val f = act(l + 1); val ai = t.a[l]; val ao = t.a[l + 1]; val wl = w[l]
            for (j in 0 until nOut) {
                var s = b[l][j]
                for (i in 0 until nIn) s += wl[j * nIn + i] * ai[i]
                ao[j] = f.f(s)
            }
        }
        return t.a[nL][0]
    }

    fun lossOf(a: DoubleArray, y: DoubleArray): Double = when (loss) {
        Loss.BCE -> { val p = a[0].coerceIn(1e-15, 1 - 1e-15); -(y[0] * ln(p) + (1 - y[0]) * ln(1 - p)) }
        Loss.HALF_SSE -> { var s = 0.0; for (k in a.indices) s += 0.5 * (a[k] - y[k]) * (a[k] - y[k]); s }
    }

    /** Backward pass for the sample last run through [forward] with target [y]: fills delta, gw, gb, lossValue. */
    fun backward(t: Trace, y: DoubleArray) {
        val aL = t.a[nL]; val dL = t.delta[nL]
        t.lossValue = lossOf(aL, y)
        for (k in aL.indices) dL[k] = when (loss) {
            Loss.BCE -> aL[k] - y[k]                                   // (p-y)/(p(1-p)) * p(1-p)
            Loss.HALF_SSE -> (aL[k] - y[k]) * out.slopeFromA(aL[k])
        }
        for (l in nL - 1 downTo 0) {
            val nIn = sizes[l]; val nOut = sizes[l + 1]; val wl = w[l]; val dOut = t.delta[l + 1]; val ai = t.a[l]
            for (j in 0 until nOut) {
                t.gb[l][j] = dOut[j]
                for (i in 0 until nIn) t.gw[l][j * nIn + i] = dOut[j] * ai[i]
            }
            // blame for layer l: slope x weighted blame downstream (inputs get the plain weighted sum: no activation)
            val dIn = t.delta[l]
            for (i in 0 until nIn) {
                var s = 0.0
                for (j in 0 until nOut) s += wl[j * nIn + i] * dOut[j]
                dIn[i] = if (l == 0) s else act(l).slopeFromA(ai[i]) * s
            }
        }
    }

    /** f'(z) of neuron [i] in layer [l] for the sample in [t] (1 for inputs). */
    fun slope(t: Trace, l: Int, i: Int): Double = if (l == 0) 1.0 else act(l).slopeFromA(t.a[l][i])

    /** sum_j w[l][j,i] * delta[l+1][j] — the weighted blame arriving at neuron i of layer l. */
    fun blameSum(t: Trace, l: Int, i: Int): Double {
        if (l >= nL) return 0.0
        var s = 0.0
        for (j in 0 until sizes[l + 1]) s += w[l][j * sizes[l] + i] * t.delta[l + 1][j]
        return s
    }

    /** One SGD step with the gradients in [t]. */
    fun sgd(t: Trace, lr: Double) {
        for (l in 0 until nL) {
            val wl = w[l]; val g = t.gw[l]
            for (k in wl.indices) wl[k] -= lr * g[k]
            val bl = b[l]; val gb = t.gb[l]
            for (k in bl.indices) bl[k] -= lr * gb[k]
        }
    }

    /** Fast single-output evaluation for the 2-D input plane (no Trace; [buf] = scratch of size >= 2 * max width). */
    fun predict(x1: Double, x2: Double, buf: DoubleArray): Double {
        var cur = 0; val maxW = buf.size / 2
        buf[0] = x1; buf[1] = x2
        for (l in 0 until nL) {
            val nIn = sizes[l]; val nOut = sizes[l + 1]; val f = act(l + 1); val wl = w[l]
            val src = cur * maxW; val dst = (1 - cur) * maxW
            for (j in 0 until nOut) {
                var s = b[l][j]
                for (i in 0 until nIn) s += wl[j * nIn + i] * buf[src + i]
                buf[dst + j] = f.f(s)
            }
            cur = 1 - cur
        }
        return buf[cur * maxW]
    }
    fun predictBuf() = DoubleArray(2 * sizes.max())
}
