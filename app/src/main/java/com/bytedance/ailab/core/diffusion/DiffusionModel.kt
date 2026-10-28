// DiffusionModel — the trained mini DDPM denoiser (Ho, Jain & Abbeel 2020), pure Kotlin inference.
//
// The weights come from tools/train_diffusion.py (PyTorch, synthetic 2-D point clouds: spiral, two moons,
// the glyph 光) and ship as assets/diffusion/model.bin; the file layout is documented in that script.
// ONE conditional network predicts the noise eps_hat(x_t, t, dataset):
//   f   = [t/T, sin(pi 2^k t/T), cos(pi 2^k t/T)]_k           time features (float64 -> float32, as in Python)
//   e   = SiLU(A2 SiLU(A1 [f; onehot(d)] + a1) + a2)           conditioning embedding
//   h1  = SiLU(W1 x  + U1 e + b1)
//   h2  = SiLU(W2 h1 + U2 e + b2)
//   h3  = SiLU(W3 h2 + U3 e + b3)
//   eps = Wo h3 + bo
// The terms U_l e + b_l depend only on (t, dataset) -> [cond] computes them once per sampling step, and a
// point then costs 2H + 2H^2 + 2H multiply-adds (H = 64: 8.4k). Points are evaluated in chunks with the
// point index innermost, so the JIT can vectorise the multiply-add loops.
//
// Forward process (variance preserving): q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I),
// abar_t = 1 - (1 - abar_T)(t/T)^2  (noise level sigma_t = sqrt(1 - abar_t) grows linearly in t).
// abar is read from the file (exactly what training used) and DiffusionCoreTest re-derives it.

package com.bytedance.ailab.core.diffusion

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin
import kotlin.math.sqrt

class DiffusionModel(bytes: ByteArray) {
    val T: Int; val K: Int; val E: Int; val H: Int
    /** Number of datasets the network is conditioned on (0 spiral, 1 moons, 2 glyph). */
    val D: Int
    /** Clean reference samples per dataset (also the "data" the app shows and the quality metric's target). */
    val nRef: Int
    /** The first [nCov] reference points are the coverage set of [Quality]. */
    val nCov: Int
    /** abar_t, t = 0..T (abar_0 = 1). */
    val abar: FloatArray
    /** ref[d] = x0, y0, x1, y1, ... (data units: each cloud has std 1). */
    val ref: Array<FloatArray>
    /** hue[d][i] in 0..1: a colour key along the shape (arc position of the spiral, which moon, left-right in 光). */
    val hue: Array<FloatArray>

    private val nIn: Int
    private val A1: FloatArray; private val a1: FloatArray; private val A2: FloatArray; private val a2: FloatArray
    private val W1: FloatArray; private val U1: FloatArray; private val b1: FloatArray
    private val W2: FloatArray; private val U2: FloatArray; private val b2: FloatArray
    private val W3: FloatArray; private val U3: FloatArray; private val b3: FloatArray
    private val Wo: FloatArray; private val bo: FloatArray

    init {
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val magic = String(ByteArray(4).also { b.get(it) }, Charsets.US_ASCII)
        require(magic == "DIF1") { "not a diffusion model file: '$magic'" }
        T = b.int; K = b.int; E = b.int; H = b.int; D = b.int; nRef = b.int; nCov = b.int
        nIn = 1 + 2 * K + D
        fun f(n: Int) = FloatArray(n).also { for (i in 0 until n) it[i] = b.float }
        abar = f(T + 1)
        A1 = f(E * nIn); a1 = f(E); A2 = f(E * E); a2 = f(E)
        W1 = f(H * 2); U1 = f(H * E); b1 = f(H)
        W2 = f(H * H); U2 = f(H * E); b2 = f(H)
        W3 = f(H * H); U3 = f(H * E); b3 = f(H)
        Wo = f(2 * H); bo = f(2)
        ref = Array(D) { FloatArray(0) }; hue = Array(D) { FloatArray(0) }
        for (d in 0 until D) { ref[d] = f(nRef * 2); hue[d] = f(nRef) }
        require(!b.hasRemaining()) { "model.bin has ${b.remaining()} trailing bytes" }
    }

    /** Noise level sigma_t = sqrt(1 - abar_t). */
    fun sigma(t: Int): Float = sqrt((1.0 - abar[t]).coerceAtLeast(0.0)).toFloat()
    /** beta_t = 1 - abar_t / abar_{t-1}, t = 1..T (the single-step forward variance). */
    fun beta(t: Int): Double = 1.0 - abar[t].toDouble() / abar[t - 1].toDouble()

    /** Time features exactly as tools/train_diffusion.py computes them (float64, then float32). */
    fun features(t: Int): FloatArray {
        val s = t.toDouble() / T
        val out = FloatArray(1 + 2 * K)
        out[0] = s.toFloat()
        var w = Math.PI
        for (k in 0 until K) { out[1 + k] = sin(s * w).toFloat(); out[1 + K + k] = cos(s * w).toFloat(); w *= 2 }
        return out
    }

    /** The per-(t, dataset) part of every layer: [U1 e + b1 | U2 e + b2 | U3 e + b3] (3H floats). */
    fun cond(d: Int, t: Int): FloatArray {
        val f = features(t)
        val inp = FloatArray(nIn); System.arraycopy(f, 0, inp, 0, f.size); inp[f.size + d] = 1f
        val e1 = FloatArray(E) { j -> var s = a1[j]; for (i in 0 until nIn) s += A1[j * nIn + i] * inp[i]; silu(s) }
        val e = FloatArray(E) { j -> var s = a2[j]; for (i in 0 until E) s += A2[j * E + i] * e1[i]; silu(s) }
        val c = FloatArray(3 * H)
        for ((l, UB) in listOf(U1 to b1, U2 to b2, U3 to b3).withIndex()) {
            val (U, bb) = UB
            for (j in 0 until H) { var s = bb[j]; for (i in 0 until E) s += U[j * E + i] * e[i]; c[l * H + j] = s }
        }
        return c
    }

    /** Scratch for [eps]: activations of one chunk, [H][CHUNK] with the point index innermost. */
    inner class Scratch {
        val h1 = FloatArray(H * CHUNK); val h2 = FloatArray(H * CHUNK)
    }

    /**
     * eps_hat for points xy[2*(i0+k)], k < n (n <= CHUNK), written to out[2*(i0+k)] — the conditioning [c]
     * from [cond]. Pure multiply-adds; allocation-free given a [Scratch].
     */
    fun eps(c: FloatArray, xy: FloatArray, i0: Int, n: Int, out: FloatArray, s: Scratch) {
        require(n in 0..CHUNK)
        val h1 = s.h1; val h2 = s.h2
        // layer 1: h1[j][k] = SiLU(W1[j,0] x_k + W1[j,1] y_k + c1[j])
        for (j in 0 until H) {
            val w0 = W1[j * 2]; val w1 = W1[j * 2 + 1]; val cj = c[j]; val o = j * CHUNK
            for (k in 0 until n) h1[o + k] = silu(w0 * xy[2 * (i0 + k)] + w1 * xy[2 * (i0 + k) + 1] + cj)
        }
        dense(W2, h1, h2, c, H, n)          // h2 = SiLU(W2 h1 + c2)
        dense(W3, h2, h1, c, 2 * H, n)      // h3 = SiLU(W3 h2 + c3)   (into h1's buffer)
        // output: eps_k = Wo h3_k + bo
        for (k in 0 until n) { out[2 * (i0 + k)] = bo[0]; out[2 * (i0 + k) + 1] = bo[1] }
        for (i in 0 until H) {
            val w0 = Wo[i]; val w1 = Wo[H + i]; val o = i * CHUNK
            for (k in 0 until n) { val h = h1[o + k]; out[2 * (i0 + k)] += w0 * h; out[2 * (i0 + k) + 1] += w1 * h }
        }
    }

    /** dst[j][k] = SiLU(sum_i W[j,i] src[i][k] + c[cOff + j]), j < H, k < n. */
    private fun dense(W: FloatArray, src: FloatArray, dst: FloatArray, c: FloatArray, cOff: Int, n: Int) {
        for (j in 0 until H) {
            val o = j * CHUNK; val cj = c[cOff + j]
            for (k in 0 until n) dst[o + k] = cj
            val wr = j * H
            for (i in 0 until H) {
                val w = W[wr + i]; val si = i * CHUNK
                for (k in 0 until n) dst[o + k] += w * src[si + k]
            }
            for (k in 0 until n) dst[o + k] = silu(dst[o + k])
        }
    }

    /** Convenience (tests, arrows): eps_hat for every point in xy (n = xy.size / 2). */
    fun epsAll(d: Int, t: Int, xy: FloatArray): FloatArray {
        val c = cond(d, t); val out = FloatArray(xy.size); val s = Scratch()
        val n = xy.size / 2
        var i = 0
        while (i < n) { val m = minOf(CHUNK, n - i); eps(c, xy, i, m, out, s); i += m }
        return out
    }

    /** Multiply-adds per point per network call (after the per-step conditioning). */
    val macsPerPoint: Int get() = 2 * H + 2 * H * H + 2 * H

    companion object {
        const val CHUNK = 64
        fun silu(x: Float): Float = x / (1f + exp(-x))
    }
}
