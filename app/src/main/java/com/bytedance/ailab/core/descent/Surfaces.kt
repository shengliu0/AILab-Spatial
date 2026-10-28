// Loss landscapes for the gradient-descent demo: f(x, y) with ANALYTIC gradients (checked against finite
// differences in DescentCoreTest), their known minima, a default start and a display height mapping.
//
// Pure Kotlin (no Android). Formulas and constants are mirrored 1:1 in tools/train_descent.py, which runs
// torch.optim on the same functions to produce the reference trajectories the tests compare against.

package com.bytedance.ailab.core.descent

import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

/** A 2-D loss landscape on the box [x0, x1] × [y0, y1] (the part that is drawn). */
abstract class LossSurface(
    /** Stable key (node names, tests). */
    val key: String,
    val x0: Double, val x1: Double, val y0: Double, val y1: Double,
    /** Default start (where the race begins until the user drags the start marker). */
    val startX: Double, val startY: Double,
    /** Learning rate the surface opens with (one of [LR_STEPS]). */
    val defaultLr: Double,
    /** Display: height ∝ ln(1 + (f − fMin)/dispS) — near the bottom linear, compressive at the top. */
    val dispS: Double,
    /** Gradient-noise size at noise = 1 (typical |∇f| on the route). */
    val noiseScale: Double,
) {
    abstract fun f(x: Double, y: Double): Double
    /** ∇f into [out] (size 2). */
    abstract fun grad(x: Double, y: Double, out: DoubleArray)
    /** The global minima (all with f = [fMin]). */
    abstract val minima: List<DoubleArray>
    open val fMin: Double get() = 0.0

    fun inside(x: Double, y: Double) = x >= x0 && x <= x1 && y >= y0 && y <= y1

    /** Largest f on the drawn box (sampled 129×129 incl. the border) — the top of the display scale. */
    val fTop: Double by lazy {
        var m = 0.0
        for (i in 0..128) for (j in 0..128) m = max(m, f(x0 + (x1 - x0) * i / 128, y0 + (y1 - y0) * j / 128))
        m
    }

    /** Height on the display, 0 (= fMin) .. 1 (= fTop). Monotone in f, so "downhill" on screen = downhill in f. */
    fun display(fv: Double): Double {
        val d = ln(1.0 + max(0.0, fv - fMin) / dispS) / ln(1.0 + (fTop - fMin) / dispS)
        return d.coerceIn(0.0, 1.0)
    }

    /**
     * The gradient an optimiser actually gets: exact when [noise] = 0, otherwise ∇f + noise·noiseScale·ξ with
     * ξ ~ N(0, I) from [rng] (a stand-in for minibatch noise). [Surfaces.LOGISTIC] overrides this with REAL
     * minibatches of its dataset.
     */
    open fun noisyGrad(x: Double, y: Double, noise: Double, rng: Rng, out: DoubleArray) {
        grad(x, y, out)
        if (noise > 0.0) { out[0] += noise * noiseScale * rng.gaussian(); out[1] += noise * noiseScale * rng.gaussian() }
    }

    /** Hessian [fxx, fxy, fyx, fyy] by central differences of the analytic gradient (h = 1e-5 · box size). */
    fun hessian(x: Double, y: Double): DoubleArray {
        val h = 1e-5 * (x1 - x0)
        val a = DoubleArray(2); val b = DoubleArray(2); val c = DoubleArray(2); val d = DoubleArray(2)
        grad(x + h, y, a); grad(x - h, y, b); grad(x, y + h, c); grad(x, y - h, d)
        return doubleArrayOf((a[0] - b[0]) / (2 * h), (a[1] - b[1]) / (2 * h), (c[0] - d[0]) / (2 * h), (c[1] - d[1]) / (2 * h))
    }

    /** Largest Hessian eigenvalue at (x, y): plain GD near there is stable iff η < 2/λmax. */
    fun lambdaMax(x: Double, y: Double): Double {
        val m = hessian(x, y)
        val p = (m[0] + m[3]) / 2; val q = (m[1] + m[2]) / 2
        val r = sqrt(((m[0] - m[3]) / 2).let { it * it } + q * q)
        return p + r
    }

    /** The minimum closest to (x, y). */
    fun nearestMin(x: Double, y: Double): DoubleArray =
        minima.minBy { (it[0] - x) * (it[0] - x) + (it[1] - y) * (it[1] - y) }
}

/** The learning-rate slider's positions: a 1-1.5-2-3-5-7 log ladder from 1e-3 to 1 (gaze-friendly snapping). */
val LR_STEPS = doubleArrayOf(0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07,
    0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0)

fun lrIndexOf(lr: Double): Int = LR_STEPS.indices.minBy { kotlin.math.abs(ln(LR_STEPS[it] / lr)) }

/**
 * Logistic-regression data (tools/train_descent.py, seed 1847): 40 points (x1, x2, label ±1), two Gaussian
 * blobs; feature 2 deliberately NOT normalised (×2.5), which is what turns the loss into a ravine.
 */
object LogisticData {
    val ROWS = doubleArrayOf(
        1.33, 4.33, 1.0,   1.09, 1.95, 1.0,   3.70, -3.94, 1.0,   0.08, 4.68, 1.0,
        0.92, -1.14, 1.0,   2.10, 2.54, 1.0,   0.35, 2.34, 1.0,   1.78, 1.36, 1.0,
        0.25, 5.00, 1.0,   -0.08, -2.96, 1.0,   2.37, 0.60, 1.0,   -0.64, 0.94, 1.0,
        -0.26, 0.45, 1.0,   1.62, 0.96, 1.0,   1.13, -1.78, 1.0,   0.52, -0.71, 1.0,
        0.52, -0.34, 1.0,   2.10, 0.96, 1.0,   2.07, 2.44, 1.0,   1.22, 1.36, 1.0,
        -1.26, -0.14, -1.0,   -2.14, 1.19, -1.0,   -2.36, 0.44, -1.0,   -1.58, -3.33, -1.0,
        -0.05, -1.42, -1.0,   -1.39, 1.54, -1.0,   0.03, -1.82, -1.0,   -0.32, -1.58, -1.0,
        -0.46, -1.24, -1.0,   -0.73, 2.12, -1.0,   -1.53, -4.53, -1.0,   -2.23, 0.67, -1.0,
        -0.10, -3.06, -1.0,   -0.35, 0.42, -1.0,   -1.34, 1.03, -1.0,   -2.26, 3.48, -1.0,
        -0.36, 2.21, -1.0,   -0.21, 0.87, -1.0,   0.32, 0.39, -1.0,   -1.29, -2.02, -1.0,
    )
    val N = ROWS.size / 3
    fun x1(i: Int) = ROWS[i * 3]
    fun x2(i: Int) = ROWS[i * 3 + 1]
    fun label(i: Int) = ROWS[i * 3 + 2]
    /** L2 penalty λ/2·|w|² (keeps the minimum finite: the blobs overlap, but only just). */
    const val LAMBDA = 0.05
    /** Smallest minibatch at noise = 1. */
    const val MIN_BATCH = 4
}

/** softplus(z) = ln(1 + e^z), stable for large |z|. */
internal fun softplus(z: Double): Double = if (z > 30) z else if (z < -30) exp(z) else ln(1.0 + exp(z))
/** σ(z) = 1/(1 + e^−z), stable. */
internal fun sigmoid(z: Double): Double = if (z >= 0) 1.0 / (1.0 + exp(-z)) else exp(z).let { it / (1.0 + it) }

object Surfaces {
    val BOWL: Bowl get() = Bowl
    val ROSENBROCK: LossSurface get() = Rosenbrock
    val HIMMELBLAU: LossSurface get() = Himmelblau
    val SADDLE: LossSurface get() = Saddle
    val LOGISTIC: Logistic get() = Logistic

    /** Shelf order. */
    val all: List<LossSurface> = listOf(Bowl, Rosenbrock, Himmelblau, Saddle, Logistic)
}

/**
 * Anisotropic quadratic bowl ½(λ₁u² + λ₂v²), (u, v) = (x, y) rotated by 70°, λ = (1, 10): the classic ravine,
 * its gentle floor running away from the viewer and its steep walls left and right (the zig-zag crosses the view).
 * GD error along each eigen-direction scales by (1 − ηλ) per step, so it converges iff η < 2/λmax = 0.2.
 */
object Bowl : LossSurface("bowl", -2.0, 2.0, -2.0, 2.0, -1.2, -1.7, 0.15, 30.0, 4.0) {
    const val L1 = 1.0
    const val L2 = 10.0
    val C = cos(Math.toRadians(70.0))
    val S = sin(Math.toRadians(70.0))
    override fun f(x: Double, y: Double): Double { val u = C * x + S * y; val v = -S * x + C * y; return 0.5 * (L1 * u * u + L2 * v * v) }
    override fun grad(x: Double, y: Double, out: DoubleArray) {
        val u = C * x + S * y; val v = -S * x + C * y
        out[0] = L1 * u * C - L2 * v * S; out[1] = L1 * u * S + L2 * v * C
    }
    override val minima = listOf(doubleArrayOf(0.0, 0.0))
}

/** Rosenbrock (1960): (1 − x)² + 100(y − x²)². A long, curved, almost flat valley; minimum (1, 1), f = 0. */
object Rosenbrock : LossSurface("rosen", -2.0, 2.0, -1.0, 3.0, -1.0, 0.7, 0.05, 1.0, 40.0) {
    override fun f(x: Double, y: Double): Double { val a = 1 - x; val b = y - x * x; return a * a + 100 * b * b }
    override fun grad(x: Double, y: Double, out: DoubleArray) {
        val b = y - x * x
        out[0] = -2 * (1 - x) - 400 * x * b; out[1] = 200 * b
    }
    override val minima = listOf(doubleArrayOf(1.0, 1.0))
}

/**
 * Himmelblau (1972): (x² + y − 11)² + (x + y² − 7)². Four minima with f = 0: (3, 2) exactly, and
 * (−2.805118, 3.131312), (−3.779310, −3.283186), (3.584428, −1.848126) (stationary points refined by Newton,
 * tools/train_descent.py; re-checked in DescentCoreTest). Where you start — and how you move — decides the valley.
 */
object Himmelblau : LossSurface("himmel", -5.0, 5.0, -5.0, 5.0, 0.0, 0.0, 0.01, 5.0, 40.0) {
    override fun f(x: Double, y: Double): Double { val a = x * x + y - 11; val b = x + y * y - 7; return a * a + b * b }
    override fun grad(x: Double, y: Double, out: DoubleArray) {
        val a = x * x + y - 11; val b = x + y * y - 7
        out[0] = 4 * x * a + 2 * b; out[1] = 2 * a + 4 * y * b
    }
    override val minima = listOf(doubleArrayOf(3.0, 2.0), doubleArrayOf(-2.805118086952745, 3.131312518250573),
        doubleArrayOf(-3.779310253377747, -3.283185991286170), doubleArrayOf(3.584428340330492, -1.848126526964404))
}

/**
 * A mountain pass: x² + ¼(y² − 1)². Saddle point at (0, 0) (∇f = 0, Hessian diag(2, −1): a minimum along x,
 * a maximum along y), two minima (0, ±1) with f = 0. Near the saddle plain GD's escape factor is only
 * (1 + η) per step; heavy-ball momentum's is the root of r² − (1 + β + η)r + β = 0 (much larger).
 */
object Saddle : LossSurface("saddle", -2.0, 2.0, -2.0, 2.0, -1.8, 0.01, 0.05, 0.2, 1.0) {
    override fun f(x: Double, y: Double): Double { val b = y * y - 1; return x * x + 0.25 * b * b }
    override fun grad(x: Double, y: Double, out: DoubleArray) { out[0] = 2 * x; out[1] = y * (y * y - 1) }
    override val minima = listOf(doubleArrayOf(0.0, 1.0), doubleArrayOf(0.0, -1.0))
}

/**
 * A REAL model's loss: logistic regression with 2 weights (no bias) on [LogisticData]:
 * L(w) = mean ln(1 + e^(−y·w·x)) + λ/2·|w|². Convex: one valley. Minimum (Newton, float64; torch LBFGS agrees
 * to 3e-9): w* = (1.510959372304, 0.267236688266), L* = 0.355557735856. Its noise is REAL minibatch noise.
 */
object Logistic : LossSurface("logistic", -4.0, 4.0, -4.0, 4.0, -3.0, 3.0, 0.1, 0.5, 0.0) {
    override fun f(x: Double, y: Double): Double {
        var s = 0.0
        for (i in 0 until LogisticData.N) s += softplus(-LogisticData.label(i) * (x * LogisticData.x1(i) + y * LogisticData.x2(i)))
        return s / LogisticData.N + LogisticData.LAMBDA / 2 * (x * x + y * y)
    }
    override fun grad(x: Double, y: Double, out: DoubleArray) = batchGrad(x, y, null, LogisticData.N, out)

    /** Mean gradient over the first [n] entries of [idx] (null = all points) + the L2 term. */
    fun batchGrad(x: Double, y: Double, idx: IntArray?, n: Int, out: DoubleArray) {
        var gx = 0.0; var gy = 0.0
        for (k in 0 until n) {
            val i = idx?.get(k) ?: k
            val yl = LogisticData.label(i); val a = LogisticData.x1(i); val b = LogisticData.x2(i)
            val s = sigmoid(-yl * (x * a + y * b))           // d/dz softplus(−z) = −σ(−z)
            gx -= yl * a * s; gy -= yl * b * s
        }
        out[0] = gx / n + LogisticData.LAMBDA * x; out[1] = gy / n + LogisticData.LAMBDA * y
    }

    private val perm = IntArray(LogisticData.N)

    /** Batch size for a noise level: 40 (= the exact gradient) at 0 … [LogisticData.MIN_BATCH] at 1. */
    fun batchSize(noise: Double): Int =
        kotlin.math.round(LogisticData.N - (LogisticData.N - LogisticData.MIN_BATCH) * noise.coerceIn(0.0, 1.0)).toInt()

    /** REAL minibatch noise: the gradient of a random batch (drawn without replacement) of the 40 points. */
    override fun noisyGrad(x: Double, y: Double, noise: Double, rng: Rng, out: DoubleArray) {
        val n = batchSize(noise)
        if (n >= LogisticData.N) { grad(x, y, out); return }
        for (i in perm.indices) perm[i] = i
        for (k in 0 until n) { val j = k + rng.nextInt(LogisticData.N - k); val t = perm[k]; perm[k] = perm[j]; perm[j] = t }
        batchGrad(x, y, perm, n, out)
    }

    override val minima = listOf(doubleArrayOf(1.510959372304, 0.267236688266))
    override val fMin: Double get() = 0.355557735856
}
