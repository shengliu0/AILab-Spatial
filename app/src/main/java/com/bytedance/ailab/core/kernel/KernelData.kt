// KernelData — the toy data sets and the "lifts" (feature maps) of the kernel-trick demo.
//
// Every data set lives inside the unit disc (|p| ≤ 0.97), 40 points, 20 per class (+1 = "A", −1 = "B"), built
// from a fixed-seed generator (deterministic, no java.util.Random so the numbers never change between JDKs).
//
// A lift maps a 2-D point to a 3-D one, (x, y) → (x, y, φ(x, y)):
//   RADIAL   φ = x² + y²                     (the paraboloid: rings become two layers — a FLAT plane cuts a circle)
//   PRODUCT  φ = x·y                         (the saddle: XOR quadrants go up/down alternately)
//   RBF      φ = ½·tanh f_RBF(x, y)          (a LEARNED third coordinate: the score of a Gaussian-kernel SVM
//                                            (γ = 8, C = 10) trained on the same points — what the kernel trick
//                                            computes implicitly, squashed to a drawable height)
// Each lift has a display floor zf and top zt so every surface is 1 unit tall over the unit disc.

package com.bytedance.ailab.core.kernel

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tanh

/** Small deterministic generator: 64-bit LCG (Knuth MMIX constants) + Box–Muller. */
class KRng(seed: Long) {
    private var s = seed * 6364136223846793005L + 1442695040888963407L
    fun next(): Double { s = s * 6364136223846793005L + 1442695040888963407L; return ((s ushr 11).toDouble()) / (1L shl 53).toDouble() }
    fun gauss(): Double { val u = next().coerceAtLeast(1e-12); val v = next(); return sqrt(-2 * ln(u)) * cos(2 * PI * v) }
}

enum class DataSet { RINGS, XOR, MOONS, LINEAR }

enum class Lift(val zf: Double, val zt: Double) {
    RADIAL(0.0, 1.0), PRODUCT(-0.5, 0.5), RBF(-0.5, 0.5);
    /** Display height (units above the floor) of a feature value. */
    fun h(phi: Double) = phi - zf
}

/** Labelled 2-D points. */
class Points(val x: DoubleArray, val y: DoubleArray, val label: IntArray) {
    val n get() = x.size
    fun copy() = Points(x.copyOf(), y.copyOf(), label.copyOf())
}

object KernelData {
    const val N = 40
    const val R_MAX = 0.97
    /** Gaussian jitter σ for the three noise levels. */
    val NOISE = doubleArrayOf(0.0, 0.06, 0.13)

    fun make(set: DataSet, noise: Int, seed: Long = 1992L): Points {
        val r = KRng(seed + set.ordinal * 7919L)
        val nr = KRng(seed * 31 + 17)                         // the noise stream is separate: noise 0 vs 1 = same base points
        val x = DoubleArray(N); val y = DoubleArray(N); val l = IntArray(N)
        for (i in 0 until N) {
            val c = if (i % 2 == 0) 1 else -1
            l[i] = c
            var px: Double; var py: Double
            when (set) {
                DataSet.RINGS -> {
                    val th = 2 * PI * r.next()
                    val rad = if (c > 0) 0.42 * sqrt(r.next()) else 0.70 + 0.25 * r.next()
                    px = rad * cos(th); py = rad * sin(th)
                }
                DataSet.XOR -> {
                    // quadrant k (i/2 mod 4 spreads them evenly): class +1 in quadrants I and III (x·y > 0)
                    val sx = if ((i / 2) % 2 == 0) 1.0 else -1.0
                    val sy = sx * c
                    px = sx * (0.16 + 0.48 * r.next()); py = sy * (0.16 + 0.48 * r.next())
                }
                DataSet.MOONS -> {
                    // the classic two moons (upper arc of A around (0,0), lower arc of B around (1, ½)), scaled ×0.55, centred
                    val t = PI * r.next()
                    val mx = if (c > 0) cos(t) else 1 - cos(t)
                    val my = if (c > 0) sin(t) else 0.5 - sin(t)
                    px = (mx - 0.5) * 0.55 + 0.035 * r.gauss(); py = (my - 0.25) * 0.55 + 0.035 * r.gauss()
                }
                DataSet.LINEAR -> {
                    val th = 2 * PI * r.next(); val rad = 0.36 * sqrt(r.next())
                    val cx = if (c > 0) -0.40 else 0.40; val cy = if (c > 0) 0.32 else -0.32
                    px = cx + rad * cos(th); py = cy + rad * sin(th)
                }
            }
            val s = NOISE[noise.coerceIn(0, 2)]
            val gx = nr.gauss(); val gy = nr.gauss()
            px += s * gx; py += s * gy
            val rr = sqrt(px * px + py * py)
            if (rr > R_MAX) { px *= R_MAX / rr; py *= R_MAX / rr }
            x[i] = px; y[i] = py
        }
        return Points(x, y, l)
    }
}

/** The learned RBF coordinate: f(p) = Σ αᵢyᵢ exp(−γ|p − pᵢ|²) + b, frozen copy of the training points. */
class RbfFeature(p: Points, val gamma: Double = 8.0, val C: Double = 10.0) {
    private val px = p.x.copyOf(); private val py = p.y.copyOf()
    private val coef: DoubleArray
    val b: Double
    val sol: SvmSolution
    init {
        val X = Array(p.n) { doubleArrayOf(p.x[it], p.y[it]) }
        sol = Svm.train(Svm.rbfGram(X, gamma), p.label, C)
        coef = DoubleArray(p.n) { sol.alpha[it] * p.label[it] }
        b = sol.b
    }
    fun f(x: Double, y: Double): Double {
        var s = b
        for (i in px.indices) if (coef[i] != 0.0) { val dx = x - px[i]; val dy = y - py[i]; s += coef[i] * kotlin.math.exp(-gamma * (dx * dx + dy * dy)) }
        return s
    }
    fun phi(x: Double, y: Double) = 0.5 * tanh(f(x, y))
}
