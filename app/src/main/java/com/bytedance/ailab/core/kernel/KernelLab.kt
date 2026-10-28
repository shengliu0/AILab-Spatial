// KernelLab — the whole experiment as plain data: a point set, a lift, the soft-margin constant C, and the two
// linear SVMs trained on it — one on the raw 2-D points (the best straight line) and one on the lifted 3-D points
// (the best flat plane). Everything the demo draws or prints is read from here; JVM tests check it.

package com.bytedance.ailab.core.kernel

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.sqrt

class KernelLab {
    var set = DataSet.RINGS; private set
    var noise = 0; private set
    var lift = Lift.RADIAL; private set
    var C = 1000.0; private set

    /** The generated points (reset target) and the current, possibly user-moved ones. */
    var base: Points = KernelData.make(set, noise); private set
    var pts: Points = base.copy(); private set
    val n get() = pts.n

    var rbf: RbfFeature? = null; private set
    lateinit var svm2: LinearSvm; private set
    lateinit var svm3: LinearSvm; private set
    /** Bumped on every retrain (the demo redraws when it changes). */
    var version = 0; private set

    init { retrain() }

    fun configure(set: DataSet = this.set, noise: Int = this.noise, lift: Lift = this.lift, C: Double = this.C) {
        val newData = set != this.set || noise != this.noise
        this.set = set; this.noise = noise.coerceIn(0, 2); this.lift = lift; this.C = C
        if (newData) { base = KernelData.make(set, this.noise); pts = base.copy() }
        retrain()
    }

    fun resetPoints() { pts = base.copy(); retrain() }

    /** Move point [i] (clamped into the disc). [refitFeature] = false keeps a learned RBF coordinate frozen (live drag). */
    fun move(i: Int, x: Double, y: Double, refitFeature: Boolean = true) {
        var px = x; var py = y
        val r = hypot(px, py); if (r > KernelData.R_MAX) { px *= KernelData.R_MAX / r; py *= KernelData.R_MAX / r }
        pts.x[i] = px; pts.y[i] = py
        retrain(refitFeature)
    }

    fun retrain(refitFeature: Boolean = true) {
        if (lift == Lift.RBF) { if (refitFeature || rbf == null) rbf = RbfFeature(pts) } else rbf = null
        val y = pts.label
        svm2 = LinearSvm.train(Array(n) { doubleArrayOf(pts.x[it], pts.y[it]) }, y, C)
        svm3 = LinearSvm.train(Array(n) { lifted(pts.x[it], pts.y[it]) }, y, C)
        version++
    }

    // ---------------------------------------------------------------- the lift
    fun phi(x: Double, y: Double): Double = when (lift) {
        Lift.RADIAL -> x * x + y * y
        Lift.PRODUCT -> x * y
        Lift.RBF -> rbf!!.phi(x, y)
    }
    fun lifted(x: Double, y: Double) = doubleArrayOf(x, y, phi(x, y))

    /** Decision function on the floor: 2-D line, or the lifted plane pulled back through the lift. */
    fun g(x: Double, y: Double, liftedModel: Boolean): Double =
        if (liftedModel) svm3.w[0] * x + svm3.w[1] * y + svm3.w[2] * phi(x, y) + svm3.b
        else svm2.w[0] * x + svm2.w[1] * y + svm2.b

    fun model(liftedModel: Boolean) = if (liftedModel) svm3 else svm2
    fun accuracy(liftedModel: Boolean) = model(liftedModel).accuracy()

    /** yᵢ·f(xᵢ) of point i (≥ 1: outside the margin, on the right side). */
    fun marginOf(i: Int, liftedModel: Boolean): Double {
        val m = model(liftedModel)
        val x = if (liftedModel) lifted(pts.x[i], pts.y[i]) else doubleArrayOf(pts.x[i], pts.y[i])
        return pts.label[i] * m.f(x)
    }

    fun isSupport(i: Int, liftedModel: Boolean): Boolean { val s = model(liftedModel).sol; val amax = s.alpha.maxOrNull() ?: 0.0; return s.alpha[i] > 1e-7 * maxOf(1.0, amax) }

    /** Unit-normalised plane (|w| = 1): (n₁, n₂, n₃, d) for the lifted model, (n₁, n₂, 0, d) for the 2-D line. */
    fun unitPlane(liftedModel: Boolean): DoubleArray {
        val m = model(liftedModel); val l = m.wNorm.coerceAtLeast(1e-300)
        return if (liftedModel) doubleArrayOf(m.w[0] / l, m.w[1] / l, m.w[2] / l, m.b / l) else doubleArrayOf(m.w[0] / l, m.w[1] / l, 0.0, m.b / l)
    }

    /** The decision circle (cx, cy, r) for the paraboloid lift, null otherwise / if the plane misses it. */
    fun circle(): DoubleArray? = if (lift == Lift.RADIAL) KernelGeom.paraboloidCut(svm3.w, svm3.b) else null

    // ---------------------------------------------------------------- the floor contour
    /** Half-size of the floor square (units). */
    val L = 1.05

    /** Zero contour of g on an n×n cell grid over the floor square. */
    fun contour(liftedModel: Boolean, cells: Int, out: ArrayList<DoubleArray> = ArrayList()): ArrayList<DoubleArray> {
        val v = DoubleArray((cells + 1) * (cells + 1))
        for (j in 0..cells) for (i in 0..cells) {
            val x = -L + 2 * L * i / cells; val y = -L + 2 * L * j / cells
            v[j * (cells + 1) + i] = g(x, y, liftedModel)
        }
        return KernelGeom.contour(v, cells, -L, L, -L, L, out)
    }

    // ---------------------------------------------------------------- lesson helpers
    /** Direction (unit, floor) from point i toward the decision boundary (−y·∇g/|∇g|). */
    fun towardBoundary(i: Int, liftedModel: Boolean): DoubleArray {
        val x = pts.x[i]; val y = pts.y[i]; val h = 1e-5
        val gx = (g(x + h, y, liftedModel) - g(x - h, y, liftedModel)) / (2 * h)
        val gy = (g(x, y + h, liftedModel) - g(x, y - h, liftedModel)) / (2 * h)
        val l = hypot(gx, gy)
        if (l < 1e-12) { val r = hypot(x, y).coerceAtLeast(1e-9); return doubleArrayOf(-x / r, -y / r) }
        val s = -pts.label[i] / l
        return doubleArrayOf(gx * s, gy * s)
    }

    /** The point farthest outside the margin (largest yᵢfᵢ) that is not a support vector, or −1. */
    fun farPoint(liftedModel: Boolean): Int = (0 until n).filter { !isSupport(it, liftedModel) }.maxByOrNull { marginOf(it, liftedModel) } ?: -1

    /** A support vector ON the margin (0 < α < C) — moving it must move the plane; the one nearest the viewer's side. */
    fun freeSupport(liftedModel: Boolean): Int {
        val s = model(liftedModel).sol
        val free = (0 until n).filter { isSupport(it, liftedModel) && s.alpha[it] < C * (1 - 1e-7) }
        val pool = free.ifEmpty { (0 until n).filter { isSupport(it, liftedModel) } }
        return pool.minByOrNull { pts.y[it] } ?: -1
    }

    /**
     * How far point i can move toward the boundary while staying outside the margin under the CURRENT model
     * (yf ≥ 1 + slack): then the current solution still satisfies every KKT condition, so it is still THE optimum.
     */
    fun safeStepTowardBoundary(i: Int, liftedModel: Boolean, want: Double, slack: Double = 0.05): Double {
        val d = towardBoundary(i, liftedModel)
        val m = model(liftedModel)
        var s = want
        repeat(12) {
            val x = pts.x[i] + d[0] * s; val y = pts.y[i] + d[1] * s
            val inside = hypot(x, y) <= KernelData.R_MAX
            val f = if (liftedModel) m.f(lifted(x, y)) else m.f(doubleArrayOf(x, y))
            if (inside && pts.label[i] * f >= 1 + slack) return s
            s /= 2
        }
        return 0.0
    }

    /** Relative change between two planes (w, b): 0 = identical. */
    companion object {
        fun planeDelta(w0: DoubleArray, b0: Double, w1: DoubleArray, b1: Double): Double {
            var s = (b1 - b0) * (b1 - b0); var n0 = b0 * b0
            for (k in w0.indices) { s += (w1[k] - w0[k]) * (w1[k] - w0[k]); n0 += w0[k] * w0[k] }
            return sqrt(s) / sqrt(n0).coerceAtLeast(1e-12)
        }
        fun near(a: Double, b: Double, tol: Double) = abs(a - b) <= tol
    }
}
