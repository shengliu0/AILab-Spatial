// Svm — a soft-margin support-vector machine solved EXACTLY (to a KKT tolerance) by SMO.
//
// Dual problem (Boser, Guyon & Vapnik 1992; Cortes & Vapnik 1995 for the soft margin C):
//     min_α  ½ αᵀQα − Σα     s.t.  Σ yᵢαᵢ = 0,  0 ≤ αᵢ ≤ C,     Q_ij = yᵢ yⱼ K(xᵢ, xⱼ)
// decision f(x) = Σ αᵢ yᵢ K(xᵢ, x) + b.  Points with αᵢ > 0 are the SUPPORT VECTORS — the only ones that matter.
//
// Solver: sequential minimal optimisation (Platt 1998) with LIBSVM's second-order working-set selection
// (Fan, Chen & Lin, JMLR 6 (2005) 1889–1918, "WSS 2") and its bias rule (b = −ρ, ρ = mean yᵢ∇ᵢ over free αᵢ).
// The update and clipping code follows LIBSVM's Solver::Solve line by line. It stops when the maximal KKT
// violation m(α) − M(α) < eps, so the result is a certified ε-optimum, deterministic (no randomness, fixed
// tie-breaking by index). n ≤ ~60 here, so the full kernel matrix is precomputed.
//
// Pure Kotlin, no Android.

package com.bytedance.ailab.core.kernel

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** A trained dual solution. [alpha] per training point, f(x) = Σ αᵢyᵢK(xᵢ,x) + [b]. */
class SvmSolution(val alpha: DoubleArray, val b: Double, val C: Double, val iterations: Int, val gap: Double) {
    /** Indices with αᵢ > tiny: the support vectors. */
    fun supportVectors(rel: Double = 1e-7): IntArray {
        val amax = alpha.maxOrNull() ?: 0.0
        val t = rel * max(1.0, amax)
        return alpha.indices.filter { alpha[it] > t }.toIntArray()
    }
    /** Support vectors at the upper bound αᵢ = C: inside the margin or misclassified. */
    fun boundSupportVectors(rel: Double = 1e-7): IntArray = alpha.indices.filter { alpha[it] >= C * (1 - rel) }.toIntArray()
}

object Svm {
    private const val TAU = 1e-12

    /**
     * Train on a precomputed kernel matrix [K] (n×n, symmetric) with labels [y] (±1).
     * [eps] = stopping tolerance on the maximal KKT violation (LIBSVM's default is 1e-3; we go far tighter).
     */
    fun train(K: Array<DoubleArray>, y: IntArray, C: Double, eps: Double = 1e-10, maxIter: Int = 200_000): SvmSolution {
        val n = y.size
        require(K.size == n && n >= 2) { "need ≥ 2 points" }
        require(y.any { it > 0 } && y.any { it < 0 }) { "need both classes" }
        val a = DoubleArray(n)
        val G = DoubleArray(n) { -1.0 }                 // ∇ = Qα − e, α = 0
        val QD = DoubleArray(n) { K[it][it] }
        fun q(i: Int, j: Int) = y[i] * y[j] * K[i][j]
        fun isUp(t: Int) = (y[t] > 0 && a[t] < C) || (y[t] < 0 && a[t] > 0)
        fun isLow(t: Int) = (y[t] > 0 && a[t] > 0) || (y[t] < 0 && a[t] < C)
        var iter = 0
        var gap = Double.POSITIVE_INFINITY
        while (iter < maxIter) {
            // ---- working set: i = maximal violator in I_up, j by second-order gain in I_low
            var gmax = Double.NEGATIVE_INFINITY; var i = -1
            for (t in 0 until n) if (isUp(t)) { val v = -y[t] * G[t]; if (v > gmax) { gmax = v; i = t } }
            var gmin = Double.POSITIVE_INFINITY; var j = -1; var objMin = Double.POSITIVE_INFINITY
            for (t in 0 until n) if (isLow(t)) {
                val v = -y[t] * G[t]
                if (v < gmin) gmin = v
                if (i >= 0) {
                    val bb = gmax - v
                    if (bb > 0) {
                        var aa = K[i][i] + K[t][t] - 2 * K[i][t]
                        if (aa <= 0) aa = TAU
                        val o = -(bb * bb) / aa
                        if (o < objMin) { objMin = o; j = t }
                    }
                }
            }
            gap = gmax - gmin
            if (i < 0 || j < 0 || gap < eps) break
            iter++
            // ---- two-variable update (LIBSVM Solver::Solve), C_i = C_j = C
            val oi = a[i]; val oj = a[j]
            if (y[i] != y[j]) {
                var quad = QD[i] + QD[j] + 2 * q(i, j); if (quad <= 0) quad = TAU
                val delta = (-G[i] - G[j]) / quad
                val diff = a[i] - a[j]
                a[i] += delta; a[j] += delta
                if (diff > 0) { if (a[j] < 0) { a[j] = 0.0; a[i] = diff } } else { if (a[i] < 0) { a[i] = 0.0; a[j] = -diff } }
                if (diff > 0) { if (a[i] > C) { a[i] = C; a[j] = C - diff } } else { if (a[j] > C) { a[j] = C; a[i] = C + diff } }
            } else {
                var quad = QD[i] + QD[j] - 2 * q(i, j); if (quad <= 0) quad = TAU
                val delta = (G[i] - G[j]) / quad
                val sum = a[i] + a[j]
                a[i] -= delta; a[j] += delta
                if (sum > C) { if (a[i] > C) { a[i] = C; a[j] = sum - C } } else { if (a[j] < 0) { a[j] = 0.0; a[i] = sum } }
                if (sum > C) { if (a[j] > C) { a[j] = C; a[i] = sum - C } } else { if (a[i] < 0) { a[i] = 0.0; a[j] = sum } }
            }
            val di = a[i] - oi; val dj = a[j] - oj
            for (k in 0 until n) G[k] += q(k, i) * di + q(k, j) * dj
        }
        // ---- bias: ρ = mean of yᵢ∇ᵢ over free αᵢ (else the midpoint of the feasible interval); b = −ρ
        var ub = Double.POSITIVE_INFINITY; var lb = Double.NEGATIVE_INFINITY; var nFree = 0; var sFree = 0.0
        for (t in 0 until n) {
            val yG = y[t] * G[t]
            when {
                a[t] >= C -> if (y[t] < 0) ub = min(ub, yG) else lb = max(lb, yG)
                a[t] <= 0 -> if (y[t] > 0) ub = min(ub, yG) else lb = max(lb, yG)
                else -> { nFree++; sFree += yG }
            }
        }
        val rho = if (nFree > 0) sFree / nFree else (ub + lb) / 2
        return SvmSolution(a, -rho, C, iter, gap)
    }

    // ---------------------------------------------------------------- kernels
    fun linearGram(X: Array<DoubleArray>): Array<DoubleArray> = Array(X.size) { i -> DoubleArray(X.size) { j -> dot(X[i], X[j]) } }
    fun rbfGram(X: Array<DoubleArray>, gamma: Double): Array<DoubleArray> = Array(X.size) { i -> DoubleArray(X.size) { j -> rbf(X[i], X[j], gamma) } }
    fun rbf(a: DoubleArray, b: DoubleArray, gamma: Double): Double { var s = 0.0; for (k in a.indices) { val d = a[k] - b[k]; s += d * d }; return exp(-gamma * s) }
    fun dot(a: DoubleArray, b: DoubleArray): Double { var s = 0.0; for (k in a.indices) s += a[k] * b[k]; return s }
}

/** A linear SVM in feature space: f(x) = w·x + b. */
class LinearSvm(val X: Array<DoubleArray>, val y: IntArray, val sol: SvmSolution) {
    val dim = X[0].size
    /** w = Σ αᵢ yᵢ xᵢ (the primal normal, unique even when α is not). */
    val w = DoubleArray(dim).also { w -> for (i in X.indices) { val c = sol.alpha[i] * y[i]; if (c != 0.0) for (k in 0 until dim) w[k] += c * X[i][k] } }
    val b get() = sol.b
    val wNorm get() = sqrt(Svm.dot(w, w))
    /** Margin width 2/|w| (distance between the planes w·x + b = ±1); ∞ when w = 0. */
    val margin get() = if (wNorm < 1e-12) Double.POSITIVE_INFINITY else 2.0 / wNorm
    fun f(x: DoubleArray) = Svm.dot(w, x) + b
    fun accuracy(): Double = X.indices.count { (if (f(X[it]) >= 0) 1 else -1) == y[it] }.toDouble() / X.size

    /**
     * KKT residual, scaled: max over i of the violation of
     *   αᵢ = 0 ⇒ yᵢf ≥ 1,   0 < αᵢ < C ⇒ yᵢf = 1,   αᵢ = C ⇒ yᵢf ≤ 1,   plus |Σαᵢyᵢ| and the bounds.
     */
    fun kktViolation(boundTol: Double = 1e-9): Double {
        var v = abs(sol.alpha.indices.sumOf { sol.alpha[it] * y[it] })
        val C = sol.C
        for (i in X.indices) {
            val a = sol.alpha[i]; val m = y[i] * f(X[i])
            v = max(v, max(-a, a - C))
            v = when {
                a <= boundTol * C -> max(v, 1 - m)
                a >= C * (1 - boundTol) -> max(v, m - 1)
                else -> max(v, abs(m - 1))
            }
        }
        return v
    }

    companion object {
        fun train(X: Array<DoubleArray>, y: IntArray, C: Double, eps: Double = 1e-10): LinearSvm =
            LinearSvm(X, y, Svm.train(Svm.linearGram(X), y, C, eps))
    }
}
