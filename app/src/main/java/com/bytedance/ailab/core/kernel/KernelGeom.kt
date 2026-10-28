// KernelGeom — geometry of the kernel-trick picture: the zero level set of the decision function on the floor
// (marching squares), the separating plane clipped to the lifted box, and the analytic circle that a plane cuts
// out of the paraboloid z = x² + y².

package com.bytedance.ailab.core.kernel

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.sqrt

object KernelGeom {

    /**
     * Zero contour of a function sampled on an (n+1)×(n+1) node grid over [x0,x1]×[y0,y1] (values row-major, j along y).
     * Marching squares with linear interpolation on the cell edges; saddle cells are resolved by the cell-centre
     * average. Returns segments as (xa, ya, xb, yb) quadruples.
     */
    fun contour(v: DoubleArray, n: Int, x0: Double, x1: Double, y0: Double, y1: Double, out: ArrayList<DoubleArray> = ArrayList()): ArrayList<DoubleArray> {
        out.clear()
        val hx = (x1 - x0) / n; val hy = (y1 - y0) / n
        fun at(i: Int, j: Int) = v[j * (n + 1) + i]
        for (j in 0 until n) for (i in 0 until n) {
            val a = at(i, j); val b = at(i + 1, j); val c = at(i + 1, j + 1); val d = at(i, j + 1)   // CCW from (x0,y0)
            val code = (if (a > 0) 1 else 0) or (if (b > 0) 2 else 0) or (if (c > 0) 4 else 0) or (if (d > 0) 8 else 0)
            if (code == 0 || code == 15) continue
            val xa = x0 + i * hx; val ya = y0 + j * hy
            // edge points: 0 bottom (a-b), 1 right (b-c), 2 top (d-c), 3 left (a-d)
            fun e(k: Int): DoubleArray = when (k) {
                0 -> doubleArrayOf(xa + hx * t(a, b), ya)
                1 -> doubleArrayOf(xa + hx, ya + hy * t(b, c))
                2 -> doubleArrayOf(xa + hx * t(d, c), ya + hy)
                else -> doubleArrayOf(xa, ya + hy * t(a, d))
            }
            fun seg(p: Int, q: Int) { val u = e(p); val w = e(q); out += doubleArrayOf(u[0], u[1], w[0], w[1]) }
            when (code) {
                1, 14 -> seg(3, 0)
                2, 13 -> seg(0, 1)
                3, 12 -> seg(3, 1)
                4, 11 -> seg(1, 2)
                6, 9 -> seg(0, 2)
                7, 8 -> seg(3, 2)
                5, 10 -> {
                    val centre = (a + b + c + d) / 4
                    val aPos = a > 0
                    if ((centre > 0) == aPos) { seg(3, 2); seg(0, 1) } else { seg(3, 0); seg(1, 2) }
                }
            }
        }
        return out
    }

    private fun t(p: Double, q: Double): Double { val d = p - q; return if (abs(d) < 1e-300) 0.5 else (p / d).coerceIn(0.0, 1.0) }

    /**
     * The polygon where the plane n·p + d = 0 cuts the box [lo, hi] (3-D), ordered around its centroid. Empty if the
     * plane misses the box. At most 6 vertices (a plane cuts a box in a triangle … hexagon).
     */
    fun planeBox(n: DoubleArray, d: Double, lo: DoubleArray, hi: DoubleArray): List<DoubleArray> {
        val pts = ArrayList<DoubleArray>()
        val corners = Array(8) { k -> doubleArrayOf(if (k and 1 == 0) lo[0] else hi[0], if (k and 2 == 0) lo[1] else hi[1], if (k and 4 == 0) lo[2] else hi[2]) }
        val s = DoubleArray(8) { n[0] * corners[it][0] + n[1] * corners[it][1] + n[2] * corners[it][2] + d }
        for (a in 0 until 8) for (bit in intArrayOf(1, 2, 4)) {
            val b = a or bit; if (b == a) continue
            val sa = s[a]; val sb = s[b]
            if ((sa > 0) == (sb > 0) && sa != 0.0 && sb != 0.0) continue
            if (sa == sb) continue
            val tt = sa / (sa - sb)
            if (tt < 0 || tt > 1) continue
            val p = DoubleArray(3) { corners[a][it] + (corners[b][it] - corners[a][it]) * tt }
            if (pts.none { q -> abs(q[0] - p[0]) + abs(q[1] - p[1]) + abs(q[2] - p[2]) < 1e-9 }) pts += p
        }
        if (pts.size < 3) return emptyList()
        // order by angle in a basis of the plane
        val nl = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
        val nn = doubleArrayOf(n[0] / nl, n[1] / nl, n[2] / nl)
        val ref = if (abs(nn[0]) < 0.9) doubleArrayOf(1.0, 0.0, 0.0) else doubleArrayOf(0.0, 1.0, 0.0)
        val u = cross(nn, ref).let { val l = sqrt(it[0] * it[0] + it[1] * it[1] + it[2] * it[2]); doubleArrayOf(it[0] / l, it[1] / l, it[2] / l) }
        val w = cross(nn, u)
        val c = DoubleArray(3) { k -> pts.sumOf { it[k] } / pts.size }
        return pts.sortedBy { p -> val q = DoubleArray(3) { p[it] - c[it] }; atan2(q[0] * w[0] + q[1] * w[1] + q[2] * w[2], q[0] * u[0] + q[1] * u[1] + q[2] * u[2]) }
    }

    private fun cross(a: DoubleArray, b: DoubleArray) = doubleArrayOf(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    /**
     * The plane w₁x + w₂y + w₃z + b = 0 meets the paraboloid z = x² + y² where w₃(x² + y²) + w₁x + w₂y + b = 0, i.e.
     * on the circle centred at (−w₁/2w₃, −w₂/2w₃) with r² = cx² + cy² − b/w₃ (complete the square). Its shadow on
     * the floor is the SAME circle — the decision boundary in 2-D. Returns (cx, cy, r) or null (no real circle).
     */
    fun paraboloidCut(w: DoubleArray, b: Double): DoubleArray? {
        if (abs(w[2]) < 1e-12) return null
        val cx = -w[0] / (2 * w[2]); val cy = -w[1] / (2 * w[2])
        val r2 = cx * cx + cy * cy - b / w[2]
        return if (r2 <= 0) null else doubleArrayOf(cx, cy, sqrt(r2))
    }
}
