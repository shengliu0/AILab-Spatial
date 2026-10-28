// Quality — how well a cloud of generated points reproduces the target shape (the demo's challenge metric).
//
//   precision P = fraction of generated points within EPS of the reference cloud ("lands on the shape")
//   coverage  C = fraction of the coverage set (the first nCov reference points) that have a generated point
//                 within EPS ("the whole shape is drawn", so collapsing onto one arm does not score)
//   quality   Q = min(P, C)
// EPS = 0.15 data units (every cloud has std 1; the drawn slice is ~6.4 units wide). Same definition as
// tools/train_diffusion.py; true samples score ~100 %, pure Gaussian noise ~35 % on the spiral.
// Neighbour queries use a uniform grid of cell EPS (3 x 3 cells per query), so a 300-point score costs
// ~0.1 ms — cheap enough to run live while the points move.

package com.bytedance.ailab.core.diffusion

import kotlin.math.floor

class Quality(private val ref: FloatArray, private val nCov: Int, val eps: Float = EPS) {
    data class Score(val precision: Float, val coverage: Float) { val q: Float get() = minOf(precision, coverage) }

    private val refGrid = Grid(ref, ref.size / 2)

    /** Score the first [n] points of [pts] (x, y pairs). */
    fun score(pts: FloatArray, n: Int = pts.size / 2): Score {
        if (n <= 0) return Score(0f, 0f)
        var hit = 0
        for (i in 0 until n) if (refGrid.anyWithin(pts[2 * i], pts[2 * i + 1], eps)) hit++
        val g = Grid(pts, n)
        var cov = 0
        for (j in 0 until nCov) if (g.anyWithin(ref[2 * j], ref[2 * j + 1], eps)) cov++
        return Score(hit.toFloat() / n, cov.toFloat() / nCov)
    }

    /** Points bucketed in square cells of side [eps], hashed into a fixed table (no allocation per query). */
    private inner class Grid(private val p: FloatArray, n: Int) {
        private val size = 1024
        private val head = IntArray(size) { -1 }
        private val next = IntArray(n)
        private val cx = IntArray(n); private val cy = IntArray(n)
        init {
            for (i in 0 until n) {
                val x = cell(p[2 * i]); val y = cell(p[2 * i + 1]); cx[i] = x; cy[i] = y
                val h = hash(x, y); next[i] = head[h]; head[h] = i
            }
        }
        private fun cell(v: Float) = floor(v / eps).toInt()
        private fun hash(x: Int, y: Int) = ((x * 73856093) xor (y * 19349663)) and (size - 1)
        fun anyWithin(x: Float, y: Float, r: Float): Boolean {
            val gx = cell(x); val gy = cell(y); val r2 = r * r
            for (dx in -1..1) for (dy in -1..1) {
                var i = head[hash(gx + dx, gy + dy)]
                while (i >= 0) {
                    if (cx[i] == gx + dx && cy[i] == gy + dy) {
                        val ex = p[2 * i] - x; val ey = p[2 * i + 1] - y
                        if (ex * ex + ey * ey < r2) return true
                    }
                    i = next[i]
                }
            }
            return false
        }
    }

    companion object { const val EPS = 0.15f }
}
