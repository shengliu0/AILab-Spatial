// TopK — which connections to DRAW. A dense layer 784 -> 64 has 50 176 weights; drawing them all is a grey fog
// (and ~100 k vertices). What explains one particular answer are the few largest CONTRIBUTIONS w[j,i] * a[i]:
// the terms that actually moved neuron j's sum for THIS input. We draw only the k largest |w a| per layer pair
// ("threads"), and, for an output class, the chain of largest POSITIVE contributions back to the pixels
// (the path that supports the answer).

package com.bytedance.ailab.core.mlp

import kotlin.math.abs

/** One drawable thread: contribution [value] = w[dst, src] * a[src] from neuron src (layer L) to dst (layer L+1). */
data class Edge(val src: Int, val dst: Int, val value: Float)

object TopK {
    /**
     * The [k] contributions w[j * nIn + i] * a[i] with the largest magnitude, over destinations [dsts] (null = all)
     * and all sources. [sign] > 0 keeps only positive contributions, < 0 only negative, 0 both. Zero
     * contributions are never returned. Result sorted by |value| descending; ties keep the lower (src, dst).
     * Cost: O(nnz(a) x |dsts| x log k) — inactive (zero) sources are skipped, which is most pixels.
     */
    fun strongest(w: FloatArray, a: FloatArray, nOut: Int, nIn: Int, k: Int, dsts: IntArray? = null, sign: Int = 0): List<Edge> {
        if (k <= 0) return emptyList()
        val hv = FloatArray(k); val hs = IntArray(k); val hd = IntArray(k); val ho = IntArray(k)
        var n = 0
        var order = 0
        // min-heap on (|v| ascending, order descending): the root is the weakest kept entry
        fun less(x: Int, y: Int) = hv[x] < hv[y] || (hv[x] == hv[y] && ho[x] > ho[y])
        fun swap(x: Int, y: Int) {
            val v = hv[x]; hv[x] = hv[y]; hv[y] = v; val s = hs[x]; hs[x] = hs[y]; hs[y] = s
            val d = hd[x]; hd[x] = hd[y]; hd[y] = d; val o = ho[x]; ho[x] = ho[y]; ho[y] = o
        }
        fun down(i0: Int) {
            var i = i0
            while (true) {
                val l = 2 * i + 1; val r = l + 1; var m = i
                if (l < n && less(l, m)) m = l
                if (r < n && less(r, m)) m = r
                if (m == i) return
                swap(i, m); i = m
            }
        }
        fun up(i0: Int) { var i = i0; while (i > 0) { val p = (i - 1) / 2; if (less(i, p)) { swap(i, p); i = p } else return } }
        val rows = dsts ?: IntArray(nOut) { it }
        for (i in 0 until nIn) {
            val ai = a[i]
            if (ai == 0f) continue
            for (j in rows) {
                val v = w[j * nIn + i] * ai
                if (v == 0f || (sign > 0 && v < 0f) || (sign < 0 && v > 0f)) continue
                val m = abs(v)
                order++
                if (n < k) { hv[n] = m; hs[n] = i; hd[n] = j; ho[n] = order; n++; up(n - 1) }
                else if (m > hv[0]) { hv[0] = m; hs[0] = i; hd[0] = j; ho[0] = order; down(0) }
            }
        }
        val idx = (0 until n).sortedWith(compareBy<Int>({ -hv[it] }, { ho[it] }))
        return idx.map { Edge(hs[it], hd[it], w[hd[it] * nIn + hs[it]] * a[hs[it]]) }
    }

    /**
     * The chain of strongest SUPPORT for output class [cls]: the [k2] hidden-2 neurons with the largest positive
     * w3 * h2 into cls; for each of those, its [k1] largest positive w2 * h1; for each hidden-1 neuron reached,
     * its [k0] largest positive w1 * x. Edges per stage, strongest first.
     */
    class Path(val cls: Int, val toOut: List<Edge>, val toH2: List<Edge>, val toH1: List<Edge>)

    fun supportPath(net: Mlp, act: Mlp.Act, cls: Int, k2: Int = 4, k1: Int = 3, k0: Int = 3): Path {
        val toOut = strongest(net.w3, act.h2, net.nOut, net.nH2, k2, intArrayOf(cls), sign = 1)
        val toH2 = toOut.flatMap { e -> strongest(net.w2, act.h1, net.nH2, net.nH1, k1, intArrayOf(e.src), sign = 1) }
        val h1s = toH2.map { it.src }.distinct()
        val toH1 = h1s.flatMap { j -> strongest(net.w1, act.x, net.nH1, net.nIn, k0, intArrayOf(j), sign = 1) }
        return Path(cls, toOut, toH2, toH1)
    }
}
