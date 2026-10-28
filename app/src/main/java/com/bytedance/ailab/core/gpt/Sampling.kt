// Sampling — from next-token scores (logits z) to a choice: temperature softmax, entropy, top-k, sampling.
//
//   p_i(T) = exp(z_i / T) / sum_j exp(z_j / T)        (the Boltzmann / Gibbs distribution at temperature T)
// T -> 0 puts all mass on argmax z (greedy decoding); T = 1 is the distribution the model was trained to
// predict; T -> infinity tends to uniform. Its entropy H(T) = -sum p ln p is non-decreasing in T:
// dH/dT = Var_p(z) / T^3 >= 0 (the textbook heat-capacity identity C = Var(E)/T^2 >= 0 with E = -z), so
// "hotter" always means "flatter".

package com.bytedance.ailab.core.gpt

import kotlin.math.exp
import kotlin.math.ln

object Sampling {
    /** Temperatures at or below this are treated as greedy (exactly one-hot on the argmax). */
    const val T_GREEDY = 1e-6

    fun softmax(z: DoubleArray, t: Double, out: DoubleArray = DoubleArray(z.size)): DoubleArray {
        val am = argmax(z)
        if (t <= T_GREEDY) { out.fill(0.0); out[am] = 1.0; return out }
        val mx = z[am]
        var s = 0.0
        for (i in z.indices) { out[i] = exp((z[i] - mx) / t); s += out[i] }
        for (i in z.indices) out[i] /= s
        return out
    }

    /** Shannon entropy in nats (0 ln 0 = 0). */
    fun entropy(p: DoubleArray): Double { var h = 0.0; for (x in p) if (x > 0.0) h -= x * ln(x); return h }

    fun argmax(z: DoubleArray): Int { var b = 0; for (i in 1 until z.size) if (z[i] > z[b]) b = i; return b }

    /** Indices of the [k] largest entries, largest first (ties: lower index first). */
    fun topK(p: DoubleArray, k: Int): IntArray {
        val idx = p.indices.sortedWith(compareByDescending<Int> { p[it] }.thenBy { it })
        return idx.take(k).toIntArray()
    }

    /** Inverse-CDF sample with a uniform [u] in [0, 1). Deterministic given u (tests and the demo seed it). */
    fun sample(p: DoubleArray, u: Double): Int {
        var c = 0.0
        for (i in p.indices) { c += p[i]; if (u < c) return i }
        // rounding: fall back to the last index with mass
        for (i in p.indices.reversed()) if (p[i] > 0.0) return i
        return 0
    }
}
