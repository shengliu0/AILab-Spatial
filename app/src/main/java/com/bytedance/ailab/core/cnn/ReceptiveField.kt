// ReceptiveField — which input pixels can influence one cell of a deeper layer.
//
// For a chain of layers with kernel k_l and stride s_l (no padding), the receptive-field size r and the
// "jump" j (input pixels between neighbouring cells) follow the standard recurrence
//     r_l = r_{l-1} + (k_l − 1) · j_{l-1},     j_l = j_{l-1} · s_l,     r_0 = j_0 = 1
// (Araujo, Norris & Sim, "Computing Receptive Fields of Convolutional Neural Networks", Distill 2019).
// Here: conv 3 (r 3, j 1) → pool 2 (r 4, j 2) → conv 3 (r 8, j 2) → pool 2 (r 10, j 4).
// Cell i of layer l starts at input row i · j_l (valid convolutions, no padding).

package com.bytedance.ailab.core.cnn

object ReceptiveField {
    /** (kernel, stride) of conv1, pool1, conv2, pool2. Layer index 0 = the input, 1..4 = these. */
    val LAYERS = listOf(3 to 1, 2 to 2, 3 to 1, 2 to 2)

    /** Receptive-field size after each layer, index 0 = input (1). */
    fun sizes(): IntArray {
        val r = IntArray(LAYERS.size + 1); var j = 1; r[0] = 1
        for ((l, ks) in LAYERS.withIndex()) { r[l + 1] = r[l] + (ks.first - 1) * j; j *= ks.second }
        return r
    }

    /** Jump (input pixels per cell step) after each layer, index 0 = input. */
    fun jumps(): IntArray {
        val out = IntArray(LAYERS.size + 1); out[0] = 1
        for ((l, ks) in LAYERS.withIndex()) out[l + 1] = out[l] * ks.second
        return out
    }

    /** Spatial side of every layer for a [n]×[n] input: 28 → 26 → 13 → 11 → 5. */
    fun sides(n: Int = 28): IntArray {
        val out = IntArray(LAYERS.size + 1); out[0] = n
        for ((l, ks) in LAYERS.withIndex()) out[l + 1] = CnnOps.outSize(out[l], ks.first, ks.second)
        return out
    }

    /**
     * The window [first, last] (inclusive, same for rows and columns) in layer [target] that cell index [i] of
     * layer [from] sees, walking back one layer at a time: cell i of layer l covers [i·s_l, i·s_l + k_l − 1].
     */
    fun window(from: Int, i: Int, target: Int = 0): IntArray {
        require(target in 0..from && from <= LAYERS.size)
        var lo = i; var hi = i
        for (l in from downTo target + 1) {
            val (k, s) = LAYERS[l - 1]
            lo *= s; hi = hi * s + k - 1
        }
        return intArrayOf(lo, hi)
    }
}
