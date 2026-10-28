// WordSpace — the mini word-embedding model of the "embedding" demo: vectors, cosine, nearest neighbours,
// analogies (3CosAdd) and the 3-D PCA views. Pure Kotlin, no Android.
//
// The vectors are skip-gram-with-negative-sampling input vectors (Mikolov et al. 2013, "Distributed
// Representations of Words and Phrases and their Compositionality") trained by tools/train_embedding.py on a
// corpus that script generates from templates. File layout (little-endian), see the script's docstring:
//   "EMB1" | int32 V, D, nRel, nView
//   | V x { int32 nz, utf8 zh, int32 ne, utf8 en, int32 category, int32 shown }
//   | float32 vec[V][D]
//   | nRel x { int32 nPairs, nPairs x int32 (a, b) }       0 gender (male, female), 1 (country, capital), 2 (base, comparative)
//   | nView x { int32 n, ascii id, float32 mean[D], float32 W[3][D] }    p = (v/|v| - mean) . W_k

package com.bytedance.ailab.core.embedding

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

class WordSpace(bytes: ByteArray) {
    val v: Int
    val d: Int
    val zh: Array<String>
    val en: Array<String>
    val category: IntArray
    val shown: BooleanArray
    /** Raw vectors [V][D]. */
    val vec: Array<FloatArray>
    /** Unit vectors in double precision (all maths below uses them). */
    val unit: Array<DoubleArray>
    /** Relation pairs: rel -> list of (a, b). */
    val relations: List<List<Pair<Int, Int>>>
    val viewIds: List<String>
    private val viewMean: List<DoubleArray>
    private val viewW: List<Array<DoubleArray>>

    init {
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val magic = ByteArray(4).also { b.get(it) }
        require(String(magic, Charsets.US_ASCII) == "EMB1") { "not an EMB1 file" }
        v = b.int; d = b.int; val nRel = b.int; val nView = b.int
        fun str(): String { val n = b.int; val a = ByteArray(n); b.get(a); return String(a, Charsets.UTF_8) }
        zh = Array(v) { "" }; en = Array(v) { "" }; category = IntArray(v); shown = BooleanArray(v)
        for (i in 0 until v) { zh[i] = str(); en[i] = str(); category[i] = b.int; shown[i] = b.int != 0 }
        vec = Array(v) { FloatArray(d) { b.float } }
        unit = Array(v) { i ->
            var s = 0.0; for (k in 0 until d) s += vec[i][k].toDouble() * vec[i][k]
            val n = sqrt(s); DoubleArray(d) { vec[i][it] / n }
        }
        relations = List(nRel) { val n = b.int; List(n) { Pair(b.int, b.int) } }
        val ids = ArrayList<String>(); val means = ArrayList<DoubleArray>(); val ws = ArrayList<Array<DoubleArray>>()
        repeat(nView) {
            val n = b.int; val a = ByteArray(n); b.get(a); ids += String(a, Charsets.US_ASCII)
            means += DoubleArray(d) { b.float.toDouble() }
            ws += Array(3) { DoubleArray(d) { b.float.toDouble() } }
        }
        viewIds = ids; viewMean = means; viewW = ws
    }

    /** Content words the demo shows and searches (hidden context words excluded). */
    val shownIds: IntArray by lazy { (0 until v).filter { shown[it] }.toIntArray() }
    fun indexOf(word: String): Int = zh.indexOf(word)

    fun cos(a: Int, b: Int): Double = dot(unit[a], unit[b])
    fun cos(t: DoubleArray, b: Int): Double = dot(t, unit[b]) / sqrt(dot(t, t))

    /** Top-[k] shown words by cosine to word [w] (itself excluded); ties by index. */
    fun nearest(w: Int, k: Int = 5): List<Pair<Int, Double>> = rank(unit[w], setOf(w), k)

    /** 3CosAdd: t = â − b̂ + ĉ (unit vectors), rank the shown words by cos(t, x), excluding a, b, c. */
    fun analogy(a: Int, b: Int, c: Int, k: Int = 3): List<Pair<Int, Double>> = rank(target(a, b, c), setOf(a, b, c), k)

    fun target(a: Int, b: Int, c: Int): DoubleArray = DoubleArray(d) { unit[a][it] - unit[b][it] + unit[c][it] }

    /** Rank of word [x] (1 = first) among the analogy candidates. */
    fun rankOf(a: Int, b: Int, c: Int, x: Int): Int {
        val t = target(a, b, c); val s = cos(t, x)
        return 1 + shownIds.count { it != a && it != b && it != c && it != x && (cos(t, it) > s || (cos(t, it) == s && it < x)) }
    }

    private fun rank(t: DoubleArray, exclude: Set<Int>, k: Int): List<Pair<Int, Double>> {
        val tn = sqrt(dot(t, t))
        return shownIds.filter { it !in exclude }.map { it to dot(t, unit[it]) / tn }
            .sortedWith(compareBy({ -it.second }, { it.first })).take(k)
    }

    /** 3-D PCA coordinates of a unit word vector in view [view]. */
    fun project(view: Int, w: Int): DoubleArray = projectVec(view, unit[w])
    /** Linear projection of any vector given in unit-vector space (no re-normalisation). */
    fun projectVec(view: Int, x: DoubleArray, centred: Boolean = true): DoubleArray {
        val m = viewMean[view]; val w = viewW[view]
        return DoubleArray(3) { k -> var s = 0.0; for (j in 0 until d) s += ((x[j] - if (centred) m[j] else 0.0) * w[k][j]); s }
    }

    /** Which relation (and orientation) links words p -> q, or null. Returns (rel, +1 if (p,q) is an (a,b) pair, -1 if (b,a)). */
    fun relationOf(p: Int, q: Int): Pair<Int, Int>? {
        for ((r, pairs) in relations.withIndex()) for ((a, b) in pairs) {
            if (a == p && b == q) return r to 1
            if (b == p && a == q) return r to -1
        }
        return null
    }

    /**
     * The word that completes the parallelogram A − B + C according to the relation table, or -1:
     * either B→A is a relation step and C→D the same step (东京 − 日本 + 法国 → 巴黎), or B→C is a relation step
     * and A→D the same step (国王 − 男人 + 女人 → 王后).
     */
    fun expected(a: Int, b: Int, c: Int): Int {
        fun partner(x: Int, rel: Int, dir: Int): Int {
            for ((p, q) in relations[rel]) { if (dir > 0 && p == x) return q; if (dir < 0 && q == x) return p }
            return -1
        }
        relationOf(b, a)?.let { (r, dir) -> val dd = partner(c, r, dir); if (dd >= 0 && dd != a && dd != b) return dd }
        relationOf(b, c)?.let { (r, dir) -> val dd = partner(a, r, dir); if (dd >= 0 && dd != b && dd != c) return dd }
        return -1
    }

    /** Every analogy question of a relation: a1 − a2 + b2 → b1 for all ordered pairs of pairs (i != j). */
    fun questions(rel: Int): List<IntArray> {
        val pairs = relations[rel]; val out = ArrayList<IntArray>()
        for (i in pairs.indices) for (j in pairs.indices) if (i != j)
            out += intArrayOf(pairs[i].first, pairs[j].first, pairs[j].second, pairs[i].second)
        return out
    }

    companion object {
        fun dot(a: DoubleArray, b: DoubleArray): Double { var s = 0.0; for (i in a.indices) s += a[i] * b[i]; return s }
    }
}
