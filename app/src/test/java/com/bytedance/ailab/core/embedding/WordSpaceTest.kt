package com.bytedance.ailab.core.embedding

import com.bytedance.ailab.lab.LabAssets
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * The embedding demo's maths against the training script's own numbers (tools/train_embedding.py writes
 * src/test/resources/embedding/ref.txt from the float32 vectors it ships):
 *   * cosine nearest neighbours == PyTorch (torch.nn.functional.normalize, float64)
 *   * 3CosAdd analogies (Mikolov et al. 2013: argmax cos(x, â − b̂ + ĉ), A, B, C excluded) == numpy
 *   * analogy accuracy on the relation test set ≥ 80 % top-1, far above a shuffled-vector baseline
 *   * 3-D PCA projections == numpy within 1e-5
 *   * training determinism: emb.bin's SHA-256 is the one two independent runs of the script produced
 */
class WordSpaceTest {
    private val bytes = LabAssets.bytes("embedding/emb.bin")
    private val ws = WordSpace(bytes)
    private val ref = File("src/test/resources/embedding/ref.txt").readLines().map { it.split(" ") }

    /** Recorded 2026-10-27 from two runs of `python3 tools/train_embedding.py` (CPU, 1 thread): identical. */
    private val PINNED_SHA = "e5c254ff0668c236a3e581deadab97072e5efd11a9edf752e48ac04f2d07830e"

    @Test fun vocabularyAndRelations() {
        println("EMBEDDING vocab ${ws.v} words (${ws.shownIds.size} shown), D = ${ws.d}, relations ${ws.relations.map { it.size }}, views ${ws.viewIds}")
        assertTrue(ws.v in 100..200)
        assertEquals(listOf("gender", "capital", "comparative", "all"), ws.viewIds)
        assertEquals("王后", ws.zh[ws.relations[0].first { ws.zh[it.first] == "国王" }.second])
        assertEquals("Paris", ws.en[ws.indexOf("巴黎")])
        // every shown word has an English label (the readout never shows bare CJK in English mode)
        for (i in ws.shownIds) assertTrue(ws.zh[i], ws.en[i].isNotEmpty() && ws.en[i].all { it.code < 128 })
    }

    @Test fun nearestNeighboursMatchPyTorch() {
        var n = 0
        for (l in ref.filter { it[0] == "nn" }) {
            val w = l[1].toInt()
            val got = ws.nearest(w, 5)
            for (k in 0 until 5) {
                assertEquals("nn of ${ws.zh[w]} #$k", l[2 + 2 * k].toInt(), got[k].first)
                assertEquals(l[3 + 2 * k].toDouble(), got[k].second, 1e-6)
            }
            n++
            println("EMBEDDING nn ${ws.en[w]}: " + got.joinToString { "%s %.3f".format(ws.en[it.first], it.second) })
        }
        assertEquals(6, n)
    }

    @Test fun analogiesMatchNumpy() {
        for (l in ref.filter { it[0] == "an" }) {
            val (a, b, c) = Triple(l[1].toInt(), l[2].toInt(), l[3].toInt())
            val got = ws.analogy(a, b, c, 3)
            for (k in 0 until 3) {
                assertEquals(l[4 + 2 * k].toInt(), got[k].first)
                assertEquals(l[5 + 2 * k].toDouble(), got[k].second, 1e-6)
            }
            println("EMBEDDING ${ws.en[a]} - ${ws.en[b]} + ${ws.en[c]} -> " + got.joinToString { "%s %.3f".format(ws.en[it.first], it.second) })
        }
        val king = ws.indexOf("国王"); val man = ws.indexOf("男人"); val woman = ws.indexOf("女人")
        assertEquals(ws.indexOf("王后"), ws.analogy(king, man, woman)[0].first)
        assertEquals(ws.indexOf("巴黎"), ws.analogy(ws.indexOf("东京"), ws.indexOf("日本"), ws.indexOf("法国"))[0].first)
    }

    private fun accuracy(u: Array<DoubleArray>): IntArray {
        var ok = 0; var tot = 0
        for (r in ws.relations.indices) for (q in ws.questions(r)) {
            val t = DoubleArray(ws.d) { u[q[0]][it] - u[q[1]][it] + u[q[2]][it] }
            val best = ws.shownIds.filter { it != q[0] && it != q[1] && it != q[2] }.maxByOrNull { WordSpace.dot(u[it], t) }
            if (best == q[3]) ok++; tot++
        }
        return intArrayOf(ok, tot)
    }

    @Test fun analogyAccuracyBeatsShuffledBaseline() {
        val (ok, tot) = accuracy(ws.unit).let { it[0] to it[1] }
        val perm = (0 until ws.v).shuffled(java.util.Random(7))
        val shuffled = Array(ws.v) { ws.unit[perm[it]] }
        val okS = accuracy(shuffled)[0]
        val per = ws.relations.indices.map { r -> ws.questions(r).count { q -> ws.analogy(q[0], q[1], q[2], 1)[0].first == q[3] } to ws.questions(r).size }
        println("EMBEDDING analogy top-1 (A,B,C excluded, candidates = ${ws.shownIds.size} shown words): $ok/$tot = %.1f %%; per relation $per; shuffled-vector baseline $okS/$tot = %.1f %%"
            .format(100.0 * ok / tot, 100.0 * okS / tot))
        assertEquals(ref.first { it[0] == "acc" }[1].toInt(), ok)
        assertTrue("accuracy ${ok}/$tot below 80 %", ok >= 0.80 * tot)
        assertTrue("shuffled baseline too high", okS <= 0.15 * tot)
        // the relation table and the analogy questions agree
        for (r in ws.relations.indices) for (q in ws.questions(r)) assertEquals(q[3], ws.expected(q[0], q[1], q[2]))
        assertEquals(-1, ws.expected(ws.indexOf("国王"), ws.indexOf("男人"), ws.indexOf("东京")))
    }

    @Test fun pcaProjectionMatchesNumpy() {
        var worst = 0.0; var n = 0
        for (l in ref.filter { it[0] == "proj" }) {
            val p = ws.project(l[1].toInt(), l[2].toInt())
            for (k in 0 until 3) worst = maxOf(worst, abs(p[k] - l[3 + k].toDouble()))
            n++
        }
        println("EMBEDDING PCA: $n projected points, max |Kotlin - numpy| = %.2e".format(worst))
        assertEquals(4 * ws.shownIds.size, n)
        assertTrue(worst < 1e-5)
        // linearity: A − B + C projects to pA − pB + pC (the drawn parallelogram is exact)
        val a = ws.indexOf("国王"); val b = ws.indexOf("男人"); val c = ws.indexOf("女人")
        val pt = ws.projectVec(0, ws.target(a, b, c), centred = false).let { q -> val m0 = ws.projectVec(0, DoubleArray(ws.d), true); DoubleArray(3) { q[it] + m0[it] } }
        val pa = ws.project(0, a); val pb = ws.project(0, b); val pc = ws.project(0, c)
        for (k in 0 until 3) assertEquals(pa[k] - pb[k] + pc[k], pt[k], 1e-9)
    }

    @Test fun trainingIsDeterministicAndAssetsSmall() {
        val sha = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
        println("EMBEDDING emb.bin sha256 $sha")
        assertEquals(ref.first { it[0] == "sha256" }[1], sha)
        assertEquals(PINNED_SHA, sha)
        val total = File("src/main/assets/embedding").listFiles()!!.sumOf { it.length() }
        println("EMBEDDING assets $total bytes")
        assertTrue(total <= 2_000_000)
        val atlas = WordAtlas(LabAssets.bytes("embedding/words.bin"))
        for (i in ws.shownIds) for (e in listOf(false, true)) { val r = atlas.rect(i, e); assertTrue(r[2] > 0 && r[0] + r[2] <= atlas.w && r[1] + r[3] <= atlas.h) }
        assertTrue(atlas.alpha.any { it.toInt() != 0 })
        // unit vectors really are unit
        for (i in 0 until ws.v) assertEquals(1.0, sqrt(WordSpace.dot(ws.unit[i], ws.unit[i])), 1e-12)
    }
}
