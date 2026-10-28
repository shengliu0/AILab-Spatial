package com.bytedance.ailab.core.mlp

import com.bytedance.ailab.lab.LabAssets
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.random.Random

/** Thread selection = the k largest |w a| contributions, checked against a brute-force full sort. */
class TopKTest {
    private fun brute(w: FloatArray, a: FloatArray, nOut: Int, nIn: Int, k: Int, dsts: IntArray? = null, sign: Int = 0): List<Edge> {
        val all = ArrayList<Pair<Edge, Int>>(); var order = 0
        val rows = dsts ?: IntArray(nOut) { it }
        for (i in 0 until nIn) { if (a[i] == 0f) continue; for (j in rows) {
            val v = w[j * nIn + i] * a[i]
            if (v == 0f || (sign > 0 && v < 0f) || (sign < 0 && v > 0f)) continue
            all += Edge(i, j, v) to order++ } }
        return all.sortedWith(compareBy({ -abs(it.first.value) }, { it.second })).take(k).map { it.first }
    }

    @Test
    fun matchesBruteForceOnRandomLayers() {
        val rnd = Random(42)
        var cases = 0
        repeat(200) {
            val nIn = 1 + rnd.nextInt(120); val nOut = 1 + rnd.nextInt(40)
            val w = FloatArray(nIn * nOut) { rnd.nextFloat() * 2 - 1 }
            val a = FloatArray(nIn) { if (rnd.nextFloat() < 0.4f) 0f else rnd.nextFloat() * 3 }   // ReLU-like: many zeros
            val k = rnd.nextInt(60)
            val sign = rnd.nextInt(3) - 1
            val dsts = if (rnd.nextBoolean()) null else IntArray(1 + rnd.nextInt(nOut)) { rnd.nextInt(nOut) }.distinct().toIntArray()
            assertEquals(brute(w, a, nOut, nIn, k, dsts, sign), TopK.strongest(w, a, nOut, nIn, k, dsts, sign))
            cases++
        }
        println("TOPK $cases random layers == brute-force sort")
    }

    @Test
    fun edgeCases() {
        val w = floatArrayOf(1f, -2f, 3f, 0f, 2f, -1f)               // 2 x 3
        val a = floatArrayOf(1f, 1f, 0f)
        // k larger than the number of non-zero contributions: all of them, strongest first, ties keep (src, dst) order
        val all = TopK.strongest(w, a, 2, 3, 99)
        assertEquals(listOf(Edge(1, 0, -2f), Edge(1, 1, 2f), Edge(0, 0, 1f)), all)     // w[1,0] = 0 contributes nothing
        assertEquals(emptyList<Edge>(), TopK.strongest(w, FloatArray(3), 2, 3, 5))       // blank input: no threads
        assertEquals(emptyList<Edge>(), TopK.strongest(w, a, 2, 3, 0))
        assertEquals(listOf(Edge(1, 0, -2f)), TopK.strongest(w, a, 2, 3, 5, sign = -1))
    }

    @Test
    fun supportPathFollowsTheLargestPositiveContributions() {
        val net = Mlp.load(LabAssets.bytes("mlp/mlp_w.bin"))
        val samples = DigitAssets.samples(LabAssets.bytes("mlp/samples.bin"))
        val act = net.forward(StrokeRaster.rasterise(samples[3]))
        val p = TopK.supportPath(net, act, 3)
        assertEquals(4, p.toOut.size)
        assertTrue(p.toOut.all { it.dst == 3 && it.value > 0f } && p.toH2.all { it.value > 0f } && p.toH1.all { it.value > 0f })
        // stage 1 = the 4 largest of w3[3, j] * h2[j], by brute force
        val want = (0 until 32).map { it to net.w3[3 * 32 + it] * act.h2[it] }.filter { it.second > 0 }.sortedByDescending { it.second }.take(4).map { it.first }
        assertEquals(want, p.toOut.map { it.src })
        // every hidden-2 neuron on the path gets its own strongest inputs; pixels feed hidden-1 neurons of the path
        assertTrue(p.toH2.map { it.dst }.toSet() == p.toOut.map { it.src }.toSet())
        assertTrue(p.toH1.map { it.dst }.toSet() == p.toH2.map { it.src }.toSet())
        val share = p.toOut.sumOf { it.value.toDouble() } / (0 until 32).sumOf { maxOf(0f, net.w3[3 * 32 + it] * act.h2[it]).toDouble() }
        println("TOPK path for '3': h2 %s, %d h1 links, %d pixel links; top-4 carry %.0f%% of the positive evidence"
            .format(p.toOut.map { it.src }, p.toH2.size, p.toH1.size, share * 100))
    }
}
