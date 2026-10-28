package com.bytedance.ailab.core.cnn

import com.bytedance.ailab.lab.LabAssets
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/**
 * The CNN demo's numbers, checked against the SAME asset files the app ships (app/src/main/assets/cnn/):
 *   * Kotlin forward pass == PyTorch (tools/train_cnn.py exported every intermediate) to 1e-4,
 *   * tensor shapes per layer, receptive-field recurrence vs brute-force perturbation,
 *   * held-out accuracy recomputed in Kotlin == torch's count on the shipped held-out sets,
 *   * hand-made kernels respond exactly as their closed forms predict on synthetic edges / spectra.
 */
class CnnCoreTest {
    private val lab by lazy { CnnLab(LabAssets::bytes) }

    // ------------------------------------------------------------------ reference vectors (ref.bin)
    private class Ref(val name: String, val idx: Int, val conv1: FloatArray, val pool1: FloatArray, val conv2: FloatArray,
                      val pool2: FloatArray, val logits: FloatArray, val batch: List<FloatArray>, val torchCorrect: Int)

    private fun refs(): List<Ref> {
        val bb = ByteBuffer.wrap(LabAssets.bytes("cnn/ref.bin")).order(ByteOrder.LITTLE_ENDIAN)
        assertEquals(0x434E4E52, bb.int)
        val n = bb.int
        return List(n) {
            val nb = ByteArray(bb.int); bb.get(nb); val name = String(nb)
            val m = lab.models.getValue(name)
            fun f(k: Int) = FloatArray(k).also { a -> for (i in 0 until k) a[i] = bb.float }
            val idx = bb.int
            val r1 = f(m.k1 * 26 * 26); val r2 = f(m.k1 * 13 * 13); val r3 = f(m.k2 * 11 * 11); val r4 = f(m.k2 * 5 * 5); val r5 = f(m.nOut)
            val bm = bb.int
            val batch = List(bm) { f(m.nOut) }
            Ref(name, idx, r1, r2, r3, r4, r5, batch, bb.int)
        }.also { assertEquals(0, bb.remaining()) }
    }

    private fun maxDiff(a: FloatArray, b: FloatArray): Float { assertEquals(a.size, b.size); var m = 0f; for (i in a.indices) m = maxOf(m, abs(a[i] - b[i])); return m }

    private fun isScene(name: String) = name.startsWith("scene")
    private fun nir(name: String) = name.startsWith("scene4")
    private fun input(name: String, i: Int) = lab.input(isScene(name), nir(name), i)

    @Test
    fun kotlinForwardEqualsTorchReference() {
        val rs = refs()
        assertEquals(6, rs.size)
        for (r in rs) {
            val m = lab.models.getValue(r.name)
            val a = m.forward(input(r.name, r.idx))
            val d = floatArrayOf(maxDiff(a.conv1.d, r.conv1), maxDiff(a.pool1.d, r.pool1), maxDiff(a.conv2.d, r.conv2),
                maxDiff(a.pool2.d, r.pool2), maxDiff(a.logits, r.logits))
            var db = 0f
            for ((i, z) in r.batch.withIndex()) db = maxOf(db, maxDiff(m.forward(input(r.name, i)).logits, z))
            println("CNN-REF ${r.name} (${m.params} params) sample ${r.idx}: max |kotlin − torch| conv1 %.1e pool1 %.1e conv2 %.1e pool2 %.1e logits %.1e; logits of ${r.batch.size} samples %.1e"
                .format(d[0], d[1], d[2], d[3], d[4], db))
            for (x in d) assertTrue("${r.name}: activation mismatch $x", x < 1e-4f)
            assertTrue("${r.name}: batch logits mismatch $db", db < 1e-4f)
        }
    }

    @Test
    fun tensorShapesPerLayer() {
        for (name in CnnLab.NAMES) {
            val m = lab.models.getValue(name)
            val a = m.forward(input(name, 0))
            val shapes = listOf(a.input.shape, a.conv1.shape, a.pool1.shape, a.conv2.shape, a.pool2.shape, "${a.logits.size}")
            println("CNN-SHAPE $name: ${shapes.joinToString(" → ")}")
            val c = m.cIn
            assertEquals(listOf("$c×28×28", "8×26×26", "8×13×13", "16×11×11", "16×5×5", if (isScene(name)) "5" else "10"), shapes)
        }
        assertEquals(3, lab.models.getValue("digits_learned").cIn)
        assertEquals(4, lab.models.getValue("scene4_learned").cIn)
        assertEquals(3, lab.models.getValue("scene3_learned").cIn)
        // parameter count: 8·C·9+8 + 16·8·9+16 + N·400+N
        assertEquals(8 * 3 * 9 + 8 + 16 * 8 * 9 + 16 + 10 * 400 + 10, lab.models.getValue("digits_learned").params)
    }

    @Test
    fun receptiveFieldRecurrenceMatchesPerturbation() {
        assertArrayEquals(intArrayOf(28, 26, 13, 11, 5), ReceptiveField.sides())
        assertArrayEquals(intArrayOf(1, 3, 4, 8, 10), ReceptiveField.sizes())
        assertArrayEquals(intArrayOf(1, 1, 2, 2, 4), ReceptiveField.jumps())
        for (l in 1..4) for (i in 0 until ReceptiveField.sides()[l]) {
            val w = ReceptiveField.window(l, i)
            assertEquals(ReceptiveField.sizes()[l], w[1] - w[0] + 1)
            assertEquals(i * ReceptiveField.jumps()[l], w[0])
        }
        // brute force: all-positive weights and inputs, so a +1000 bump at a pixel always reaches every cell whose
        // window contains it (max-pool picks it) and can never reach any other cell
        val rnd = java.util.Random(7)
        val m = CnnModel("probe", 1, 8, 16, 1, false, FloatArray(8 * 9) { 0.1f + rnd.nextFloat() }, FloatArray(8),
            FloatArray(16 * 8 * 9) { 0.1f + rnd.nextFloat() }, FloatArray(16), FloatArray(400) { 1f }, FloatArray(1))
        val x0 = Tensor(1, 28, 28, FloatArray(784) { 0.2f + 0.6f * rnd.nextFloat() })
        val base = m.forward(x0)
        for ((ci, cj) in listOf(0 to 0, 2 to 3, 4 to 4, 1 to 4)) {
            val seen = HashSet<Int>()
            for (p in 0 until 784) {
                val x = Tensor(1, 28, 28, x0.d.copyOf()); x.d[p] += 1000f
                val a = m.forward(x)
                if ((0 until 16).any { a.pool2[it, ci, cj] != base.pool2[it, ci, cj] }) seen += p
            }
            val wr = ReceptiveField.window(4, ci); val wc = ReceptiveField.window(4, cj)
            val want = HashSet<Int>(); for (r in wr[0]..wr[1]) for (c in wc[0]..wc[1]) want += r * 28 + c
            assertEquals("pool2 cell ($ci,$cj) receptive field", want, seen)
            println("CNN-RF pool2 cell ($ci,$cj): ${seen.size} input pixels = rows ${wr[0]}..${wr[1]} × cols ${wc[0]}..${wc[1]} (10×10)")
        }
        // the floor in the two poolings: input rows/cols 26 and 27 never reach the output
        val x = Tensor(1, 28, 28, x0.d.copyOf()); for (r in 0 until 28) { x.d[r * 28 + 27] += 1000f; x.d[27 * 28 + r] += 1000f; x.d[26 * 28 + r] += 1000f }
        assertArrayEquals(base.logits, m.forward(x).logits, 0f)
        println("CNN-RF sizes 1 → 3 → 4 → 8 → 10 px (jumps 1,1,2,2,4); input rows/cols 26–27 never reach the dense layer")
    }

    @Test
    fun heldOutAccuracyInKotlinEqualsTorch() {
        val rs = refs().associateBy { it.name }
        for (name in CnnLab.NAMES) {
            val m = lab.models.getValue(name)
            val set = if (isScene(name)) lab.scenes else lab.digits
            var ok = 0
            val perLabel = IntArray(5)
            for (i in 0 until set.count) {
                val a = m.forward(input(name, i))
                if (!isScene(name)) { if (a.top() == set.labels[i]) ok++ }
                else {
                    var all = true
                    for (c in 0 until 5) { val hit = (a.logits[c] > 0f) == SceneClass.has(set.labels[i], c); if (hit) perLabel[c]++ else all = false }
                    if (all) ok++
                }
            }
            val big = lab.heldOut(name)
            println("CNN-ACC $name: Kotlin %d/%d = %.1f%% on the shipped held-out set (torch %d); training script held-out %.2f%% on %s"
                .format(ok, set.count, 100f * ok / set.count, rs.getValue(name).torchCorrect, big * 100, lab.metrics["$name.heldout_n"]) +
                if (isScene(name)) "; per label veg/water/turf/roof/soil " + perLabel.joinToString("/") { "%.0f%%".format(100f * it / set.count) } else "")
            assertEquals("$name: Kotlin and torch disagree on the shipped set", rs.getValue(name).torchCorrect, ok)
        }
        // the task metrics the demo states
        assertTrue("digits (learned) held-out accuracy", lab.heldOut("digits_learned") > 0.93f)
        assertTrue("digits (hand-made conv1) held-out accuracy", lab.heldOut("digits_hand") > 0.80f)
        assertTrue("scenes RGB+NIR (learned) exact-match accuracy", lab.heldOut("scene4_learned") > 0.80f)
        // the sensor insight: without NIR the network cannot tell turf from vegetation as well
        val t4 = lab.labelAcc("scene4_learned", "turf"); val t3 = lab.labelAcc("scene3_learned", "turf")
        val v4 = lab.labelAcc("scene4_learned", "veg"); val v3 = lab.labelAcc("scene3_learned", "veg")
        println("CNN-NIR turf label: RGB+NIR %.1f%% vs RGB-only %.1f%%; vegetation label %.1f%% vs %.1f%%".format(t4 * 100, t3 * 100, v4 * 100, v3 * 100))
        assertTrue("NIR must help the turf label", t4 > t3 + 0.05f)
    }

    @Test
    fun handMadeKernelsRespondAsTheirClosedFormsSay() {
        val dh = lab.models.getValue("digits_hand")
        // frozen during training: biases stay exactly 0, the V-edge kernel is Sobel-x / 4, split over R, G, B
        assertArrayEquals(FloatArray(8), dh.b1, 0f)
        val sob = floatArrayOf(-1f, 0f, 1f, -2f, 0f, 2f, -1f, 0f, 1f)
        for (c in 0 until 3) for (t in 0 until 9) assertEquals(sob[t] / 12f, dh.kernel(0)[c * 9 + t], 1e-7f)
        // a vertical step edge (dark left, bright right at column 14), all three channels
        val step = Tensor(3, 28, 28).also { for (c in 0 until 3) for (y in 0 until 28) for (x in 14 until 28) it[c, y, x] = 1f }
        val a = dh.forward(step)
        for (i in 0 until 26) for (j in 0 until 26) {
            val wantV = if (j == 12 || j == 13) 1f else 0f      // only windows straddling the edge, Σ = (1+2+1)/4
            assertEquals("V-edge at ($i,$j)", wantV, a.conv1[0, i, j], 1e-5f)
            assertEquals("H-edge at ($i,$j)", 0f, a.conv1[1, i, j], 1e-5f)
            assertEquals("blur at ($i,$j)", ((j + 3 - 14).coerceIn(0, 3)) / 3f, a.conv1[4, i, j], 1e-5f)
        }
        // the mirrored edge gives −1 there: ReLU removes it (feature maps only keep the polarity a kernel likes)
        val flip = Tensor(3, 28, 28).also { for (c in 0 until 3) for (y in 0 until 28) for (x in 0 until 14) it[c, y, x] = 1f }
        assertEquals(-1f, dh.forward(flip).conv1[0, 5, 12], 1e-5f)
        assertEquals(0f, CnnOps.relu(dh.forward(flip).conv1)[0, 5, 12], 0f)
        // multispectral: the NDVI-like kernel (NIR − R) on uniform spectra — leaves light up, turf and water do not
        val sh = lab.models.getValue("scene4_hand")
        fun uniform(r: Float, g: Float, b: Float, n: Float) = Tensor(4, 28, 28).also { t -> for (y in 0 until 28) for (x in 0 until 28) { t[0, y, x] = r; t[1, y, x] = g; t[2, y, x] = b; t[3, y, x] = n } }
        val leaf = sh.forward(uniform(0.15f, 0.40f, 0.12f, 0.80f)).conv1[0, 10, 10]
        val turf = sh.forward(uniform(0.15f, 0.40f, 0.12f, 0.14f)).conv1[0, 10, 10]
        val water = sh.forward(uniform(0.07f, 0.18f, 0.36f, 0.02f)).conv1[0, 10, 10]
        assertEquals(0.80f - 0.15f, leaf, 1e-5f); assertEquals(0.14f - 0.15f, turf, 1e-5f); assertEquals(0.02f - 0.07f, water, 1e-5f)
        // ...while the "green" kernel cannot tell leaf from turf (same RGB)
        assertEquals(sh.forward(uniform(0.15f, 0.40f, 0.12f, 0.80f)).conv1[1, 3, 3], sh.forward(uniform(0.15f, 0.40f, 0.12f, 0.14f)).conv1[1, 3, 3], 1e-6f)
        // the RGB-only sensor: the NIR − R kernel loses its NIR slice and goes blind (≤ 0 everywhere → ReLU 0)
        val s3 = lab.models.getValue("scene3_hand")
        val blind = s3.forward(lab.input(true, false, 0)).conv1
        for (i in 0 until 26) for (j in 0 until 26) assertTrue(blind[0, i, j] <= 1e-6f)
        println("CNN-HAND V-edge on a step edge = 1.000 at the 2 straddling columns, 0 elsewhere; NIR−R: leaf %.2f, turf %.2f, water %.2f".format(leaf, turf, water))
    }

    @Test
    fun perChannelPartialSumsAddUpToTheFeatureMap() {
        for (name in listOf("digits_learned", "scene4_learned")) {
            val m = lab.models.getValue(name)
            val x = input(name, 3)
            val a = m.forward(x)
            for ((k, i, j) in listOf(Triple(0, 0, 0), Triple(3, 12, 7), Triple(7, 25, 25), Triple(5, 9, 20))) {
                val p = CnnOps.partials(x, m.w1, k, i, j)
                assertEquals(m.cIn, p.size)
                assertEquals(a.conv1[k, i, j], p.sum() + m.b1[k], 1e-5f)
            }
        }
    }

    @Test
    fun slantProbeFindsAKernelForTheSeven() {
        val i7 = lab.digitShow.first { lab.digits.labels[it] == 7 }
        val a = lab.acts(false, true, true, i7)
        val s = SlantProbe.selectivity(a)
        val best = s.indices.maxBy { s[it] }
        println("CNN-SLANT digit 7 (shipped #$i7, predicted ${a.top()} %.0f%%): selectivity per learned kernel ${s.joinToString { "%+.2f".format(it) }} → best #${best + 1}".format(a.probs[a.top()] * 100))
        assertEquals(7, a.top())
        assertTrue(s[best] > 0f)
        assertEquals(1f, CnnScale.robust(floatArrayOf()), 0f)
        assertEquals(2f, CnnScale.robust(floatArrayOf(-2f, 1f, 0.5f), 1f), 0f)
    }
}
