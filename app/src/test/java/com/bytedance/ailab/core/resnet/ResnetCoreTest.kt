package com.bytedance.ailab.core.resnet

import com.bytedance.ailab.lab.LabAssets
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.security.MessageDigest
import kotlin.math.abs

/**
 * The shipped measurements of tools/train_resnet.py (plain vs residual MLPs, two spirals) — loaded from the very
 * asset bytes the app ships. Reference: He, Zhang, Ren & Sun, "Deep Residual Learning for Image Recognition"
 * (CVPR 2016, arXiv 1512.03385), §4.1: a deeper PLAIN net has HIGHER training error ("degradation"); the
 * residual net of the same depth does not.
 */
class ResnetCoreTest {
    private val logs by lazy { ResnetLogs(LabAssets.bytes(ResnetLogs.PATH)) }

    private fun sha(path: String) = MessageDigest.getInstance("SHA-256").digest(LabAssets.bytes(path)).joinToString("") { "%02x".format(it) }

    @Test
    fun logsLoadAndAreSelfConsistent() {
        assertTrue(logs.depths.contentEquals(intArrayOf(4, 8, 16, 32, 56)))
        assertEquals(60, logs.epochs)
        for (res in listOf(false, true)) for (d in logs.depths) {
            val r = logs.run(res, d)
            assertEquals(61, r.loss.size); assertEquals(61, r.acc.size); assertEquals(61, r.g.size)
            for (e in 0..60) {
                assertEquals(d, r.g[e].size)
                assertTrue(r.loss[e].isFinite() && r.loss[e] > 0f)
                assertTrue(r.acc[e] in 0f..1f)
                assertTrue(r.g[e].all { it.isFinite() && it >= 0f })
            }
            // epoch axis is monotone: interpolation hits the samples exactly and is linear in between
            assertEquals(r.loss[7], r.lossAt(7f), 1e-7f)
            assertEquals((r.loss[7] + r.loss[8]) / 2, r.lossAt(7.5f), 1e-6f)
            assertEquals(r.loss[60], r.lossAt(99f), 0f)
        }
        // both kinds start from chance on a balanced 2-class task: loss ~ ln 2 = 0.693 (plain), acc ~ 50 %
        assertEquals(Spirals.CHANCE_LOSS, logs.run(false, 8).loss[0], 0.05f)
    }

    @Test
    fun degradationIsInTheMeasurements() {
        val p8 = logs.run(false, 8).loss[60]; val p16 = logs.run(false, 16).loss[60]
        val p32 = logs.run(false, 32).loss[60]; val p56 = logs.run(false, 56).loss[60]
        val r56 = logs.run(true, 56).loss[60]
        println("RESNET final train loss  plain 8 %.4f  16 %.4f  32 %.4f  56 %.4f  | residual 56 %.4f".format(p8, p16, p32, p56, r56))
        println("RESNET final train acc   plain 8 %.4f  56 %.4f  | residual 56 %.4f".format(logs.run(false, 8).acc[60], logs.run(false, 56).acc[60], logs.run(true, 56).acc[60]))
        // the measured numbers (tools/train_resnet.py, seed 1)
        assertEquals(0.0174f, p8, 5e-4f); assertEquals(0.6962f, p56, 5e-4f); assertEquals(0.0247f, r56, 5e-4f)
        assertTrue("plain-56 trains WORSE than plain-8", p56 > 10 * p8)
        assertTrue("plain-56 never leaves chance", abs(p56 - Spirals.CHANCE_LOSS) < 0.01f)
        assertTrue("the plain net starts to degrade at 32 layers", p32 > p16 && p16 <= p8)
        assertTrue("residual-56 <= plain-56", r56 <= p56)
        assertTrue("residual-56 reaches >= 99 % training accuracy", logs.run(true, 56).acc[60] >= 0.99f)
        for (d in listOf(8, 16, 32, 56)) assertTrue("residual $d <= 0.03", logs.run(true, d).loss[60] < 0.03f)
    }

    @Test
    fun gradientsFadeInThePlainNetButNotInTheResidualNet() {
        val p = logs.run(false, 56); val r = logs.run(true, 56)
        val ratioEnd = r.g[60][0] / p.g[60][0]
        val ratio0 = r.g[0][0] / p.g[0][0]
        val fadePlain = p.g[60][55] / p.g[60][0]
        val fadeRes = r.g[60][55] / r.g[60][0]
        println("RESNET |dL/dW_0| 56 layers: plain %.3e vs residual %.3e at init (x%.0f), %.3e vs %.3e after 60 epochs (x%.3g)"
            .format(p.g[0][0], r.g[0][0], ratio0, p.g[60][0], r.g[60][0], ratioEnd))
        println("RESNET head/stem gradient ratio after training: plain %.3g, residual %.3g".format(fadePlain, fadeRes))
        assertEquals(3.792e-7f, p.g[60][0], 0.01e-7f)
        assertEquals(9.636e-2f, r.g[60][0], 0.01e-2f)
        assertTrue("residual stem gradient >= 1e5 x plain", ratioEnd > 1e5f)
        assertTrue("already at init the residual stem gets more gradient", ratio0 > 10f)
        assertTrue("plain gradient fades by > 4 decades from head to stem", fadePlain > 1e4f)
        assertTrue("residual gradient stays within 1 decade", fadeRes in 0.1f..10f)
        // log-space interpolation of the gradient is exact at the samples
        assertEquals(p.g[30][10], p.gradAt(10, 30f), p.g[30][10] * 1e-5f)
    }

    @Test
    fun kotlinForwardMatchesPyTorch() {
        val nets = MlpNet.loadAll(LabAssets.bytes(MlpNet.PATH))
        val refs = ResnetRefs(LabAssets.bytes(ResnetRefs.PATH), logs.depths)
        assertEquals(10, nets.size)
        var worst = 0f
        for ((key, ref) in refs.logits) {
            val net = nets.getValue(key)
            for (i in 0 until refs.n) {
                val o = net.forward(refs.x[i * 2], refs.x[i * 2 + 1])
                for (c in 0..1) worst = maxOf(worst, abs(o[c] - ref[i * 2 + c]))
            }
        }
        println("RESNET Kotlin vs PyTorch logits: max |diff| = %.2e over %d nets x %d inputs".format(worst, nets.size, refs.n))
        assertTrue("max diff $worst", worst < 1e-4f)
        // task metric: the trained residual-56 classifies the two (noiseless) spiral arms; plain-56 is at chance
        fun armAcc(net: MlpNet): Float {
            var ok = 0; var n = 0
            for (cls in 0..1) { val a = Spirals.arm(cls, 400); for (i in 0 until 400) { if ((net.p1(a[i * 2], a[i * 2 + 1]) > 0.5f) == (cls == 1)) ok++; n++ } }
            return ok.toFloat() / n
        }
        val accR = armAcc(nets.getValue(true to 56)); val accP = armAcc(nets.getValue(false to 56))
        println("RESNET spiral-arm accuracy (800 noiseless points): residual-56 %.3f, plain-56 %.3f".format(accR, accP))
        assertTrue(accR >= 0.97f); assertTrue(accP in 0.4f..0.6f)
    }

    @Test
    fun shippedFilesAreTheDeterministicTrainingOutput() {
        // tools/train_resnet.py run twice on 2026-10-27 printed these same hashes (torch 2.11, CPU, deterministic algorithms)
        assertEquals("0184a5c5a75a8c566c9342dfd897a080d263fd84dc9623da4d322f2c040c81b0", sha(ResnetLogs.PATH))
        assertEquals("00164673ae3ce0e5697e0596e32de8be318ea1cfdf491bb929bc0b52effb3a54", sha(MlpNet.PATH))
        assertEquals("a09c2dcc624bee31f5da75acf8d9ddb790991485c1035df8157166e141dadadc", sha(ResnetRefs.PATH))
    }
}
