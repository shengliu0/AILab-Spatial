package com.bytedance.ailab.core.mlp

import com.bytedance.ailab.lab.LabAssets
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/**
 * The Kotlin MLP is the SAME function PyTorch trained: tools/train_mlp.py exported PyTorch's own activations for
 * 15 inputs (the 10 app sample digits, 4 held-out examples, 1 noisy one) to assets/mlp/ref.bin; the shipped
 * weights (assets/mlp/mlp_w.bin) must reproduce every hidden activation, logit and probability to 1e-4.
 * Then the task metric: accuracy on the shipped 1000-example slice of the held-out synthetic set
 * (assets/mlp/heldout.bin), which the training script also scored (it printed "993/1000 correct").
 */
class MlpReferenceTest {
    private val net = Mlp.load(LabAssets.bytes("mlp/mlp_w.bin"))

    @Test
    fun kotlinMatchesPyTorchOnExportedVectors() {
        val fb = ByteBuffer.wrap(LabAssets.bytes("mlp/ref.bin")).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val n = fb.get().toInt()
        assertTrue("reference vectors", n >= 10)
        var worst = 0.0
        val act = net.newAct()
        for (v in 0 until n) {
            val x = FloatArray(784).also { fb.get(it) }
            val h1 = FloatArray(64).also { fb.get(it) }; val h2 = FloatArray(32).also { fb.get(it) }
            val z = FloatArray(10).also { fb.get(it) }; val p = FloatArray(10).also { fb.get(it) }
            net.forward(x, act)
            for ((mine, ref) in listOf(act.h1 to h1, act.h2 to h2, act.logits to z, act.probs to p))
                for (k in ref.indices) worst = maxOf(worst, abs(mine[k] - ref[k]).toDouble())
            assertEquals("vector $v: predicted class", Mlp.argmax(p), act.predicted)
        }
        assertEquals("ref.bin fully consumed", 0, fb.remaining())
        println("MLP ref: $n vectors, worst |Kotlin - PyTorch| = %.2e (tolerance 1e-4); T = %.2f, scales %.3f / %.3f"
            .format(worst, net.temperature, net.actScale1, net.actScale2))
        assertTrue("Kotlin differs from PyTorch by $worst", worst < 1e-4)
    }

    @Test
    fun heldOutSyntheticAccuracy() {
        val bb = ByteBuffer.wrap(LabAssets.bytes("mlp/heldout.bin")).order(ByteOrder.LITTLE_ENDIAN)
        val n = bb.int
        val px = ByteArray(n * 784).also { bb.get(it) }
        val labels = ByteArray(n).also { bb.get(it) }
        val act = net.newAct(); val x = FloatArray(784)
        var correct = 0
        val perClass = IntArray(10); val perTotal = IntArray(10)
        for (s in 0 until n) {
            for (k in 0 until 784) x[k] = (px[s * 784 + k].toInt() and 0xFF) / 255f
            net.forward(x, act)
            val y = labels[s].toInt()
            perTotal[y]++
            if (act.predicted == y) { correct++; perClass[y]++ }
        }
        val acc = correct.toDouble() / n
        println("MLP held-out synthetic accuracy: $correct/$n = %.1f %% (per digit: %s)".format(acc * 100,
            (0 until 10).joinToString(" ") { "$it:${perClass[it]}/${perTotal[it]}" }))
        assertEquals("same count as the training script printed for these bytes", 993, correct)
        assertTrue("accuracy %.3f below 95 %%".format(acc), acc >= 0.95)
    }

    @Test
    fun calibratedProbabilitiesAreAProperDistribution() {
        val act = net.forward(FloatArray(784))
        assertEquals(1.0, act.probs.sum().toDouble(), 1e-5)
        assertTrue(act.probs.all { it in 0f..1f })
        // temperature only rescales the logits: it never changes the predicted class
        val p1 = FloatArray(10); Mlp.softmax(act.logits, 1f, p1)
        assertEquals(Mlp.argmax(p1), act.predicted)
    }
}
