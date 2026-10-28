package com.bytedance.ailab.core.mlp

import com.bytedance.ailab.demos.mlp.MlpDemo
import com.bytedance.ailab.lab.LabAssets
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.math.sin
import kotlin.random.Random

/**
 * End to end through the SAME path a hand takes: pinch a start cell on the pad (the stroke starts at that
 * cell's centre — gaze picks it), then hand deltas in volume metres via onDrag, release. The hand is
 * imperfect on purpose: tremor (+-4 mm), a different gain per axis (0.85-1.2), a little depth wobble.
 * The network must read the digit that was written.
 */
class MlpDemoDrawTest {
    private val samples = DigitAssets.samples(LabAssets.bytes("mlp/samples.bin"))
    private val PAD = 0.27f; private val GAIN = 1.4f; private val CELLS = 5

    private fun write(strokes: List<FloatArray>, rnd: Random) {
        val gx = 0.85f + 0.35f * rnd.nextFloat(); val gy = 0.85f + 0.35f * rnd.nextFloat()
        val f1 = 3f + 6f * rnd.nextFloat(); val f2 = 3f + 6f * rnd.nextFloat(); val ph = 6f * rnd.nextFloat()
        for (s in strokes) {
            val n = s.size / 2
            val col = (s[0] * CELLS).toInt().coerceIn(0, CELLS - 1); val row = (s[1] * CELLS).toInt().coerceIn(0, CELLS - 1)
            val cell = "Pad_${row}_$col"
            MlpDemo.onDragStart(cell)
            var px = 0f; var py = 0f
            for (i in 1 until n) {
                val t = i.toFloat() / n
                val tx = (s[i * 2] - s[0]) * PAD / GAIN * gx + 0.004f * sin(f1 * 6.283f * t + ph)
                val ty = -(s[i * 2 + 1] - s[1]) * PAD / GAIN * gy + 0.004f * sin(f2 * 6.283f * t)
                MlpDemo.onDrag(cell, tx - px, ty - py, 0.003f * (rnd.nextFloat() - 0.5f))
                px = tx; py = ty
                if (i % 3 == 0) MlpDemo.update(1f / 60f)
            }
            MlpDemo.onDragEnd(cell)
            repeat(10) { MlpDemo.update(1f / 60f) }
        }
        repeat(150) { MlpDemo.update(1f / 60f) }
    }

    @Test
    fun handWrittenDigitsThroughTheDragApiAreRead() {
        MlpDemo.runHeadless(File("build/review-cache").apply { mkdirs() }, 2.5f)
        val rnd = Random(2026)
        var ok = 0; var n = 0
        val wrong = ArrayList<String>()
        for (trial in 0 until 3) for (d in 0 until 10) {
            MlpDemo.onTap("Clear"); repeat(30) { MlpDemo.update(1f / 60f) }
            write(samples[d], rnd)
            n++
            if (MlpDemo.prediction == d) ok++ else wrong += "$d->${MlpDemo.prediction}"
        }
        // two-stroke digits: the second stroke starts at a gaze-snapped cell centre, not where the hand meant
        val four = listOf(floatArrayOf(0.36f, 0.18f, 0.33f, 0.40f, 0.31f, 0.58f, 0.50f, 0.58f, 0.68f, 0.58f),
                          floatArrayOf(0.60f, 0.30f, 0.60f, 0.55f, 0.60f, 0.86f))
        val seven = listOf(floatArrayOf(0.30f, 0.18f, 0.50f, 0.18f, 0.70f, 0.18f, 0.60f, 0.45f, 0.48f, 0.84f),
                           floatArrayOf(0.42f, 0.50f, 0.56f, 0.50f, 0.70f, 0.50f))
        val multi = listOf(4 to four, 7 to seven).map { (d, st) ->
            val dense = st.map { s -> val out = ArrayList<Float>()
                for (i in 0 until s.size / 2 - 1) for (k in 0 until 10) { val t = k / 10f
                    out += s[i * 2] + (s[i * 2 + 2] - s[i * 2]) * t; out += s[i * 2 + 1] + (s[i * 2 + 3] - s[i * 2 + 1]) * t }
                out += s[s.size - 2]; out += s[s.size - 1]; out.toFloatArray() }
            MlpDemo.onTap("Clear"); repeat(30) { MlpDemo.update(1f / 60f) }
            write(dense, rnd)
            "$d->${MlpDemo.prediction} (%.0f%%)".format(MlpDemo.confidence * 100) to (MlpDemo.prediction == d)
        }
        println("DRAW via drag API with tremor and gain error: $ok/$n single-stroke digits read correctly ${if (wrong.isEmpty()) "" else "(wrong: $wrong)"}; " +
            "two-stroke with snapped starts: ${multi.joinToString { it.first }}")
        assertTrue("only $ok/$n hand-written digits read", ok >= n - 2)
        assertTrue("two-stroke digits misread: $multi", multi.all { it.second })
    }
}
