package com.bytedance.ailab.core.diffusion

import com.bytedance.ailab.demos.diffusion.DiffusionDemo
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * The demo's per-frame cost on the Mac JVM (brief: <= ~2 ms; the headset is ~5-10x slower, which the
 * network's wall-clock budget absorbs by stretching the playback). Plays one full 200-step DDPM generation
 * (60 000 network calls) at 60 Hz and one forward sweep, and reports the mean / 95th percentile / worst frame.
 */
class DiffusionDemoFrameTest {
    @Test
    fun frameCost() {
        val d = DiffusionDemo
        d.runHeadless(File("build/review-cache").apply { mkdirs() }, 0.5f)
        repeat(2) { d.generate(); var k = 0; while (d.generating && k++ < 4000) d.update(1f / 60f) }   // JIT warm-up
        fun measure(label: String, start: () -> Unit, until: () -> Boolean): DoubleArray {
            start()
            val ms = ArrayList<Double>()
            var k = 0
            while (!until() && k++ < 5000) { val t0 = System.nanoTime(); d.update(1f / 60f); ms += (System.nanoTime() - t0) / 1e6 }
            val s = ms.sorted()
            val out = doubleArrayOf(ms.average(), s[(s.size * 0.95).toInt().coerceAtMost(s.size - 1)], s.last(), ms.size.toDouble())
            println("DIFFUSION frame %-28s frames %4.0f   mean %.2f ms   p95 %.2f ms   worst %.2f ms".format(label, out[3], out[0], out[1], out[2]))
            return out
        }
        d.setSteps(200)
        val g = measure("generate (DDPM 200 steps)", { d.generate() }, { !d.generating })
        d.resetToData()
        val f = measure("forward sweep t: 0 -> T", { d.onTap("TimeHandle") }, { d.readoutText.startsWith("t = 200") })
        assertTrue("generation frames average ${g[0]} ms", g[0] < 2.5)
        assertTrue("forward frames average ${f[0]} ms", f[0] < 2.5)
    }
}
