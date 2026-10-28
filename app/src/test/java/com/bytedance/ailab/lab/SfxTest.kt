package com.bytedance.ailab.lab

import org.junit.Assert.assertTrue
import org.junit.Test

/** Every synthesised sound is non-empty, finite, peak-normalised (audible but never clipping) and short. */
class SfxTest {
    @Test
    fun soundsAreWellFormed() {
        val all = Sfx.renderAll()
        assertTrue(all.size >= 10)
        for ((name, s) in all) {
            val peak = s.maxOf { kotlin.math.abs(it) }
            println("SFX $name: ${s.size} samples (%.0f ms), peak %.2f".format(s.size / 44.1, peak))
            assertTrue("$name empty", s.size > 400)
            assertTrue("$name non-finite", s.all { it.isFinite() })
            assertTrue("$name peak $peak", peak in 0.5f..0.95f)
            assertTrue("$name too long", s.size <= 44100 * 1.2)
        }
        Sfx.play(Sfx.CLACK)              // JVM: not initialised -> must be a harmless no-op
    }
}
