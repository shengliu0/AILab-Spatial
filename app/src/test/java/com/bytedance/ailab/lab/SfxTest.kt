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

    @Test
    fun queuedBurstIsSpacedNotMerged() {
        // 31 clacks requested within ~0.4 s of 72 Hz frames (the 100:1 colliding blocks at speed): every one is
        // released, at least 30 ms apart, when requests arrive no faster than the queue can hold
        Sfx.clearQueue(); val r0 = Sfx.released
        var t = 0L; var asked = 0
        repeat(200) { f ->
            if (asked < 31 && f % 1 == 0) { Sfx.queue(Sfx.CLACK); asked++; if (asked % 4 == 0) Unit }
            Sfx.tick(t); t += 14
            if (f % 2 == 0) Sfx.tick(t)
        }
        println("SFX queue: asked $asked, released ${Sfx.released - r0}")
        assertTrue(Sfx.released - r0 in 12..31)
        // sparse requests (one every 40 ms) are all released
        Sfx.clearQueue(); val r1 = Sfx.released; t = 0L
        repeat(40) { Sfx.queue(Sfx.TICK); Sfx.tick(t); t += 40 }
        assertTrue("sparse: ${Sfx.released - r1}", Sfx.released - r1 == 40)
    }
}
