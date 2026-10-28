package com.bytedance.ailab.lab

import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Every pinch must be VISIBLY answered (users asked "I pinched it — what did it do?"). For every
 * demo, every panel twin (TapTarget) is tapped in both languages and must produce
 *   - a note: the demo's tapMessage or the generic "✓ object → action" label, and
 *   - an anchor just above a real node (finite, inside the volume), where the note is shown.
 * Panel-only actions (no object to point at, e.g. solar "next planet") are listed explicitly.
 */
class TapFeedbackTest {
    private val panelOnly = setOf("solar/PlanetNext")

    @Test
    fun everyTapTargetGetsANoteAtItsObject() {
        val cache = File("build/review-cache").apply { mkdirs() }
        val bad = ArrayList<String>()
        var n = 0
        for (en in listOf(false, true)) {
            Lang.set(en)
            for (d in DemoRegistry.all.filterIsInstance<SpecDemo>()) {
                d.runHeadless(cache, 1f)
                for (t in d.tapTargets) {
                    d.onTap(t.node); d.update(1f / 60f)
                    val msg = d.tapMessage(t.node) ?: ("✓ " + tr(t.zh, t.en))
                    val at = d.anchorOf(t.node)
                    val key = "${d.id}/${t.node}"
                    if (en && msg.any { Character.UnicodeScript.of(it.code) == Character.UnicodeScript.HAN }) bad += "$key: CJK in EN note '$msg'"
                    if (key in panelOnly) { n++; continue }
                    if (at == null) bad += "$key: no node to anchor the note"
                    else if (at.any { !it.isFinite() } || kotlin.math.abs(at[0]) > 1.1f || at[1] !in -0.6f..0.75f || kotlin.math.abs(at[2]) > 1.1f)
                        bad += "$key: note anchor outside the volume ${at.joinToString { "%.2f".format(it) }}"
                    if (!en) println("TAP $key -> \"$msg\" at (${at?.joinToString { "%.2f".format(it) }})")
                    n++
                }
            }
        }
        Lang.set(false)
        println("TAP checked $n target×language pairs; problems: ${if (bad.isEmpty()) "none" else bad.joinToString("; ")}")
        assertTrue(bad.joinToString("\n"), bad.isEmpty())
        assertTrue("no tap targets checked", n >= 2 * DemoRegistry.all.count { it.tapTargets.isNotEmpty() })
    }
}
