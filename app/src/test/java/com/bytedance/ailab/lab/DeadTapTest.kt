package com.bytedance.ailab.lab

import com.bytedance.ailab.lab.spec.Node
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * "I pinched it and nothing happened": for every demo, in its default state and every review state, press
 * each panel twin / pinch target and check that SOMETHING changed — the scene graph (transforms, visibility,
 * meshes, materials) or the readout. A target that does nothing in a state is either a bug or a by-design
 * no-op that must be shown disabled (TapTarget.enabled) so the user knows why.
 */
class DeadTapTest {
    private fun hash(n: Node): Long {
        var h = 17L
        fun rec(x: Node) { h = h * 31 + x.tfVersion; h = h * 31 + x.meshVersion; h = h * 31 + x.matVersion; h = h * 31 + x.visVersion; h = h * 31 + (if (x.visible) 1 else 0)
            // values too: two runs that both animate every frame have equal version counters even when they move differently
            h = h * 31 + x.pos.hashCode(); h = h * 31 + x.rot.hashCode(); h = h * 31 + x.scale.hashCode(); x.children.forEach(::rec) }
        rec(n); return h
    }

    @Test
    fun everyEnabledTargetDoesSomething() {
        val cache = File("build/review-cache").apply { mkdirs() }
        val dead = ArrayList<String>()
        var n = 0
        for (d in DemoRegistry.all.filterIsInstance<SpecDemo>()) {
            val states = listOf<Pair<String, () -> Unit>>("default" to {}) + d.reviewVariants().toList()
            for ((st, apply) in states) {
                val targets = run { d.runHeadless(cache, 0.5f); apply(); repeat(20) { d.update(1f / 60f) }; d.tapTargets }
                // control: the same state left alone for 30 frames
                fun settle(tap: String?): Pair<Long, String>? {
                    val root = d.runHeadless(cache, 0.5f); apply(); repeat(20) { d.update(1f / 60f) }
                    if (tap != null) {
                        val t = d.tapTargets.firstOrNull { it.node == tap } ?: return null
                        if (t.enabled?.invoke() == false) return null
                        d.onTap(tap)
                    }
                    repeat(30) { d.update(1f / 60f) }
                    return hash(root) to d.readoutText
                }
                val control = settle(null)
                for (t in targets) {
                    val r = settle(t.node) ?: continue
                    n++
                    if (r == control) dead += "${d.id} [$st] ${t.node} (${t.zh})"
                }
            }
        }
        println("DEADTAP $n presses, ${dead.size} with no visible effect:")
        dead.forEach { println("DEADTAP   $it") }
        assertTrue(dead.joinToString("\n"), dead.isEmpty())
    }
}
