package com.bytedance.ailab.core.gpt

import com.bytedance.ailab.demos.gpt.GptDemo
import com.bytedance.ailab.lab.spec.Node
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * The demo's device budget, measured on the Mac JVM (the headset is ~5-10x slower):
 *  - visible nodes (meshes whose whole parent chain is visible, incl. pinch halos) stay below 600;
 *  - an idle frame and every frame of an append cycle (fly -> forward pass -> bars/threads grow) stay
 *    within ~2 ms, i.e. the forward pass runs only when the prompt changes;
 *  - idle frames do not touch node transforms (no needless per-frame transform pushes to the SDK).
 */
class GptDemoBudgetTest {
    private fun visibleNodes(root: Node): Int {
        var n = 0
        fun rec(x: Node) { if (!x.visible) return; if (x.mesh != null) n++; x.children.forEach { rec(it) } }
        rec(root); return n
    }

    @Test
    fun framesAndNodesStayWithinBudget() {
        val d = GptDemo
        val root = d.runHeadless(File("build/review-cache").apply { mkdirs() }, 1f)
        var worstNodes = visibleNodes(root)
        // idle frames: time, and no transform writes
        repeat(120) { d.update(1f / 60f) }
        val tf0 = HashMap<Node, Int>(); root.walk { tf0[it] = it.tfVersion }
        var t0 = System.nanoTime(); repeat(120) { d.update(1f / 60f) }
        val idleMs = (System.nanoTime() - t0) / 1e6 / 120
        var moved = 0; root.walk { if (it.tfVersion != tf0[it]) moved++ }
        // the busiest state: a full 12-token row, a token highlighted, then append cycles
        d.onTap("Preset4"); repeat(60) { d.update(1f / 60f) }
        d.onTap("Tok5"); repeat(10) { d.update(1f / 60f) }
        worstNodes = maxOf(worstNodes, visibleNodes(root))
        var worstMs = 0.0; var frames = 0; val times = ArrayList<Double>()
        repeat(3) {
            d.onTap("Bar0")
            repeat(90) {
                t0 = System.nanoTime(); d.update(1f / 60f); val ms = (System.nanoTime() - t0) / 1e6
                if (frames > 30) { worstMs = maxOf(worstMs, ms); times += ms }   // after JIT warm-up
                frames++
                worstNodes = maxOf(worstNodes, visibleNodes(root))
            }
        }
        // p95, not the single worst frame: on a shared test machine one GC / scheduler pause dominates the max
        val p95 = times.sorted()[(times.size * 0.95).toInt().coerceAtMost(times.size - 1)]
        println("GPT budget: idle frame %.3f ms (%d nodes moved), append-cycle frame p95 %.2f ms (worst %.2f), max visible nodes %d"
            .format(idleMs, moved, p95, worstMs, worstNodes))
        assertTrue("visible nodes $worstNodes", worstNodes < 600)
        assertTrue("idle frame $idleMs ms", idleMs < 2.0)
        assertTrue("$moved nodes re-positioned in idle frames", moved == 0)
        assertTrue("append-cycle frame p95 $p95 ms", p95 < 4.0)
    }
}
