package com.bytedance.ailab.core.mlp

import com.bytedance.ailab.demos.mlp.MlpDemo
import com.bytedance.ailab.lab.spec.Node
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Device budget for the demo (AGENTS.md): < ~600 visible nodes, per-frame work <= ~2 ms on the Mac JVM, and the
 * input-change path (rasterise + 2 forward passes + thread selection) cheap enough to run on interaction.
 */
class MlpDemoBudgetTest {
    private fun visibleCount(root: Node): Int {
        var n = 0
        fun rec(x: Node) { if (!x.visible) return; if (x.mesh != null) n++; x.children.forEach { rec(it) } }
        rec(root); return n
    }

    @Test
    fun nodesAndFrameTimeStayInBudget() {
        val cache = File("build/review-cache").apply { mkdirs() }
        val root = MlpDemo.runHeadless(cache, 3f)
        val idle = visibleCount(root)
        // worst case: weights card + full ink while a long scribble is being written
        MlpDemo.pickNeuron(5)
        MlpDemo.onDragStart("Pad_0_0")
        var t = 0f
        repeat(400) { k -> t += 0.05f; MlpDemo.onDrag("Pad_0_0", 0.012f * kotlin.math.cos(t * 3f), 0.012f * kotlin.math.sin(t * 2f), 0f); MlpDemo.update(1f / 60f) }
        val writing = visibleCount(root)
        MlpDemo.onDragEnd("Pad_0_0")
        repeat(120) { MlpDemo.update(1f / 60f) }
        val after = visibleCount(root)
        // per-frame cost while a wave runs (no input change) and the cost of one input change
        MlpDemo.replayWave()
        val t0 = System.nanoTime(); repeat(600) { MlpDemo.update(1f / 60f) }
        val frameMs = (System.nanoTime() - t0) / 1e6 / 600
        var changeMs = 0.0
        repeat(20) { k ->
            val a = System.nanoTime(); MlpDemo.onTap("Sample_${k % 10}"); repeat(60) { MlpDemo.update(1f / 60f) }
            changeMs = maxOf(changeMs, (System.nanoTime() - a) / 1e6 / 60)
        }
        println("BUDGET mlp: visible meshes idle $idle, writing $writing, after $after; update %.3f ms/frame (wave running), %.3f ms/frame avg over a sample load"
            .format(frameMs, changeMs))
        assertTrue("too many visible nodes: $writing", maxOf(idle, writing, after) < 600)
        assertTrue("update too slow: $frameMs ms", frameMs < 2.0)
    }
}
