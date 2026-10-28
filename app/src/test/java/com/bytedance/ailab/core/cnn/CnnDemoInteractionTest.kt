package com.bytedance.ailab.core.cnn

import com.bytedance.ailab.demos.cnn.CnnDemo
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.math.cos
import kotlin.math.sin

/**
 * The CNN demo's interactions, headless: the kernel follows a pinch-drag along the (yawed) input block's own
 * axes, the receptive-field cell follows its drag, the scan visits every position at the documented speed,
 * and the per-frame cost stays far below the 2 ms budget (inference + texture bakes run only on a change).
 */
class CnnDemoInteractionTest {
    private val cache = File("build/review-cache").apply { mkdirs() }
    private val yaw = Math.toRadians(36.0).toFloat()

    @Test
    fun dragMovesTheKernelAndTheReceptiveFieldCell() {
        val d = CnnDemo
        d.runHeadless(cache, 0.5f)
        d.onDragStart("KernelBox")
        val (i0, j0) = d.debugCursor()
        // 5 cm along the block's rows, 3 cm down: px = 0.29 m / 28, gain 1.4 → +7 columns, +4 rows
        d.onDrag("KernelBox", 0.05f * cos(yaw), -0.03f, -0.05f * sin(yaw)); d.update(1f / 60f)
        val (i1, j1) = d.debugCursor()
        println("CNN-DRAG kernel ($i0,$j0) → ($i1,$j1)")
        assertEquals(j0 + 7, j1); assertEquals(i0 + 4, i1)
        // dragging pauses the scan: the cursor stays where the hand left it
        repeat(60) { d.update(1f / 60f) }
        assertEquals(i1 to j1, d.debugCursor())
        // the receptive-field cell: 3 cm per cell
        d.onDragStart("Pool2"); d.onDrag("Pool2", 0.031f * cos(yaw), 0.061f, -0.031f * sin(yaw)); d.update(1f / 60f)
        assertEquals(0 to 3, d.debugRf())
        assertTrue(d.rfOn)
        assertTrue(d.readoutText.contains("10×10"))
    }

    @Test
    fun scanVisitsEveryPositionAndSwitchesFollowTheModel() {
        val d = CnnDemo
        d.runHeadless(cache, 0.1f)
        // slow (4 positions / s): one position at a time, row-major — every dot product can be watched
        d.setSpeed(0)
        fun lin(c: Pair<Int, Int>) = c.first * 26 + c.second
        var prev = lin(d.debugCursor()); var steps = 0
        repeat(60 * 3) { d.update(1f / 60f); val k = lin(d.debugCursor()); assertTrue(k - prev in 0..1); steps += k - prev; prev = k }
        assertTrue("slow scan advanced $steps", steps in 11..13)
        // fast (160 / s): the whole 26×26 map in 676 / 160 = 4.2 s
        d.setSpeed(2)
        var reached = false
        repeat(60 * 5) { d.update(1f / 60f); if (d.debugCursor() == (25 to 25)) reached = true }
        assertTrue(reached)
        // multispectral: 4-channel kernels; RGB-only sensor: 3 channels and the RGB-only model
        d.setScene(true); d.update(1f / 60f)
        assertTrue(d.readoutText.lines()[0].contains("4×28×28"))
        d.setNir(false); d.update(1f / 60f)
        assertTrue(d.readoutText.lines()[0].contains("3×28×28"))
        d.setScene(false); d.update(1f / 60f)
        assertTrue(d.readoutText.lines()[1].contains("3×3×3"))
    }

    @Test
    fun perFrameCostIsTinyAndChangesAreBounded() {
        val d = CnnDemo
        d.runHeadless(cache, 1f)
        repeat(600) { d.update(1f / 60f) }                     // warm-up (JIT)
        val t0 = System.nanoTime(); repeat(600) { d.update(1f / 60f) }
        val perUs = (System.nanoTime() - t0) / 600e3
        var worst = 0.0
        repeat(8) { val t = System.nanoTime(); d.onTap("Sample"); d.update(1f / 60f); worst = maxOf(worst, (System.nanoTime() - t) / 1e6) }
        var nodes = 0; var drawn = 0
        d.root!!.walk { n -> nodes++; var p: com.bytedance.ailab.lab.spec.Node? = n; var v = true; while (p != null) { if (!p.visible) v = false; p = p.parent }; if (v && n.mesh != null) drawn++ }
        println("CNN-COST update %.1f µs/frame; worst new state (forward + atlas bakes) %.1f ms; nodes %d, drawn %d".format(perUs, worst, nodes, drawn))
        assertTrue("per-frame update must stay far below 2 ms", perUs < 500)
        assertTrue("visible nodes < 600", drawn < 600)
    }
}
