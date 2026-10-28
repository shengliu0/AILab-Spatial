package com.bytedance.ailab.core.descent

import com.bytedance.ailab.demos.descent.DescentDemo
import com.bytedance.ailab.lab.spec.Node
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * The demo on top of the core, headless: per-frame cost, node budget, and that what is drawn is where the maths
 * says (the start drag moves the start the way the hand moves; the balls ride on the drawn surface).
 */
class DescentDemoBudgetTest {
    private val cache = File("build/review-cache").apply { mkdirs() }

    private fun visibleMeshNodes(root: Node): Int {
        var n = 0
        fun rec(x: Node) { if (!x.visible) return; if (x.mesh != null) n++; x.children.forEach { rec(it) } }
        rec(root); return n
    }

    @Test
    fun perFrameCostAndNodeBudget() {
        val d = DescentDemo
        val root = d.runHeadless(cache, 0.5f)
        var worstNodes = 0; var worstMs = 0.0; var total = 0.0; var frames = 0
        for (k in Surfaces.all.indices) {
            d.selectSurface(k); d.setSpeed(2)
            repeat(900) {                       // 15 s at the fast speed: trails full, chart rescaled, all balls racing
                val t0 = System.nanoTime(); d.update(1f / 60f); val ms = (System.nanoTime() - t0) / 1e6
                if (it > 30) { worstMs = maxOf(worstMs, ms); total += ms; frames++ }
            }
            worstNodes = maxOf(worstNodes, visibleMeshNodes(root))
        }
        println("BUDGET descent: mean update %.3f ms, worst %.2f ms over $frames frames; visible mesh nodes ≤ $worstNodes".format(total / frames, worstMs))
        assertTrue("visible nodes $worstNodes", worstNodes < 600)
        assertTrue("mean per-frame update %.3f ms".format(total / frames), total / frames < 2.0)
    }

    @Test
    fun dragMovesTheStartWithTheHand() {
        val d = DescentDemo
        d.runHeadless(cache, 0.2f)
        d.selectSurface(0)
        val x0 = d.race.startX; val y0 = d.race.startY
        d.onDragStart("StartPin"); d.onDrag("StartPin", 0.05f, 0f, 0f)        // hand 5 cm to the right
        assertTrue("right → +x", d.race.startX > x0 + 0.2); assertEquals(y0, d.race.startY, 1e-9)
        d.onDrag("StartPin", 0f, 0.05f, 0f)                                    // hand up = up the tilted board = away = +y
        assertTrue("up → +y", d.race.startY > y0 + 0.1)
        d.onDragEnd("StartPin")
        assertTrue(d.playing && d.race.steps == 0)
        d.onTap("StartPin")                                                    // pinch = back to the default start
        assertEquals(Surfaces.BOWL.startX, d.race.startX, 0.0); assertEquals(Surfaces.BOWL.startY, d.race.startY, 0.0)
    }

    @Test
    fun ballTogglesAndShelf() {
        val d = DescentDemo
        d.runHeadless(cache, 0.2f)
        d.onTap("Ball_2"); assertTrue(!d.race.active[2])
        val before = d.race.racers[2].steps
        repeat(60) { d.update(1f / 60f) }
        assertEquals(before, d.race.racers[2].steps)                            // an off ball does not race
        assertTrue(d.race.racers[0].steps > before)                             // ... the others do
        d.onTap("Ball_2"); assertTrue(d.race.active[2] && d.race.steps == 0)     // back on: everyone restarts
        d.onTap("Shelf_2"); d.update(1f / 60f)
        assertTrue(d.race.surface === Surfaces.HIMMELBLAU && d.race.lr == Surfaces.HIMMELBLAU.defaultLr)
        // LR slider snaps to the 1-1.5-2-3-5-7 ladder
        d.setLr(0.0042); assertEquals(0.005, d.race.lr, 0.0)
        d.setLrIndex(99); assertEquals(1.0, d.race.lr, 0.0)
    }
}
