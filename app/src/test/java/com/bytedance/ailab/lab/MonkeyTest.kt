package com.bytedance.ailab.lab

import com.bytedance.ailab.lab.spec.Node
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.random.Random

/**
 * Crash hunt: a seeded "monkey" plays every demo — thousands of random pinches on its current targets
 * (target lists change with modes), random pinch-drags with random hand motion, random frame lengths,
 * and rebuilds — the way an impatient visitor would. Any exception, or a non-finite transform reaching
 * the scene graph (the SDK rejects it), fails the test with the demo, step and action that caused it.
 */
class MonkeyTest {
    @Test
    fun randomPlayNeverCrashes() {
        val cache = File("build/review-cache").apply { mkdirs() }
        val bad = ArrayList<String>()
        var actions = 0
        for (d in DemoRegistry.all.filterIsInstance<SpecDemo>()) {
            val rnd = Random(d.id.hashCode())
            var root: Node = d.runHeadless(cache, 0.5f)
            val log = ArrayDeque<String>()
            fun note(s: String) { log.addLast(s); if (log.size > 8) log.removeFirst() }
            try {
                for (step in 0 until 1500) {
                    val r = rnd.nextFloat()
                    when {
                        r < 0.45f -> {
                            val ts = d.tapTargets
                            if (ts.isNotEmpty()) { val t = ts[rnd.nextInt(ts.size)].node; note("tap $t"); d.onTap(t) }
                        }
                        r < 0.65f -> {
                            val names = ArrayList<String>()
                            fun rec(n: Node) { if (d.isDragTarget(n.name)) names += n.name; n.children.forEach(::rec) }
                            rec(root)
                            if (names.isNotEmpty()) {
                                val nm = names[rnd.nextInt(names.size)]
                                note("drag $nm"); d.onDragStart(nm)
                                repeat(1 + rnd.nextInt(20)) {
                                    d.onDrag(nm, (rnd.nextFloat() - 0.5f) * 0.1f, (rnd.nextFloat() - 0.5f) * 0.1f, (rnd.nextFloat() - 0.5f) * 0.1f)
                                    d.update(1f / 72f)
                                }
                                d.onDragEnd(nm)
                            }
                        }
                        r < 0.995f -> { val k = 1 + rnd.nextInt(30); note("frames $k"); repeat(k) { d.update(1f / 90f + rnd.nextFloat() * 0.04f) } }
                        else -> { note("rebuild"); root = d.runHeadless(cache, 0.2f) }
                    }
                    actions++
                    if (step % 50 == 0) {
                        var badNode: String? = null
                        fun chk(n: Node) {
                            if (badNode != null) return
                            val v = floatArrayOf(n.pos.x, n.pos.y, n.pos.z, n.scale.x, n.scale.y, n.scale.z, n.rot.x, n.rot.y, n.rot.z, n.rot.w)
                            if (n.visible && v.any { !it.isFinite() }) badNode = n.name
                            n.children.forEach(::chk)
                        }
                        chk(root)
                        if (badNode != null) { bad += "${d.id}: non-finite transform on '$badNode' after ${log.joinToString(" / ")}"; break }
                    }
                }
            } catch (e: Throwable) {
                bad += "${d.id}: ${e::class.simpleName}: ${e.message} after ${log.joinToString(" / ")}\n    at " + e.stackTrace.take(4).joinToString("\n    at ")
            }
            println("MONKEY ${d.id}: ${if (bad.none { it.startsWith(d.id + ":") }) "ok" else "FAILED"}")
        }
        // HAMMER: the user's crash pattern — the same button pinched again and again, in every review state
        var hammered = 0
        for (d in DemoRegistry.all.filterIsInstance<SpecDemo>()) {
            val states = listOf<Pair<String, () -> Unit>>("default" to {}) + d.reviewVariants().toList()
            for ((st, apply) in states) {
                d.runHeadless(cache, 0.3f); apply(); repeat(10) { d.update(1f / 60f) }
                for (t in d.tapTargets.map { it.node }) {
                    try {
                        repeat(15) { k ->
                            if (d.tapTargets.none { it.node == t }) return@repeat
                            d.onTap(t); hammered++
                            repeat(1 + (k * 7) % 5) { d.update(1f / 72f) }
                        }
                        repeat(30) { d.update(1f / 72f) }
                    } catch (e: Throwable) {
                        bad += "${d.id} [$st] hammering $t: ${e::class.simpleName}: ${e.message}\n    at " + e.stackTrace.take(4).joinToString("\n    at ")
                    }
                }
            }
        }
        println("MONKEY hammer: $hammered repeated presses")
        println("MONKEY $actions actions, problems: ${if (bad.isEmpty()) "none" else bad.size}")
        assertTrue(bad.joinToString("\n"), bad.isEmpty())
    }
}
