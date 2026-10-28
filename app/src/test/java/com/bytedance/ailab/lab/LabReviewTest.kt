package com.bytedance.ailab.lab

import com.bytedance.ailab.lab.spec.SoftRender
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * THE AUTOMATED VISUAL REVIEW (headless). For every SpecDemo in the registry: build its scene,
 * simulate 2 s at 60 Hz (same settle time the Unity ShotDriver used), and render
 *   build/review/shots/demo_<id>_{hero,side,detail}.png   -> scored by visual_review/score_shots.py
 *   build/review/viewer/demo_<id>_viewer.png              -> what a Shared-Space user sees (2.5 m)
 * Select one demo with -Dreview.only=<id>.
 */
class LabReviewTest {
    @Test
    fun renderAllSpecDemos() {
        val only = System.getProperty("review.only")
        val out = File("build/review")
        val cache = File("build/review-cache").apply { mkdirs() }
        var n = 0
        for (d in DemoRegistry.all.filterIsInstance<SpecDemo>()) {
            if (only != null && only.isNotEmpty() && d.id != only) continue
            val t0 = System.nanoTime()
            val root = d.runHeadless(cache, 2f)
            for (cam in SoftRender.framings(root)) {
                val dir = if (cam.name == "viewer") "viewer" else "shots"
                val img = SoftRender.render(root, cam)
                val f = File(out, "$dir/demo_${d.id}_${cam.name}.png")
                SoftRender.writePng(img, f)
                n++
            }
            for ((name, apply) in d.reviewVariants()) {
                apply(); repeat(60) { d.update(1f / 60f) }
                val cam = SoftRender.framings(root).first { it.name == "viewer" }
                SoftRender.writePng(SoftRender.render(root, cam), File(out, "viewer/demo_${d.id}-${name}_viewer.png"))
                n++
            }
            println("REVIEW ${d.id}: ${"%.0f".format((System.nanoTime() - t0) / 1e6)} ms, labels=${d.labels.size}, readout=\n${d.readoutText}")
        }
        assertTrue("no frames rendered", n > 0)
    }
}
