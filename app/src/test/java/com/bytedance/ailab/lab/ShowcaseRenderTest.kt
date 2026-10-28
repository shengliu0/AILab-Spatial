package com.bytedance.ailab.lab

import com.bytedance.ailab.lab.spec.Cam
import com.bytedance.ailab.core.P3
import com.bytedance.ailab.lab.spec.SoftRender
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * HD showcase stills (README / store poster): every demo at 2560 x 1440, 2x supersampled, the pinch
 * halos hidden. Opt-in, it takes a while:
 *   ./gradlew :app:testDebugUnitTest --tests "*ShowcaseRenderTest*" -Dshowcase=true [-Dreview.only=<id>]
 * -> app/build/showcase/<id>.png
 */
class ShowcaseRenderTest {
    /** Simulated seconds before the still, and an optional review variant, per demo (interesting moments). */
    private val moment = mapOf(
        "embedding" to (2f to "king"),
        "kernel" to (2f to "lifted"),
        "diffusion" to (2f to "generating"),
        "gpt" to (2f to "copy"),
    )

    /** View direction (target -> eye) per demo; default = the review "hero" 3/4 view. */
    private val view = mapOf("solar" to floatArrayOf(0.15f, 0.62f, 1f), "young" to floatArrayOf(0.35f, 0.38f, 1f),
        "cyclotron" to floatArrayOf(0.30f, 0.40f, 1f), "photoelectric" to floatArrayOf(0.40f, 0.38f, 1f))

    /**
     * Camera along [dir] that makes the VISIBLE geometry fill [margin] of a [fovDeg] x [aspect] frame:
     * projects (sub-sampled) world vertices, solves the distance per point, recentres, iterates.
     */
    private fun fit(root: com.bytedance.ailab.lab.spec.Node, dir: FloatArray, fovDeg: Float, aspect: Float, margin: Float = 0.90f): Cam {
        val pts = ArrayList<FloatArray>()
        SoftRender.forEachMesh(root) { n, w ->
            val m = n.mesh!!; val step = maxOf(1, m.vertexCount / 600); var k = 0
            while (k < m.vertexCount) { pts += w.apply(m.pos[k * 3], m.pos[k * 3 + 1], m.pos[k * 3 + 2]); k += step }
        }
        val l = kotlin.math.sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2])
        val back = floatArrayOf(dir[0] / l, dir[1] / l, dir[2] / l)            // target -> eye
        val f = floatArrayOf(-back[0], -back[1], -back[2])
        var r = floatArrayOf(f[1] * 0f - f[2] * 1f, f[2] * 0f - f[0] * 0f, f[0] * 1f - f[1] * 0f)   // f x Y
        val rl = kotlin.math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]); r = floatArrayOf(r[0] / rl, r[1] / rl, r[2] / rl)
        val u = floatArrayOf(r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0])
        val ty = kotlin.math.tan(Math.toRadians(fovDeg / 2.0)).toFloat() * margin; val tx = ty * aspect
        fun dot(a: FloatArray, b: FloatArray) = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        var c = FloatArray(3) { a -> (pts.minOf { it[a] } + pts.maxOf { it[a] }) / 2 }
        var d = 1f
        repeat(4) {
            d = 0.05f
            for (p in pts) {
                val q = floatArrayOf(p[0] - c[0], p[1] - c[1], p[2] - c[2])
                val px = dot(q, r); val py = dot(q, u); val pz = dot(q, f)
                d = maxOf(d, kotlin.math.abs(px) / tx - pz, kotlin.math.abs(py) / ty - pz)
            }
            var x0 = 1e9f; var x1 = -1e9f; var y0 = 1e9f; var y1 = -1e9f
            for (p in pts) {
                val q = floatArrayOf(p[0] - c[0], p[1] - c[1], p[2] - c[2])
                val z = d + dot(q, f); val sx = dot(q, r) / z; val sy = dot(q, u) / z
                x0 = minOf(x0, sx); x1 = maxOf(x1, sx); y0 = minOf(y0, sy); y1 = maxOf(y1, sy)
            }
            val mx = (x0 + x1) / 2 * d; val my = (y0 + y1) / 2 * d
            c = floatArrayOf(c[0] + r[0] * mx + u[0] * my, c[1] + r[1] * mx + u[1] * my, c[2] + r[2] * mx + u[2] * my)
        }
        return Cam(P3(c[0] + back[0] * d, c[1] + back[1] * d, c[2] + back[2] * d), P3(c[0], c[1], c[2]), fovDeg, "showcase")
    }

    @Test
    fun renderShowcase() {
        assumeTrue(System.getProperty("showcase") == "true")
        val only = System.getProperty("review.only").orEmpty()
        val out = File("build/showcase").apply { mkdirs() }
        val cache = File("build/review-cache").apply { mkdirs() }
        for (d in DemoRegistry.all.filterIsInstance<SpecDemo>()) {
            if (only.isNotEmpty() && d.id != only) continue
            val (secs, variant) = moment[d.id] ?: (2f to null)
            val root = d.runHeadless(cache, secs)
            if (variant != null) { d.reviewVariants()[variant]?.invoke(); repeat(90) { d.update(1f / 60f) } }
            root.walk { if (it.name.startsWith("Halo_")) it.visible = false }
            val cam = fit(root, view[d.id] ?: floatArrayOf(0.55f, 0.42f, 1f), 40f, 16f / 9f)
            val t0 = System.nanoTime()
            SoftRender.writePng(SoftRender.render(root, cam, 2560, 1440, 2), File(out, "${d.id}.png"))
            println("SHOWCASE ${d.id}: ${"%.1f".format((System.nanoTime() - t0) / 1e9)} s")
        }
    }
}
