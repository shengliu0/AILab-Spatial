// ResGfx — textures and small meshes for the ResNet demo (pure Kotlin: MeshData / Image only). All built once.

package com.bytedance.ailab.demos.resnet

import com.bytedance.ailab.core.resnet.Spirals
import com.bytedance.ailab.render.Image
import com.bytedance.ailab.render.MeshData
import com.bytedance.ailab.render.MeshGen
import com.bytedance.ailab.render.OptGeo
import com.bytedance.ailab.render.ProcTex
import com.bytedance.ailab.render.Rgb
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin
import kotlin.math.sqrt

internal object ResGfx {
    val CLS1 = Rgb(1.00f, 0.66f, 0.16f)       // spiral arm 1 = amber
    val CLS0 = Rgb(0.22f, 0.92f, 0.74f)       // spiral arm 0 = teal
    private val MID = Rgb(0.07f, 0.07f, 0.10f)

    /** Map colour ramp (256 x 4): U = P(class 1): teal -> dark -> amber, thin white band at 0.5 (the boundary). */
    val mapRamp: Image by lazy {
        val w = 256; val img = Image(w, 4)
        for (y in 0 until 4) for (x in 0 until w) {
            val p = x / (w - 1f); val d = sqrt((abs(p - 0.5f) * 2f).coerceIn(0f, 1f))
            val c = if (abs(p - 0.5f) < 0.006f) Rgb(0.85f, 0.85f, 0.85f)
                    else ProcTex.okLerp(MID, scale(if (p >= 0.5f) CLS1 else CLS0, 0.55f), d)
            img.px[y * w + x] = ProcTex.argb(1f, c.r, c.g, c.b)
        }
        img
    }

    /** The two spiral arms as bright dots on a transparent tile, same frame as the map ([-R, R]^2, row 0 = top). */
    fun dotsTex(n: Int, range: Float): Image {
        val img = Image(n, n)
        for (cls in 0..1) {
            val a = Spirals.arm(cls, 260); val c = if (cls == 1) CLS1 else CLS0
            for (i in 0 until 260) {
                val cx = (a[i * 2] + range) / (2 * range) * (n - 1); val cy = (range - a[i * 2 + 1]) / (2 * range) * (n - 1)
                for (yy in (cy - 2).toInt()..(cy + 2).toInt()) for (xx in (cx - 2).toInt()..(cx + 2).toInt()) {
                    if (xx !in 0 until n || yy !in 0 until n) continue
                    val r = sqrt((xx - cx) * (xx - cx) + (yy - cy) * (yy - cy))
                    if (r <= 1.5f) img.px[yy * n + xx] = ProcTex.argb(1f, c.r, c.g, c.b)
                }
            }
        }
        return img
    }

    /**
     * Chart plate: dark glass, plot frame, loss grid every 0.1 (0.2 brighter) from 0 to [lossMax], epoch grid every
     * 10, and a dashed gold line at the chance loss ln 2. Plot area = pixels [x0, x1] x [y0, y1] (row 0 = top).
     */
    fun chartTex(w: Int, h: Int, x0: Int, x1: Int, y0: Int, y1: Int, lossMax: Float, epochs: Int): Image {
        val img = Image(w, h)
        val chanceRow = y1 - (Spirals.CHANCE_LOSS / lossMax) * (y1 - y0)
        for (r in 0 until h) for (c in 0 until w) {
            var rr = 0.03f; var gg = 0.035f; var bb = 0.06f
            if (c in x0..x1 && r in y0..y1) {
                val lv = (y1 - r).toFloat() / (y1 - y0) * lossMax
                val k = Math.round(lv * 10f); val near = abs(lv * 10f - k) * (y1 - y0) / (lossMax * 10f) < 0.6f
                if (near) { val m = if (k % 2 == 0) 0.16f else 0.09f; rr = m; gg = m * 1.15f; bb = m * 1.6f }
                val ep = (c - x0).toFloat() / (x1 - x0) * epochs
                if (abs(ep / 10f - Math.round(ep / 10f)) * (x1 - x0) / (epochs / 10f) < 0.6f) { rr = maxOf(rr, 0.09f); gg = maxOf(gg, 0.10f); bb = maxOf(bb, 0.15f) }
                if (abs(r - chanceRow) < 1.1f && (c / 7) % 2 == 0) { rr = 0.80f; gg = 0.64f; bb = 0.25f }
            }
            if ((c in x0 - 2..x0 && r in y0..y1 + 2) || (r in y1..y1 + 2 && c in x0 - 2..x1)) { rr = 0.62f; gg = 0.68f; bb = 0.80f }
            img.px[r * w + c] = ProcTex.argb(1f, rr, gg, bb)
        }
        return img
    }

    /** Soft round glow (linear brightness), for ADD discs. */
    val softDisc: Image by lazy {
        OptGeo.lumTex(48, 48) { u, v ->
            val r = sqrt((u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f)) * 2f
            (exp(-r * r * 4.0f) * (1f - ((r - 0.85f) / 0.15f).coerceIn(0f, 1f))).coerceIn(0f, 1f)
        }
    }

    /** Unit skip arc: a half-ellipse tube in the XY plane from (0, -0.5) bulging to (1, 0) and back to (0, +0.5). */
    val arcMesh: MeshData by lazy {
        val n = 17; val path = FloatArray(n * 3)
        for (i in 0 until n) { val t = PI * i / (n - 1); path[i * 3] = sin(t).toFloat(); path[i * 3 + 1] = (-0.5 * cos(t)).toFloat() }
        MeshGen.tube(path, 0.09f, 6, false)
    }

    fun scale(c: Rgb, k: Float) = Rgb(c.r * k, c.g * k, c.b * k)
}
