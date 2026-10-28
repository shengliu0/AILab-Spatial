// BpGfx — geometry and texture helpers for the backpropagation demo (pure Kotlin: MeshData / Image only).
//
// The decision surface is a 29 x 29 vertex sheet split into 4 strips (8 rows each, sharing their border rows, 232
// vertices per strip = ~4.6 ms upload on the headset, one strip per frame by the adapter's round-robin). Colour AND
// shading travel in the UVs, OptGeo's trick: U = output probability p through a colour map, V = a per-vertex
// Lambert term — so the texture is baked once and the sheet looks lit although it is Unlit. A thin white band at
// p = 0.5 in the colour map draws the decision boundary exactly where the GPU interpolates U across 0.5.

package com.bytedance.ailab.demos.backprop

import com.bytedance.ailab.core.P3
import com.bytedance.ailab.core.backprop.Data
import com.bytedance.ailab.lab.Q
import com.bytedance.ailab.render.Image
import com.bytedance.ailab.render.MeshData
import com.bytedance.ailab.render.OptGeo
import com.bytedance.ailab.render.ProcTex
import com.bytedance.ailab.render.Rgb
import kotlin.math.abs
import kotlin.math.sqrt

internal object BpGfx {
    // class colours (sRGB): label 1 = amber, label 0 = teal; the sheet uses dimmer versions of the same hues
    val CLS1 = Rgb(1.00f, 0.66f, 0.16f)
    val CLS0 = Rgb(0.22f, 0.92f, 0.74f)
    private val MID = Rgb(0.10f, 0.10f, 0.15f)

    const val TEX_W = 256
    const val TEX_H = 32

    /** Colour of probability p on the sheet (no shading): teal (p = 0) -> dark (0.5) -> amber (1). */
    fun sheetColour(p: Float): Rgb {
        val d = abs(p - 0.5f) * 2f
        val c = if (p >= 0.5f) CLS1 else CLS0
        return ProcTex.okLerp(MID, scale(c, 0.58f), d.coerceIn(0f, 1f).let { sqrt(it) })
    }

    /** Colour map x shading: texel (u, v) = sheetColour(u) * shade(1 - v); white band at u = 0.5 (the boundary). */
    fun sheetTex(): Image {
        val img = Image(TEX_W, TEX_H)
        for (y in 0 until TEX_H) {
            val shade = 1f - y / (TEX_H - 1f)
            for (x in 0 until TEX_W) {
                val p = x / (TEX_W - 1f)
                val band = abs(p - 0.5f) < 0.0045f
                val c = if (band) Rgb(1f, 1f, 1f) else sheetColour(p)
                val k = if (band) 0.75f + 0.25f * shade else shade
                img.px[y * TEX_W + x] = ProcTex.argb(1f, c.r * k, c.g * k, c.b * k)
            }
        }
        return img
    }

    /** V coordinate for a shade in [0, 1] (row 0 = brightest). */
    fun shadeV(shade: Float): Float = ((1f - shade.coerceIn(0f, 1f)) * (TEX_H - 1) + 0.5f) / TEX_H

    /** A small picture of a dataset for its shelf tile: class-coloured dots on dark glass. */
    fun dataTex(d: Data, n: Int = 72): Image {
        val img = Image(n, n)
        for (k in img.px.indices) img.px[k] = ProcTex.argb(1f, 0.035f, 0.04f, 0.06f)
        for (i in 0 until d.n) {
            val cx = ((d.x1(i) * 0.5 + 0.5) * (n - 1)).toFloat(); val cy = ((0.5 - d.x2(i) * 0.5) * (n - 1)).toFloat()
            val c = if (d.y[i] > 0.5) CLS1 else CLS0
            for (yy in (cy - 3).toInt()..(cy + 3).toInt()) for (xx in (cx - 3).toInt()..(cx + 3).toInt()) {
                if (xx !in 0 until n || yy !in 0 until n) continue
                val r = sqrt((xx - cx) * (xx - cx) + (yy - cy) * (yy - cy))
                if (r <= 2.3f) img.px[yy * n + xx] = ProcTex.argb(1f, c.r, c.g, c.b)
            }
        }
        return img
    }

    /** Soft round glow (linear brightness), for ADD discs. */
    val softDisc: Image by lazy {
        OptGeo.lumTex(48, 48) { u, v ->
            val r = sqrt((u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f)) * 2f
            (kotlin.math.exp(-r * r * 4.5f) * (1f - ((r - 0.85f) / 0.15f).coerceIn(0f, 1f))).coerceIn(0f, 1f)
        }
    }

    /** Grid lines for the input plane: quarters major, eighths minor, axes brighter. */
    fun gridTex(n: Int = 256): Image {
        val img = Image(n, n)
        for (y in 0 until n) for (x in 0 until n) {
            val ax = x == n / 2 || x == n / 2 - 1 || y == n / 2 || y == n / 2 - 1
            val major = x % (n / 4) == 0 || y % (n / 4) == 0 || x == n - 1 || y == n - 1
            val minor = x % (n / 8) == 0 || y % (n / 8) == 0
            val c = when { ax -> 0.70f; major -> 0.42f; minor -> 0.20f; else -> 0f }
            img.px[y * n + x] = ProcTex.argb(1f, c, c, c)
        }
        return img
    }

    fun scale(c: Rgb, k: Float) = Rgb(c.r * k, c.g * k, c.b * k)

    /** Static geometry merged into one MeshData: parts placed by translation, rotation and per-axis scale. */
    class Merge {
        private val pos = ArrayList<Float>(); private val nrm = ArrayList<Float>(); private val uv = ArrayList<Float>(); private val idx = ArrayList<Int>()
        fun add(m: MeshData, t: P3, q: Q, s: P3) {
            val base = pos.size / 3
            for (k in 0 until m.vertexCount) {
                val p = q.rotate(floatArrayOf(m.pos[k * 3] * s.x, m.pos[k * 3 + 1] * s.y, m.pos[k * 3 + 2] * s.z))
                pos += p[0] + t.x; pos += p[1] + t.y; pos += p[2] + t.z
                val n = q.rotate(floatArrayOf(m.nrm[k * 3] / s.x, m.nrm[k * 3 + 1] / s.y, m.nrm[k * 3 + 2] / s.z))
                val l = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).coerceAtLeast(1e-9f)
                nrm += n[0] / l; nrm += n[1] / l; nrm += n[2] / l
                uv += m.uv[k * 2]; uv += m.uv[k * 2 + 1]
            }
            for (i in m.idx) idx += base + i
        }
        fun build() = MeshData(pos.toFloatArray(), nrm.toFloatArray(), uv.toFloatArray(), idx.toIntArray())
    }
}
