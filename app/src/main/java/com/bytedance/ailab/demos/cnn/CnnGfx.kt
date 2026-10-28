// CnnGfx — geometry and texture helpers for the CNN demo (pure Kotlin: MeshData / Image only).
//
// Tensors are drawn as stacks of textured planes. Each LAYER of a state is ONE texture atlas (a vertical strip
// of n cells, one per channel), baked only when the user changes the input or the network (never per frame)
// and cached per state, so the adapter's identity texture cache stays bounded. Every pixel is s×s texels with a
// 1-texel dark gap, so a 28×28 image reads as 784 numbers instead of a blur.

package com.bytedance.ailab.demos.cnn

import com.bytedance.ailab.core.P3
import com.bytedance.ailab.core.cnn.Glyph
import com.bytedance.ailab.lab.Q
import com.bytedance.ailab.lab.spec.Node
import com.bytedance.ailab.render.Image
import com.bytedance.ailab.render.MeshData
import com.bytedance.ailab.render.MeshGen
import com.bytedance.ailab.render.OptGeo
import com.bytedance.ailab.render.ProcTex
import com.bytedance.ailab.render.Rgb
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

internal object CnnGfx {
    /** Several meshes as one (fewer nodes): each part translated by its offset. */
    fun merge(parts: List<Pair<MeshData, P3>>): MeshData {
        val nv = parts.sumOf { it.first.vertexCount }; val ni = parts.sumOf { it.first.idx.size }
        val pos = FloatArray(nv * 3); val nrm = FloatArray(nv * 3); val uv = FloatArray(nv * 2); val idx = IntArray(ni)
        var v0 = 0; var i0 = 0
        for ((m, o) in parts) {
            for (k in 0 until m.vertexCount) {
                pos[(v0 + k) * 3] = m.pos[k * 3] + o.x; pos[(v0 + k) * 3 + 1] = m.pos[k * 3 + 1] + o.y; pos[(v0 + k) * 3 + 2] = m.pos[k * 3 + 2] + o.z
                for (a in 0 until 3) nrm[(v0 + k) * 3 + a] = m.nrm[k * 3 + a]
                uv[(v0 + k) * 2] = m.uv[k * 2]; uv[(v0 + k) * 2 + 1] = m.uv[k * 2 + 1]
            }
            for (k in m.idx.indices) idx[i0 + k] = m.idx[k] + v0
            v0 += m.vertexCount; i0 += m.idx.size
        }
        return MeshData(pos, nrm, uv, idx)
    }

    /** The 12 edges of an sx×sy×sz box as square bars of thickness [t] (constant on screen, unlike a texture frame). */
    fun wireBox(sx: Float, sy: Float, sz: Float, t: Float): MeshData {
        val hx = sx / 2; val hy = sy / 2; val hz = sz / 2
        val parts = ArrayList<Pair<MeshData, P3>>()
        val ex = MeshGen.box(sx + t, t, t); val ey = MeshGen.box(t, sy + t, t); val ez = MeshGen.box(t, t, sz + t)
        for (a in listOf(-1f, 1f)) for (b in listOf(-1f, 1f)) {
            parts += ex to P3(0f, a * hy, b * hz); parts += ey to P3(a * hx, 0f, b * hz); parts += ez to P3(a * hx, b * hy, 0f)
        }
        return merge(parts)
    }

    /** Quad in the XY plane facing +Z, centred, with the given UV rectangle (v down). */
    fun quadUv(w: Float, h: Float, u0: Float, v0: Float, u1: Float, v1: Float): MeshData =
        OptGeo.quad(1f, 1f).also { setQuad(it, -w / 2, h / 2, w / 2, -h / 2, 0f, u0, v0, u1, v1) }

    /** Rewrite a quad (vertex order of OptGeo.quad: TL, TR, BL, BR) in place: corners (x0,y0) top-left, (x1,y1) bottom-right. */
    fun setQuad(md: MeshData, x0: Float, y0: Float, x1: Float, y1: Float, z: Float, u0: Float, v0: Float, u1: Float, v1: Float) {
        OptGeo.setPos(md, 0, x0, y0, z); OptGeo.setUv(md, 0, u0, v0)
        OptGeo.setPos(md, 1, x1, y0, z); OptGeo.setUv(md, 1, u1, v0)
        OptGeo.setPos(md, 2, x0, y1, z); OptGeo.setUv(md, 2, u0, v1)
        OptGeo.setPos(md, 3, x1, y1, z); OptGeo.setUv(md, 3, u1, v1)
    }

    /** Layout of a vertical-strip atlas: [n] cells of side·s texels plus a 1-texel frame on every side. */
    class Atlas(val n: Int, val side: Int, val s: Int) {
        val cell = side * s + 2
        val w = cell; val h = n * cell
        /** UV of cell [k] content rows [r0, r1) and columns [c0, c1) (pixel units). */
        fun uv(k: Int, r0: Float, c0: Float, r1: Float, c1: Float): FloatArray =
            floatArrayOf((1 + c0 * s) / w, (k * cell + 1 + r0 * s) / h, (1 + c1 * s) / w, (k * cell + 1 + r1 * s) / h)
        /** Whole cell including its frame. */
        fun uvCell(k: Int): FloatArray = floatArrayOf(0.5f / w, (k * cell + 0.5f) / h, 1f - 0.5f / w, ((k + 1) * cell - 0.5f) / h)
        /** World size of the whole cell when the content is [size] metres. */
        fun cellSize(size: Float) = size * cell / (side * s)

        /** Bake: [color] gives the packed ARGB of pixel (k, i, j); [frame] the 1-texel border (shows the plane's edge). */
        fun bake(frame: Int, gaps: Boolean = s >= 4, color: (Int, Int, Int) -> Int): Image {
            // colour once per tensor pixel (the s×s texels of a pixel share it), then fill: ~s² fewer colour calls
            val cols = IntArray(n * side * side)
            for (k in 0 until n) for (i in 0 until side) for (j in 0 until side) cols[(k * side + i) * side + j] = color(k, i, j)
            val img = Image(w, h)
            val px = img.px
            for (k in 0 until n) {
                val y0 = k * cell
                for (y in 0 until cell) {
                    val row = (y0 + y) * w
                    if (y == 0 || y == cell - 1) { java.util.Arrays.fill(px, row, row + w, frame); continue }
                    px[row] = frame; px[row + w - 1] = frame
                    val ty = y - 1
                    if (gaps && ty % s == s - 1) { java.util.Arrays.fill(px, row + 1, row + w - 1, 0); continue }
                    val src = (k * side + ty / s) * side
                    for (tx in 0 until side * s) px[row + 1 + tx] = if (gaps && tx % s == s - 1) 0 else cols[src + tx / s]
                }
            }
            return img
        }
    }

    /** Diverging colour (sRGB, ADD-friendly: black = nothing): warm for +, cool for −, brightness ∝ |v|^0.75. */
    val WARM = Rgb(1.00f, 0.62f, 0.16f)
    val COOL = Rgb(0.20f, 0.60f, 1.00f)
    fun diverging(v: Float): Int {
        val t = abs(v).coerceIn(0f, 1f).pow(0.75f)
        val c = if (v >= 0f) WARM else COOL
        return ProcTex.argb(1f, c.r * t, c.g * t, c.b * t)
    }

    /** Same hues for alpha-blended maps: colour at full strength, opacity ∝ |v|^0.75 (zero = see-through). */
    fun divergingA(v: Float): Int {
        val t = abs(v).coerceIn(0f, 1f).pow(0.75f)
        val c = if (v >= 0f) WARM else COOL
        return ProcTex.argb(t, c.r, c.g, c.b)
    }

    fun grey(v: Float): Int { val c = v.coerceIn(0f, 1f); return ProcTex.argb(1f, c, c, c) }

    /** A white text label with the glyph's alpha (tinted by the material colour). */
    fun glyphImage(g: Glyph): Image {
        val img = Image(g.w, g.h)
        for (y in 0 until g.h) for (x in 0 until g.w) img.px[y * g.w + x] = ProcTex.argb(g.a(x, y), 1f, 1f, 1f)
        return img
    }

    /** A soft square frame (bright border, transparent inside) — the output-pixel marker. */
    fun frameTex(n: Int = 32, border: Int = 5): Image {
        val img = Image(n, n)
        for (y in 0 until n) for (x in 0 until n) {
            val d = minOf(minOf(x, y), minOf(n - 1 - x, n - 1 - y))
            val a = if (d < border) 1f else 0.28f
            img.px[y * n + x] = ProcTex.argb(1f, a, a, a)
        }
        return img
    }

    /** Point a unit cylinder (axis +Y, height 1, centred) from [a] to [b] with radius [r]. */
    fun placeLine(n: Node, a: P3, b: P3, r: Float) {
        val dx = b.x - a.x; val dy = b.y - a.y; val dz = b.z - a.z
        val len = sqrt(dx * dx + dy * dy + dz * dz).coerceAtLeast(1e-4f)
        n.pos = P3((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2)
        n.rot = Q.fromTo(0f, 1f, 0f, dx, dy, dz)
        n.scale = P3(r, len, r)
    }
}
