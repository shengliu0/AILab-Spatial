// DescentGfx — geometry and texture helpers for the gradient-descent demo (pure Kotlin: MeshData / Image only).
//
// The landscape is a height field in a "board" frame (local X = landscape x, local Z = −landscape y, local Y =
// height), drawn as ONE grid mesh per landscape plus ONE baked texture (colour by height × baked lighting +
// luminous iso-lines). Both are built once per landscape (never per frame); the balls read heights from the same
// piecewise-linear triangles the mesh is made of, so they sit exactly on what is drawn.

package com.bytedance.ailab.demos.descent

import com.bytedance.ailab.core.P3
import com.bytedance.ailab.core.descent.LossSurface
import com.bytedance.ailab.lab.Q
import com.bytedance.ailab.lab.spec.Node
import com.bytedance.ailab.render.Image
import com.bytedance.ailab.render.MeshData
import com.bytedance.ailab.render.OptGeo
import com.bytedance.ailab.render.ProcTex
import com.bytedance.ailab.render.Rgb
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.round
import kotlin.math.sin
import kotlin.math.sqrt

internal object DescentGfx {

    // ================================================================ colour by height
    /** Low → high: deep navy, blue, violet, magenta, coral, peach (OKLab-interpolated; the balls stay brighter). */
    private val STOPS = listOf(
        0.00f to Rgb(0.04f, 0.07f, 0.20f), 0.18f to Rgb(0.08f, 0.20f, 0.52f), 0.38f to Rgb(0.30f, 0.22f, 0.66f),
        0.60f to Rgb(0.66f, 0.24f, 0.62f), 0.80f to Rgb(0.92f, 0.42f, 0.42f), 1.00f to Rgb(1.00f, 0.78f, 0.56f))
    private val LUT: Array<Rgb> = Array(256) { k ->
        val t = k / 255f
        var i = 0; while (i < STOPS.size - 2 && t > STOPS[i + 1].first) i++
        val (t0, c0) = STOPS[i]; val (t1, c1) = STOPS[i + 1]
        ProcTex.okLerp(c0, c1, ((t - t0) / (t1 - t0)).coerceIn(0f, 1f))
    }
    fun cmap(d: Float): Rgb = LUT[(d.coerceIn(0f, 1f) * 255f).toInt()]

    // ================================================================ the board frame
    /** Size of the square footprint (m), clearance under the lowest point, and the height of the display range. */
    class Board(val size: Float, val gap: Float, val height: Float) {
        /** Landscape → board-local metres. */
        fun lx(s: LossSurface, x: Double) = ((x - (s.x0 + s.x1) / 2) * size / (s.x1 - s.x0)).toFloat()
        fun lz(s: LossSurface, y: Double) = (-(y - (s.y0 + s.y1) / 2) * size / (s.y1 - s.y0)).toFloat()
        fun ly(s: LossSurface, f: Double) = gap + height * s.display(f).toFloat()
    }

    /** One landscape, ready to draw: grid heights (for the mesh AND the balls), the mesh, its texture, the floor map. */
    class Terrain(val s: LossSurface, val b: Board, val n: Int) {
        /** Board-local heights of the grid vertices, [j][i] flattened (i along x, j along y). */
        val h = FloatArray(n * n)
        init {
            for (j in 0 until n) for (i in 0 until n) {
                val x = s.x0 + (s.x1 - s.x0) * i / (n - 1); val y = s.y0 + (s.y1 - s.y0) * j / (n - 1)
                h[j * n + i] = b.ly(s, s.f(x, y))
            }
        }
        lateinit var mesh: MeshData
        /** The four glowing side curtains (surface edge → floor): the landscape's profile seen from the side. */
        lateinit var curtain: MeshData
        lateinit var tex: Image
        lateinit var floor: Image

        /** Height of the DRAWN mesh at (x, y): the same two triangles per cell as [gridMesh]. */
        fun heightAt(x: Double, y: Double): Float {
            val fx = ((x - s.x0) / (s.x1 - s.x0) * (n - 1)).coerceIn(0.0, n - 1.0)
            val fy = ((y - s.y0) / (s.y1 - s.y0) * (n - 1)).coerceIn(0.0, n - 1.0)
            val i = minOf(floor(fx).toInt(), n - 2); val j = minOf(floor(fy).toInt(), n - 2)
            val a = (fx - i).toFloat(); val c = (fy - j).toFloat()
            val ha = h[j * n + i]; val hb = h[j * n + i + 1]; val hc = h[(j + 1) * n + i]; val hd = h[(j + 1) * n + i + 1]
            return if (a + c <= 1f) ha + a * (hb - ha) + c * (hc - ha) else hd + (1 - a) * (hc - hd) + (1 - c) * (hb - hd)
        }

        /** Board-local point on the drawn surface above landscape point (x, y), lifted by [lift] along local +Y. */
        fun local(x: Double, y: Double, lift: Float = 0f) = P3(b.lx(s, x), heightAt(x, y) + lift, b.lz(s, y))
    }

    /** Grid mesh of a terrain (normals +Y; triangles CCW seen from above), UVs onto a [tw]-texel texture. */
    fun gridMesh(t: Terrain, size: Float = t.b.size, hScale: Float = 1f, tw: Int = 256): MeshData {
        val n = t.n
        val pos = FloatArray(n * n * 3); val nrm = FloatArray(n * n * 3); val uv = FloatArray(n * n * 2)
        for (j in 0 until n) for (i in 0 until n) {
            val k = j * n + i; val fx = i / (n - 1f); val fy = j / (n - 1f)
            pos[k * 3] = (fx - 0.5f) * size; pos[k * 3 + 1] = t.h[k] * hScale; pos[k * 3 + 2] = (0.5f - fy) * size
            nrm[k * 3 + 1] = 1f
            uv[k * 2] = (0.5f + fx * (tw - 1)) / tw            // texel centres: U wraps in the samplers
            uv[k * 2 + 1] = 1f - fy                              // row 0 = far edge = high y
        }
        val idx = IntArray((n - 1) * (n - 1) * 6); var q = 0
        for (j in 0 until n - 1) for (i in 0 until n - 1) {
            val a = j * n + i; val b = a + 1; val c = a + n; val d = c + 1
            idx[q++] = a; idx[q++] = b; idx[q++] = c       // (a→b)×(a→c) points +Y (c is further away, −Z)
            idx[q++] = b; idx[q++] = d; idx[q++] = c
        }
        return MeshData(pos, nrm, uv, idx)
    }

    // ================================================================ bakes (once per landscape)
    /**
     * Surface texture [tw]×[tw]: colour by display height × Lambert lighting from [light] (board-local, baked: the
     * surface is Unlit, so it looks the same on the headset as here), plus bright iso-lines every 1/16 of the display
     * range and a faint coordinate grid. Row 0 = far edge (high y).
     */
    fun bakeSurface(t: Terrain, tw: Int, light: FloatArray, gridStep: Double): Image {
        val s = t.s; val b = t.b
        val d = FloatArray(tw * tw); val y3 = FloatArray(tw * tw)
        for (r in 0 until tw) for (c in 0 until tw) {
            val x = s.x0 + (s.x1 - s.x0) * c / (tw - 1); val y = s.y1 - (s.y1 - s.y0) * r / (tw - 1)
            val dv = s.display(s.f(x, y)).toFloat()
            d[r * tw + c] = dv; y3[r * tw + c] = b.gap + b.height * dv
        }
        val px = b.size / (tw - 1)
        val img = Image(tw, tw)
        val gx = (gridStep / (s.x1 - s.x0) * (tw - 1)).toFloat()      // grid spacing in texels
        for (r in 0 until tw) for (c in 0 until tw) {
            val k = r * tw + c
            val cl = maxOf(c - 1, 0); val cr = minOf(c + 1, tw - 1); val ru = maxOf(r - 1, 0); val rd = minOf(r + 1, tw - 1)
            val dYdX = (y3[r * tw + cr] - y3[r * tw + cl]) / ((cr - cl) * px)
            val dYdZ = (y3[rd * tw + c] - y3[ru * tw + c]) / ((rd - ru) * px)     // rows run toward +Z
            val nl = 1f / sqrt(dYdX * dYdX + 1f + dYdZ * dYdZ)
            val ndl = max(0f, (-dYdX * light[0] + light[1] - dYdZ * light[2]) * nl)
            val shade = 0.30f + 0.85f * ndl
            val col = cmap(d[k])
            // iso-lines of the display height, constant width on screen: distance to the nearest level / |∇level| per texel
            val lv = d[k] * 16f
            val gl = maxOf(abs(d[r * tw + cr] - d[r * tw + cl]) / (cr - cl), abs(d[rd * tw + c] - d[ru * tw + c]) / (rd - ru)) * 16f
            val dist = abs(lv - round(lv)) / max(gl, 1e-4f)
            val line = if (lv < 0.5f) 0f else (1f - ((dist - 0.55f) / 0.9f).coerceIn(0f, 1f)) * 0.55f
            // faint landscape-unit grid
            val gxx = abs(((c / gx + 0.5f) % 1f) - 0.5f) * gx; val gyy = abs(((r / gx + 0.5f) % 1f) - 0.5f) * gx
            val grid = if (minOf(gxx, gyy) < 0.6f) 0.16f else 0f
            var rr = col.r * shade; var gg = col.g * shade; var bb = col.b * shade
            rr = rr * (1 - grid) + 0.85f * line; gg = gg * (1 - grid) + 0.92f * line; bb = bb * (1 - grid) + 1.0f * line
            img.px[k] = ProcTex.argb(1f, rr.coerceIn(0f, 1f), gg.coerceIn(0f, 1f), bb.coerceIn(0f, 1f))
        }
        return img
    }

    /** Floor map: the same iso-lines, glowing in the height colour on a dark plate (the classic contour plot). */
    fun bakeFloor(t: Terrain, tw: Int): Image {
        val s = t.s
        val d = FloatArray(tw * tw)
        for (r in 0 until tw) for (c in 0 until tw) {
            val x = s.x0 + (s.x1 - s.x0) * c / (tw - 1); val y = s.y1 - (s.y1 - s.y0) * r / (tw - 1)
            d[r * tw + c] = s.display(s.f(x, y)).toFloat()
        }
        val img = Image(tw, tw)
        for (r in 0 until tw) for (c in 0 until tw) {
            val k = r * tw + c
            val cl = maxOf(c - 1, 0); val cr = minOf(c + 1, tw - 1); val ru = maxOf(r - 1, 0); val rd = minOf(r + 1, tw - 1)
            val lv = d[k] * 16f
            val gl = maxOf(abs(d[r * tw + cr] - d[r * tw + cl]) / (cr - cl), abs(d[rd * tw + c] - d[ru * tw + c]) / (rd - ru)) * 16f
            val dist = abs(lv - round(lv)) / max(gl, 1e-4f)
            val line = if (lv < 0.5f) 0f else 1f - ((dist - 0.6f) / 1.0f).coerceIn(0f, 1f)
            val edge = minOf(minOf(r, c), minOf(tw - 1 - r, tw - 1 - c)) < 2
            val col = cmap(d[k])
            val base = 0.035f + 0.05f * d[k]
            val rr = if (edge) 0.35f else base * 0.8f + col.r * line * 0.9f
            val gg = if (edge) 0.40f else base * 0.85f + col.g * line * 0.9f
            val bb = if (edge) 0.50f else base * 1.2f + col.b * line * 0.9f
            img.px[k] = ProcTex.argb(1f, rr.coerceIn(0f, 1f), gg.coerceIn(0f, 1f), bb.coerceIn(0f, 1f))
        }
        return img
    }

    /** Four side strips from the surface's edge (v = 0, bright) down to the floor (v = 1, dark): 4·2·n vertices. */
    fun curtainMesh(t: Terrain): MeshData {
        val n = t.n; val S = t.b.size
        val nv = 4 * 2 * n
        val pos = FloatArray(nv * 3); val nrm = FloatArray(nv * 3); val uv = FloatArray(nv * 2)
        val idx = IntArray(4 * (n - 1) * 6); var q = 0
        for (side in 0 until 4) for (i in 0 until n) {
            // walk the border: side 0 near edge (j = 0), 1 right (i = n−1), 2 far (j = n−1), 3 left (i = 0)
            val (gi, gj) = when (side) { 0 -> i to 0; 1 -> (n - 1) to i; 2 -> i to (n - 1); else -> 0 to i }
            val x = (gi / (n - 1f) - 0.5f) * S; val z = (0.5f - gj / (n - 1f)) * S
            val top = (side * n + i) * 2; val bot = top + 1
            pos[top * 3] = x; pos[top * 3 + 1] = t.h[gj * n + gi]; pos[top * 3 + 2] = z
            pos[bot * 3] = x; pos[bot * 3 + 1] = 0.002f; pos[bot * 3 + 2] = z
            uv[top * 2] = 0.5f; uv[top * 2 + 1] = 0.02f; uv[bot * 2] = 0.5f; uv[bot * 2 + 1] = 0.98f
            nrm[top * 3 + 1] = 1f; nrm[bot * 3 + 1] = 1f
            if (i < n - 1) { val a = top; val b = bot; val c = top + 2; val d = bot + 2
                idx[q++] = a; idx[q++] = b; idx[q++] = c; idx[q++] = c; idx[q++] = b; idx[q++] = d }
        }
        return MeshData(pos, nrm, uv, idx)
    }

    /** Vertical glow for the curtains: a bright rim at the top (the profile line), fading to nothing at the floor. */
    fun curtainTex(): Image = OptGeo.lumTex(4, 128) { _, v -> if (v < 0.035f) 1f else 0.30f * exp(-(v - 0.035f) * 3.2f) }

    /** Soft round glow for ADD sprites (white, brightness falls off like a Gaussian). */
    fun glowTex(n: Int = 64): Image = OptGeo.lumTex(n, n) { u, v ->
        val dx = (u - 0.5f) * 2; val dy = (v - 0.5f) * 2; val r2 = dx * dx + dy * dy
        if (r2 >= 1f) 0f else exp(-r2 * 4.5f) * (1 - r2)
    }

    /** Loss-chart backdrop: dark plate, faint decade lines, a gold line at the 1e-3 target. Row 0 = top. */
    fun chartTex(w: Int, h: Int, decades: Int, targetRow: Float, x0: Int, x1: Int, y0: Int, y1: Int): Image {
        val img = Image(w, h)
        for (r in 0 until h) for (c in 0 until w) {
            var rr = 0.035f; var gg = 0.045f; var bb = 0.07f
            val inside = c in x0..x1 && r in y0..y1
            if (inside) {
                val fr = (r - y0).toFloat() / (y1 - y0) * decades
                if (abs(fr - round(fr)) * (y1 - y0) / decades < 0.7f) { rr = 0.12f; gg = 0.15f; bb = 0.22f }
                if (abs(r - targetRow) < 1.2f && (c / 6) % 2 == 0) { rr = 0.95f; gg = 0.78f; bb = 0.35f }
            }
            if ((c in x0 - 2..x0 && r in y0..y1 + 2) || (r in y1..y1 + 2 && c in x0 - 2..x1)) { rr = 0.62f; gg = 0.68f; bb = 0.78f }
            img.px[r * w + c] = ProcTex.argb(1f, rr, gg, bb)
        }
        return img
    }

    // ================================================================ small geometry
    /** Cone along +Y: base circle radius [r] at y = 0, apex at y = [h]. */
    fun cone(r: Float, h: Float, seg: Int = 14): MeshData {
        val pos = ArrayList<Float>(); val nrm = ArrayList<Float>(); val uv = ArrayList<Float>(); val idx = ArrayList<Int>()
        val sl = sqrt(r * r + h * h)
        for (s in 0 until seg) {
            val a0 = 2.0 * Math.PI * s / seg; val a1 = 2.0 * Math.PI * (s + 1) / seg; val am = (a0 + a1) / 2
            val b = pos.size / 3
            fun v(x: Float, y: Float, z: Float, nx: Float, ny: Float, nz: Float) { pos += x; pos += y; pos += z; nrm += nx; nrm += ny; nrm += nz; uv += 0.5f; uv += 0.5f }
            val nx = (cos(am) * h / sl).toFloat(); val nz = (sin(am) * h / sl).toFloat(); val ny = r / sl
            v((cos(a0) * r).toFloat(), 0f, (sin(a0) * r).toFloat(), nx, ny, nz)
            v(0f, h, 0f, nx, ny, nz)
            v((cos(a1) * r).toFloat(), 0f, (sin(a1) * r).toFloat(), nx, ny, nz)
            idx += b; idx += b + 1; idx += b + 2
            // base
            val c = pos.size / 3
            v(0f, 0f, 0f, 0f, -1f, 0f); v((cos(a0) * r).toFloat(), 0f, (sin(a0) * r).toFloat(), 0f, -1f, 0f); v((cos(a1) * r).toFloat(), 0f, (sin(a1) * r).toFloat(), 0f, -1f, 0f)
            idx += c; idx += c + 1; idx += c + 2
        }
        return MeshData(pos.toFloatArray(), nrm.toFloatArray(), uv.toFloatArray(), idx.toIntArray())
    }

    /** Point a unit cylinder (axis +Y, height 1, centred) from [a] to [b] with radius [r]. */
    fun placeLine(n: Node, a: P3, b: P3, r: Float) {
        val dx = b.x - a.x; val dy = b.y - a.y; val dz = b.z - a.z
        val len = sqrt(dx * dx + dy * dy + dz * dz).coerceAtLeast(1e-4f)
        n.pos = P3((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2)
        n.rot = Q.fromTo(0f, 1f, 0f, dx, dy, dz)
        n.scale = P3(r, len, r)
    }

    /**
     * Rewrite a 2×K grid (OptGeo.grid(2, K)) as a flat ribbon of width [w] through the first [count] points of
     * [xy] (x, y pairs, chart-local), at depth [z]; unused vertices collapse onto the last point. In place.
     */
    fun polylineInto(md: MeshData, xy: FloatArray, count: Int, w: Float, z: Float) {
        val k = md.vertexCount / 2
        val m = count.coerceIn(1, k)
        for (i in 0 until k) {
            val p = minOf(i, m - 1)
            val x = xy[p * 2]; val y = xy[p * 2 + 1]
            // direction: average of the neighbouring segments
            val pa = maxOf(p - 1, 0); val pb = minOf(p + 1, m - 1)
            var dx = xy[pb * 2] - xy[pa * 2]; var dy = xy[pb * 2 + 1] - xy[pa * 2 + 1]
            val l = sqrt(dx * dx + dy * dy)
            if (l < 1e-6f) { dx = 1f; dy = 0f } else { dx /= l; dy /= l }
            val ox = -dy * w / 2; val oy = dx * w / 2
            OptGeo.setPos(md, i, x + ox, y + oy, z); OptGeo.setUv(md, i, 0.5f, 0f)
            OptGeo.setPos(md, k + i, x - ox, y - oy, z); OptGeo.setUv(md, k + i, 0.5f, 1f)
        }
    }
}
