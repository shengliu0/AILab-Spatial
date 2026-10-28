// OptGeo — geometry + texture helpers for the light-and-optics demos (prism, double slit).
// Pure Kotlin (MeshData / Image only), shared by demos/prism and demos/young.
//
// THE KEY TRICK — light patterns without per-frame textures. The adapter caches textures by Image
// identity, so a demo cannot repaint a texture every time a slider moves. Instead each light
// pattern is a vertex GRID whose U coordinate carries the INTENSITY, sampled through a static
// ramp texture (texel u -> linear brightness u). Three coplanar ADD layers tinted pure red, green
// and blue then sum to any colour, per vertex, exactly — the fringes, white-light orders and the
// continuous spectrum are all painted this way. Editing U in place (meshEdited) is cheap; the
// texture never changes.

package com.bytedance.ailab.render

import com.bytedance.ailab.render.Image
import com.bytedance.ailab.render.MeshData
import com.bytedance.ailab.render.ProcTex
import kotlin.math.pow
import kotlin.math.sqrt

object OptGeo {
    const val RAMP_W = 256

    /** Intensity 0..1 -> texture U inside the ramp, kept half a texel off both ends (U wraps). */
    fun rampU(t: Float): Float = (0.5f + t.coerceIn(0f, 1f) * (RAMP_W - 1)) / RAMP_W

    /** sRGB encode (linear 0..1 -> 0..1). */
    fun enc(l: Float): Float {
        val v = l.coerceIn(0f, 1f)
        return if (v <= 0.0031308f) 12.92f * v else 1.055f * v.pow(1f / 2.4f) - 0.055f
    }

    /**
     * Ramp texture: texel (u, v) = linear brightness u * [profile](v), white, opaque. V runs top
     * (row 0, v = 0) to bottom (v = 1). Used with a pure R / G / B ADD material.
     */
    fun rampTex(h: Int, profile: (Float) -> Float): Image {
        val img = Image(RAMP_W, h)
        for (y in 0 until h) {
            val p = profile(if (h == 1) 0.5f else y / (h - 1f))
            for (x in 0 until RAMP_W) {
                val c = enc(x / (RAMP_W - 1f) * p)
                img.px[y * RAMP_W + x] = ProcTex.argb(1f, c, c, c)
            }
        }
        return img
    }

    /** A static 2-D image from a linear brightness function f(u, v), white, opaque. */
    fun lumTex(w: Int, h: Int, f: (Float, Float) -> Float): Image {
        val img = Image(w, h)
        for (y in 0 until h) for (x in 0 until w) {
            val c = enc(f((x + 0.5f) / w, (y + 0.5f) / h))
            img.px[y * w + x] = ProcTex.argb(1f, c, c, c)
        }
        return img
    }

    /** rows x cols vertex grid with quads between neighbours; positions / UVs to be filled in. */
    fun grid(rows: Int, cols: Int): MeshData {
        val n = rows * cols
        val nrm = FloatArray(n * 3); for (i in 0 until n) nrm[i * 3 + 2] = 1f
        val idx = IntArray((rows - 1) * (cols - 1) * 6)
        var t = 0
        for (r in 0 until rows - 1) for (c in 0 until cols - 1) {
            val a = r * cols + c; val b = a + 1; val d = a + cols; val e = d + 1
            idx[t++] = a; idx[t++] = d; idx[t++] = b
            idx[t++] = b; idx[t++] = d; idx[t++] = e
        }
        return MeshData(FloatArray(n * 3), nrm, FloatArray(n * 2), idx)
    }

    fun setPos(md: MeshData, i: Int, x: Float, y: Float, z: Float) { md.pos[i * 3] = x; md.pos[i * 3 + 1] = y; md.pos[i * 3 + 2] = z }
    fun setUv(md: MeshData, i: Int, u: Float, v: Float) { md.uv[i * 2] = u; md.uv[i * 2 + 1] = v }

    /** Unit quad in the XY plane facing +Z, centred, uv (0,0) top-left. */
    fun quad(w: Float, h: Float): MeshData {
        val md = grid(2, 2)
        setPos(md, 0, -w / 2, h / 2, 0f); setUv(md, 0, 0f, 0f)
        setPos(md, 1, w / 2, h / 2, 0f); setUv(md, 1, 1f, 0f)
        setPos(md, 2, -w / 2, -h / 2, 0f); setUv(md, 2, 0f, 1f)
        setPos(md, 3, w / 2, -h / 2, 0f); setUv(md, 3, 1f, 1f)
        return md
    }

    /**
     * A solid triangular prism: cross-section [poly] (3 CCW x,y vertices) extruded from z = -[depth]/2
     * to +[depth]/2. Flat-shaded, outward CCW. Faces 0..2 are the sides (edge i -> i+1).
     */
    fun triPrism(poly: DoubleArray, depth: Float): MeshData {
        val pos = ArrayList<Float>(); val nrm = ArrayList<Float>(); val uv = ArrayList<Float>(); val idx = ArrayList<Int>()
        fun v(x: Float, y: Float, z: Float, nx: Float, ny: Float, nz: Float, u: Float, w: Float): Int {
            pos += x; pos += y; pos += z; nrm += nx; nrm += ny; nrm += nz; uv += u; uv += w; return pos.size / 3 - 1
        }
        val hz = depth / 2
        val px = FloatArray(3) { poly[it * 2].toFloat() }; val py = FloatArray(3) { poly[it * 2 + 1].toFloat() }
        // caps
        val f0 = v(px[0], py[0], hz, 0f, 0f, 1f, 0f, 0f); v(px[1], py[1], hz, 0f, 0f, 1f, 0f, 1f); v(px[2], py[2], hz, 0f, 0f, 1f, 1f, 1f)
        idx += f0; idx += f0 + 1; idx += f0 + 2
        val b0 = v(px[0], py[0], -hz, 0f, 0f, -1f, 0f, 0f); v(px[1], py[1], -hz, 0f, 0f, -1f, 0f, 1f); v(px[2], py[2], -hz, 0f, 0f, -1f, 1f, 1f)
        idx += b0; idx += b0 + 2; idx += b0 + 1
        // sides
        for (i in 0 until 3) {
            val j = (i + 1) % 3
            val ex = px[j] - px[i]; val ey = py[j] - py[i]; val l = sqrt(ex * ex + ey * ey)
            val nx = ey / l; val ny = -ex / l
            val a = v(px[i], py[i], -hz, nx, ny, 0f, 0f, 1f); v(px[j], py[j], -hz, nx, ny, 0f, 1f, 1f)
            v(px[j], py[j], hz, nx, ny, 0f, 1f, 0f); v(px[i], py[i], hz, nx, ny, 0f, 0f, 0f)
            idx += a; idx += a + 1; idx += a + 2; idx += a; idx += a + 2; idx += a + 3
        }
        return MeshData(pos.toFloatArray(), nrm.toFloatArray(), uv.toFloatArray(), idx.toIntArray())
    }

    /** Rewrite a [triPrism] mesh in place for a new cross-section (same topology). */
    fun triPrismInto(md: MeshData, poly: DoubleArray, depth: Float) {
        val t = triPrism(poly, depth)
        System.arraycopy(t.pos, 0, md.pos, 0, t.pos.size); System.arraycopy(t.nrm, 0, md.nrm, 0, t.nrm.size)
    }

    /** Thin flat segment (a ribbon of width [w] in the plane with normal ~[nz] = z) from (x0,y0) to (x1,y1). */
    fun ribbon(x0: Float, y0: Float, x1: Float, y1: Float, w: Float, z: Float = 0f): MeshData {
        val md = grid(2, 2); ribbonInto(md, x0, y0, x1, y1, w, z); return md
    }

    fun ribbonInto(md: MeshData, x0: Float, y0: Float, x1: Float, y1: Float, w: Float, z: Float = 0f) {
        val dx = x1 - x0; val dy = y1 - y0; val l = sqrt(dx * dx + dy * dy).coerceAtLeast(1e-6f)
        val ox = -dy / l * w / 2; val oy = dx / l * w / 2
        setPos(md, 0, x0 + ox, y0 + oy, z); setUv(md, 0, 0f, 0f)
        setPos(md, 1, x1 + ox, y1 + oy, z); setUv(md, 1, 1f, 0f)
        setPos(md, 2, x0 - ox, y0 - oy, z); setUv(md, 2, 0f, 1f)
        setPos(md, 3, x1 - ox, y1 - oy, z); setUv(md, 3, 1f, 1f)
    }

    /** Gaussian-across, flat-along beam texture (soft glow edges): v = 0.5 is the beam centre. */
    fun beamTex(sharp: Float = 3.2f): Image = lumTex(8, 64) { _, v ->
        val s = (v - 0.5f) * 2f * sharp
        kotlin.math.exp(-s * s / 2)
    }
}
