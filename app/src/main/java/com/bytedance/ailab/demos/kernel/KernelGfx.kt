// KernelGfx — meshes and textures for the kernel-trick demo (pure Kotlin: MeshData / Image only).
// All bakes are small (≤ 256², built once per demo build); per-frame motion is by node transforms only.

package com.bytedance.ailab.demos.kernel

import com.bytedance.ailab.core.P3
import com.bytedance.ailab.core.kernel.Points
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
import kotlin.math.hypot
import kotlin.math.round
import kotlin.math.sin
import kotlin.math.sqrt

internal object KernelGfx {
    /** Height colour: deep blue → violet → magenta → peach (OKLab), like the loss landscapes. */
    private val STOPS = listOf(0.00f to Rgb(0.10f, 0.22f, 0.62f), 0.35f to Rgb(0.36f, 0.26f, 0.80f),
        0.65f to Rgb(0.78f, 0.30f, 0.70f), 1.00f to Rgb(1.00f, 0.72f, 0.50f))
    fun cmap(t0: Float): Rgb {
        val t = t0.coerceIn(0f, 1f)
        var i = 0; while (i < STOPS.size - 2 && t > STOPS[i + 1].first) i++
        val (a, ca) = STOPS[i]; val (b, cb) = STOPS[i + 1]
        return ProcTex.okLerp(ca, cb, ((t - a) / (b - a)).coerceIn(0f, 1f))
    }

    /**
     * Surface texture for ADD drawing: u = height fraction (0 floor … 1 top), v = angle fraction.
     * Height colour at low intensity, bright iso-height rings every 0.1 and faint meridians every 22.5°.
     */
    fun surfTex(w: Int = 128, h: Int = 64): Image {
        val img = Image(w, h)
        for (r in 0 until h) for (c in 0 until w) {
            val u = (c + 0.5f) / w; val v = (r + 0.5f) / h
            val lv = u * 10f; val dl = abs(lv - round(lv)) * w / 10f            // texels to the nearest level
            val iso = (1f - ((dl - 0.6f) / 1.1f).coerceIn(0f, 1f))
            val mv = v * 16f; val dm = abs(mv - round(mv)) * h / 16f
            val mer = (1f - ((dm - 0.4f) / 0.8f).coerceIn(0f, 1f)) * 0.45f
            val k = (0.30f + 0.85f * maxOf(iso, mer)).coerceAtMost(1f)
            val col = cmap(u)
            img.px[r * w + c] = ProcTex.argb(1f, col.r * k, col.g * k, col.b * k)
        }
        return img
    }

    /** Floor plate: dark navy, grid every 0.25 units, brighter axes, the unit circle. Row 0 = far edge (+y). */
    fun floorTex(n: Int, L: Double): Image {
        val img = Image(n, n)
        val px = 2 * L / (n - 1)
        for (r in 0 until n) for (c in 0 until n) {
            val x = -L + c * px; val y = L - r * px
            var rr = 0.030f; var gg = 0.040f; var bb = 0.075f
            val gx = abs(x / 0.25 - round(x / 0.25)) * 0.25 / px; val gy = abs(y / 0.25 - round(y / 0.25)) * 0.25 / px
            if (minOf(gx, gy) < 0.7) { rr = 0.075f; gg = 0.095f; bb = 0.16f }
            if (abs(x) / px < 1.0 || abs(y) / px < 1.0) { rr = 0.16f; gg = 0.20f; bb = 0.32f }
            val dc = abs(hypot(x, y) - 1.0) / px
            if (dc < 1.0) { rr = 0.20f; gg = 0.26f; bb = 0.42f }
            val edge = minOf(minOf(r, c), minOf(n - 1 - r, n - 1 - c)) < 2
            if (edge) { rr = 0.35f; gg = 0.42f; bb = 0.55f }
            img.px[r * n + c] = ProcTex.argb(1f, rr, gg, bb)
        }
        return img
    }

    /** Catalogue tile of a data set: the 40 points as coloured dots on a dark plate. */
    fun dataTex(p: Points, n: Int, a: Rgb, b: Rgb): Image {
        val img = Image(n, n)
        for (k in img.px.indices) {
            val r = k / n; val c = k % n
            val edge = minOf(minOf(r, c), minOf(n - 1 - r, n - 1 - c)) < 2
            img.px[k] = if (edge) ProcTex.argb(1f, 0.35f, 0.42f, 0.55f) else ProcTex.argb(1f, 0.03f, 0.04f, 0.07f)
        }
        val rad = n * 0.045f
        for (i in 0 until p.n) {
            val cx = ((p.x[i] + 1.05) / 2.1 * n).toFloat(); val cy = ((1.05 - p.y[i]) / 2.1 * n).toFloat()
            val col = if (p.label[i] > 0) a else b
            for (r in (cy - rad - 1).toInt()..(cy + rad + 1).toInt()) for (c in (cx - rad - 1).toInt()..(cx + rad + 1).toInt()) {
                if (r !in 2 until n - 2 || c !in 2 until n - 2) continue
                val d = hypot(c + 0.5f - cx, r + 0.5f - cy)
                if (d < rad) img.px[r * n + c] = ProcTex.argb(1f, col.r, col.g, col.b)
            }
        }
        return img
    }

    fun glowTex(n: Int = 64): Image = OptGeo.lumTex(n, n) { u, v ->
        val dx = (u - 0.5f) * 2; val dy = (v - 0.5f) * 2; val r2 = dx * dx + dy * dy
        if (r2 >= 1f) 0f else exp(-r2 * 4.5f) * (1 - r2)
    }

    /** Polar grid over the unit disc: [rings]+1 radii × [seg]+1 angles. Positions filled by [fillSurface]. */
    fun polarMesh(rings: Int, seg: Int): MeshData {
        val nv = (rings + 1) * (seg + 1)
        val idx = IntArray(rings * seg * 6); var t = 0
        for (i in 0 until rings) for (j in 0 until seg) {
            val a = i * (seg + 1) + j; val b = a + 1; val c = a + seg + 1; val d = c + 1
            idx[t++] = a; idx[t++] = c; idx[t++] = b; idx[t++] = b; idx[t++] = c; idx[t++] = d
        }
        val nrm = FloatArray(nv * 3); for (k in 0 until nv) nrm[k * 3 + 1] = 1f
        return MeshData(FloatArray(nv * 3), nrm, FloatArray(nv * 2), idx)
    }

    /**
     * Fill a [polarMesh] with the surface height(x, y) (units above the floor, 0..1) over the disc of radius [R]
     * (units), drawn at [k] metres per unit: local X = x·k, Y = height·k·[hk], Z = −y·k.
     */
    fun fillSurface(md: MeshData, rings: Int, seg: Int, R: Double, k: Float, hk: Float = 1f, height: (Double, Double) -> Double) {
        for (i in 0..rings) for (j in 0..seg) {
            val v = i * (seg + 1) + j
            val r = R * i / rings; val th = 2 * Math.PI * j / seg
            val x = r * cos(th); val y = r * sin(th)
            val h = height(x, y).coerceIn(0.0, 1.0)
            md.pos[v * 3] = (x * k).toFloat(); md.pos[v * 3 + 1] = (h * k * hk).toFloat(); md.pos[v * 3 + 2] = (-y * k).toFloat()
            md.uv[v * 2] = (0.5f + h.toFloat() * 126f) / 128f; md.uv[v * 2 + 1] = j.toFloat() / seg
        }
    }

    /** Ring in the XY plane (faces +Z: for billboards). */
    fun ringXY(rIn: Float, rOut: Float, seg: Int = 32): MeshData {
        val pos = FloatArray((seg + 1) * 2 * 3); val nrm = FloatArray(pos.size); val uv = FloatArray((seg + 1) * 2 * 2)
        for (s in 0..seg) for (j in 0..1) {
            val a = 2.0 * Math.PI * s / seg; val r = if (j == 0) rIn else rOut; val v = s * 2 + j
            pos[v * 3] = (r * cos(a)).toFloat(); pos[v * 3 + 1] = (r * sin(a)).toFloat(); nrm[v * 3 + 2] = 1f
            uv[v * 2] = j.toFloat(); uv[v * 2 + 1] = s.toFloat() / seg
        }
        val idx = IntArray(seg * 6); var t = 0
        for (s in 0 until seg) { val a0 = s * 2; idx[t++] = a0; idx[t++] = a0 + 1; idx[t++] = a0 + 2; idx[t++] = a0 + 1; idx[t++] = a0 + 3; idx[t++] = a0 + 2 }
        return MeshData(pos, nrm, uv, idx)
    }

    /** A convex polygon of up to 6 vertices as a double-sided fan (centre + 6 rim vertices). Filled by [fanInto]. */
    fun fanMesh(): MeshData {
        val idx = IntArray(6 * 6); var t = 0
        for (k in 1..6) { val b = if (k == 6) 1 else k + 1; idx[t++] = 0; idx[t++] = k; idx[t++] = b; idx[t++] = 0; idx[t++] = b; idx[t++] = k }
        val nrm = FloatArray(7 * 3); for (k in 0 until 7) nrm[k * 3 + 1] = 1f
        val uv = FloatArray(7 * 2) { 0.5f }
        return MeshData(FloatArray(7 * 3), nrm, uv, idx)
    }

    fun fanInto(md: MeshData, pts: List<P3>) {
        val m = pts.size.coerceAtMost(6)
        if (m == 0) { md.pos.fill(0f); return }
        var cx = 0f; var cy = 0f; var cz = 0f
        for (k in 0 until m) { cx += pts[k].x; cy += pts[k].y; cz += pts[k].z }
        OptGeo.setPos(md, 0, cx / m, cy / m, cz / m)
        for (k in 0 until 6) { val p = pts[minOf(k, m - 1)]; OptGeo.setPos(md, k + 1, p.x, p.y, p.z) }
    }

    /** Point a unit cylinder (axis +Y, height 1, centred) from [a] to [b] with radius [r]; skips no-op updates. */
    fun line(n: Node, a: P3, b: P3, r: Float) {
        val dx = b.x - a.x; val dy = b.y - a.y; val dz = b.z - a.z
        val len = sqrt(dx * dx + dy * dy + dz * dz).coerceAtLeast(1e-4f)
        val p = P3((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2); val sc = P3(r, len, r)
        if (n.pos == p && n.scale == sc) return
        n.pos = p; n.rot = Q.fromTo(0f, 1f, 0f, dx, dy, dz); n.scale = sc
    }

    /** Cone along +Y: base radius [r] at y = 0, apex at y = [h]. */
    fun cone(r: Float, h: Float, seg: Int = 18): MeshData {
        val pos = ArrayList<Float>(); val nrm = ArrayList<Float>(); val uv = ArrayList<Float>(); val idx = ArrayList<Int>()
        fun v(x: Float, y: Float, z: Float, nx: Float, ny: Float, nz: Float) { pos += x; pos += y; pos += z; nrm += nx; nrm += ny; nrm += nz; uv += 0.5f; uv += 0.5f }
        val sl = sqrt(r * r + h * h)
        for (s in 0 until seg) {
            val a0 = 2.0 * Math.PI * s / seg; val a1 = 2.0 * Math.PI * (s + 1) / seg; val am = (a0 + a1) / 2
            val b = pos.size / 3
            val nx = (cos(am) * h / sl).toFloat(); val nz = (sin(am) * h / sl).toFloat(); val ny = r / sl
            v((cos(a0) * r).toFloat(), 0f, (sin(a0) * r).toFloat(), nx, ny, nz); v(0f, h, 0f, nx, ny, nz); v((cos(a1) * r).toFloat(), 0f, (sin(a1) * r).toFloat(), nx, ny, nz)
            idx += b; idx += b + 1; idx += b + 2
            val c = pos.size / 3
            v(0f, 0f, 0f, 0f, -1f, 0f); v((cos(a0) * r).toFloat(), 0f, (sin(a0) * r).toFloat(), 0f, -1f, 0f); v((cos(a1) * r).toFloat(), 0f, (sin(a1) * r).toFloat(), 0f, -1f, 0f)
            idx += c; idx += c + 1; idx += c + 2
        }
        return MeshData(pos.toFloatArray(), nrm.toFloatArray(), uv.toFloatArray(), idx.toIntArray())
    }
}
