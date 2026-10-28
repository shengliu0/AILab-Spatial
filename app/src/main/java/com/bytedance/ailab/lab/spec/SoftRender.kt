// SoftRender — a small CPU rasteriser for Spec scenes, used ONLY by JVM tests to produce review
// frames on the Mac (the emulator download is network-blocked, and the headset only yields frames
// while worn). It is a stand-in for the headset renderer, not a copy of it:
//   * Unlit   : texture x colour, exactly as on device (tone mapping off there too)
//   * Pbr     : Lambert + normalised Blinn-Phong with Schlick Fresnel under a key light, a cool
//               fill and a sky/ground hemisphere that stands in for the room IBL; Reinhard tone map
//   * blending: opaque (z-write, back-face cull), then TRANSPARENT/ADD sorted back-to-front,
//               double-sided, z-test without z-write — the same rules lab/Gfx applies on device
// Good enough to judge composition, colour, contrast, readability and "does it look like a toy",
// which is what the review loop scores. Final look is still confirmed on the headset.

package com.bytedance.ailab.lab.spec

import com.bytedance.ailab.core.P3
import com.bytedance.ailab.render.Image
import com.bytedance.ailab.render.ProcTex
import com.bytedance.ailab.render.Rgb
import java.io.File
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.math.tan

class Cam(val eye: P3, val target: P3, val fovDeg: Float = 55f, val name: String = "cam")

object SoftRender {
    // backdrop = the Unity ShotDriver's neutral dark grey, so scores compare with the Unity baseline
    val BACKDROP = Rgb(0.055f, 0.06f, 0.075f)

    // ---------------------------------------------------------------- framing

    /** World-space AABB of all visible meshes (min, max). */
    fun bounds(root: Node): Pair<P3, P3> {
        var lo = floatArrayOf(1e9f, 1e9f, 1e9f); var hi = floatArrayOf(-1e9f, -1e9f, -1e9f)
        forEachMesh(root) { n, w ->
            val m = n.mesh!!
            for (k in 0 until m.vertexCount) {
                val p = w.apply(m.pos[k * 3], m.pos[k * 3 + 1], m.pos[k * 3 + 2])
                for (a in 0 until 3) { lo[a] = min(lo[a], p[a]); hi[a] = max(hi[a], p[a]) }
            }
        }
        if (lo[0] > hi[0]) { lo = floatArrayOf(-0.3f, -0.3f, -0.3f); hi = floatArrayOf(0.3f, 0.3f, 0.3f) }
        return P3(lo[0], lo[1], lo[2]) to P3(hi[0], hi[1], hi[2])
    }

    /**
     * The review framings: the four the Unity ShotDriver used (hero 3/4, level side, close detail —
     * adapted to bounds) plus "viewer": where a Shared-Space user actually stands (2.5 m in front).
     */
    fun framings(root: Node): List<Cam> {
        val (lo, hi) = bounds(root)
        val c = P3((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2)
        val ext = P3((hi.x - lo.x) / 2, (hi.y - lo.y) / 2, (hi.z - lo.z) / 2)
        val r = max(0.25f, ext.length)
        fun pose(name: String, dx: Float, dy: Float, dz: Float, dist: Float, height: Float): Cam {
            val l = sqrt(dx * dx + dy * dy + dz * dz)
            val t = P3(c.x, c.y + height * r, c.z)
            val d = r * dist * 1.9f
            return Cam(P3(t.x + dx / l * d, t.y + dy / l * d, t.z + dz / l * d), t, 55f, name)
        }
        return listOf(
            pose("hero", 0.55f, 0.42f, 1f, 1.55f, 0.20f),
            pose("side", 1f, 0.10f, 0.15f, 1.45f, 0.05f),
            pose("detail", 0.25f, 0.30f, 1f, 0.85f, 0.10f),
            Cam(P3(0f, 0.10f, 2.5f), P3(0f, 0.02f, 0f), 55f, "viewer"),
        )
    }

    // ---------------------------------------------------------------- render

    fun render(root: Node, cam: Cam, w: Int = 1280, h: Int = 960, ss: Int = 2, bg: Rgb = BACKDROP): Image {
        val W = w * ss; val H = h * ss
        val col = FloatArray(W * H * 3)
        val zb = FloatArray(W * H) { Float.MAX_VALUE }
        val bgl = floatArrayOf(lin(bg.r), lin(bg.g), lin(bg.b))
        for (i in 0 until W * H) { col[i * 3] = bgl[0]; col[i * 3 + 1] = bgl[1]; col[i * 3 + 2] = bgl[2] }

        // camera basis
        val f = norm(sub(cam.target, cam.eye))
        var r = cross(f, floatArrayOf(0f, 1f, 0f)); if (len(r) < 1e-6f) r = floatArrayOf(1f, 0f, 0f)
        r = norm(r); val u = cross(r, f)
        val th = tan(Math.toRadians(cam.fovDeg / 2.0).toFloat()); val aspect = W.toFloat() / H
        val eye = floatArrayOf(cam.eye.x, cam.eye.y, cam.eye.z)

        val tris = ArrayList<Tri>()
        forEachMesh(root, cam.eye) { n, wm ->
            val mesh = n.mesh!!; val mat = n.mat ?: return@forEachMesh
            val nmat = wm.normalMatrix()
            val vs = Array(mesh.vertexCount) { k ->
                val p = wm.apply(mesh.pos[k * 3], mesh.pos[k * 3 + 1], mesh.pos[k * 3 + 2])
                val nn = norm(nmat.mul(mesh.nrm[k * 3], mesh.nrm[k * 3 + 1], mesh.nrm[k * 3 + 2]))
                val d = floatArrayOf(p[0] - eye[0], p[1] - eye[1], p[2] - eye[2])
                val vx = dot(d, r); val vy = dot(d, u); val vz = dot(d, f)
                V(p, nn, mesh.uv[k * 2], mesh.uv[k * 2 + 1],
                  (vx / vz / th / aspect * 0.5f + 0.5f) * W, (-vy / vz / th * 0.5f + 0.5f) * H, vz)
            }
            for (t in 0 until mesh.triangleCount) {
                val a = vs[mesh.idx[t * 3]]; val b = vs[mesh.idx[t * 3 + 1]]; val c = vs[mesh.idx[t * 3 + 2]]
                if (a.z < 0.02f || b.z < 0.02f || c.z < 0.02f) continue          // behind near plane
                tris += Tri(a, b, c, mat, (a.z + b.z + c.z) / 3f)
            }
        }
        val ctx = Ctx(W, H, col, zb, eye)
        for (t in tris) if (isOpaque(t.mat)) raster(ctx, t, cull = true, zWrite = true)
        val trans = tris.filter { !isOpaque(it.mat) }.sortedByDescending { it.depth }
        for (t in trans) raster(ctx, t, cull = false, zWrite = false)

        // downsample + tone map + encode
        val out = Image(w, h)
        for (y in 0 until h) for (x in 0 until w) {
            var cr = 0f; var cg = 0f; var cb = 0f
            for (sy in 0 until ss) for (sx in 0 until ss) {
                val i = ((y * ss + sy) * W + (x * ss + sx)) * 3
                cr += col[i]; cg += col[i + 1]; cb += col[i + 2]
            }
            val k = 1f / (ss * ss)
            out.px[y * w + x] = ProcTex.argb(1f, enc(cr * k), enc(cg * k), enc(cb * k))
        }
        return out
    }

    // ---------------------------------------------------------------- internals

    private class V(val p: FloatArray, val n: FloatArray, val u: Float, val v: Float, val sx: Float, val sy: Float, val z: Float)
    private class Tri(val a: V, val b: V, val c: V, val mat: Mat, val depth: Float)
    private class Ctx(val W: Int, val H: Int, val col: FloatArray, val zb: FloatArray, val eye: FloatArray)

    private fun isOpaque(m: Mat) = when (m) {
        is Mat.Unlit -> m.blend == Blend.OPAQUE
        is Mat.Pbr -> m.alpha >= 0.999f
    }

    private fun raster(c: Ctx, t: Tri, cull: Boolean, zWrite: Boolean) {
        val a = t.a; val b = t.b; val d = t.c
        val area = (b.sx - a.sx) * (d.sy - a.sy) - (b.sy - a.sy) * (d.sx - a.sx)
        if (abs(area) < 1e-9f) return
        // screen y points down: a counter-clockwise (front-facing) triangle has NEGATIVE area here
        if (cull && area > 0) return
        val x0 = max(0, floor(min(a.sx, min(b.sx, d.sx))).toInt()); val x1 = min(c.W - 1, floor(max(a.sx, max(b.sx, d.sx))).toInt())
        val y0 = max(0, floor(min(a.sy, min(b.sy, d.sy))).toInt()); val y1 = min(c.H - 1, floor(max(a.sy, max(b.sy, d.sy))).toInt())
        if (x0 > x1 || y0 > y1) return
        val iza = 1f / a.z; val izb = 1f / b.z; val izc = 1f / d.z
        for (y in y0..y1) {
            val py = y + 0.5f
            for (x in x0..x1) {
                val px = x + 0.5f
                var w0 = (b.sx - px) * (d.sy - py) - (b.sy - py) * (d.sx - px)
                var w1 = (d.sx - px) * (a.sy - py) - (d.sy - py) * (a.sx - px)
                var w2 = (a.sx - px) * (b.sy - py) - (a.sy - py) * (b.sx - px)
                if (area > 0) { if (w0 < 0 || w1 < 0 || w2 < 0) continue } else { if (w0 > 0 || w1 > 0 || w2 > 0) continue }
                w0 /= area; w1 /= area; w2 /= area
                // perspective-correct weights
                val iz = w0 * iza + w1 * izb + w2 * izc
                val z = 1f / iz
                val idx = y * c.W + x
                if (z >= c.zb[idx]) continue
                val q0 = w0 * iza * z; val q1 = w1 * izb * z; val q2 = w2 * izc * z
                val u = q0 * a.u + q1 * b.u + q2 * d.u; val v = q0 * a.v + q1 * b.v + q2 * d.v
                val nx = q0 * a.n[0] + q1 * b.n[0] + q2 * d.n[0]
                val ny = q0 * a.n[1] + q1 * b.n[1] + q2 * d.n[1]
                val nz = q0 * a.n[2] + q1 * b.n[2] + q2 * d.n[2]
                val wx = q0 * a.p[0] + q1 * b.p[0] + q2 * d.p[0]
                val wy = q0 * a.p[1] + q1 * b.p[1] + q2 * d.p[1]
                val wz = q0 * a.p[2] + q1 * b.p[2] + q2 * d.p[2]
                shade(c, t.mat, idx, u, v, nx, ny, nz, wx, wy, wz)
                if (zWrite) c.zb[idx] = z
            }
        }
    }

    private val KEY = norm(floatArrayOf(0.45f, 0.85f, 0.55f))      // toward the key light
    private val FILL = norm(floatArrayOf(-0.6f, 0.25f, -0.4f))
    private val KEY_C = floatArrayOf(2.6f, 2.45f, 2.25f)
    private val FILL_C = floatArrayOf(0.35f, 0.42f, 0.55f)
    private val SKY = floatArrayOf(0.34f, 0.36f, 0.40f)
    private val GROUND = floatArrayOf(0.13f, 0.12f, 0.11f)

    private fun shade(c: Ctx, m: Mat, idx: Int, u: Float, v: Float, nx0: Float, ny0: Float, nz0: Float,
                      wx: Float, wy: Float, wz: Float) {
        val o = idx * 3
        when (m) {
            is Mat.Unlit -> {
                var r = lin(m.color.r); var g = lin(m.color.g); var b = lin(m.color.b); var a = m.alpha
                m.tex?.let { t -> val s = sample(t, u, v); r *= s[0]; g *= s[1]; b *= s[2]; a *= s[3] }
                blend(c.col, o, r, g, b, a, if (m.blend == Blend.ADD) 2 else if (m.blend == Blend.TRANSPARENT) 1 else 0)
            }
            is Mat.Pbr -> {
                var br = lin(m.color.r); var bg = lin(m.color.g); var bb = lin(m.color.b); var a = m.alpha
                m.tex?.let { t -> val s = sample(t, u, v); br *= s[0]; bg *= s[1]; bb *= s[2]; a *= s[3] }
                var n = norm(floatArrayOf(nx0, ny0, nz0))
                val vdir = norm(floatArrayOf(c.eye[0] - wx, c.eye[1] - wy, c.eye[2] - wz))
                if (dot(n, vdir) < 0f) n = floatArrayOf(-n[0], -n[1], -n[2])      // double-sided glass
                val met = m.metallic.coerceIn(0f, 1f); val rough = m.roughness.coerceIn(0.04f, 1f)
                val f0r = 0.04f + (br - 0.04f) * met; val f0g = 0.04f + (bg - 0.04f) * met; val f0b = 0.04f + (bb - 0.04f) * met
                val dr = br * (1 - met); val dg = bg * (1 - met); val db = bb * (1 - met)
                val sp = (2f / rough.pow(4) - 2f).coerceIn(1f, 2048f)
                var cr = 0f; var cg = 0f; var cb = 0f
                for (li in 0..1) {
                    val l = if (li == 0) KEY else FILL; val lc = if (li == 0) KEY_C else FILL_C
                    val ndl = max(0f, dot(n, l)); if (ndl <= 0f) continue
                    val hv = norm(floatArrayOf(l[0] + vdir[0], l[1] + vdir[1], l[2] + vdir[2]))
                    val ndh = max(0f, dot(n, hv)); val vdh = max(0f, dot(vdir, hv))
                    val fr = (1 - vdh).pow(5)
                    val spec = (sp + 8f) / (8f * PI.toFloat()) * ndh.pow(sp) * ndl
                    cr += lc[0] * (dr / PI.toFloat() * ndl + spec * (f0r + (1 - f0r) * fr))
                    cg += lc[1] * (dg / PI.toFloat() * ndl + spec * (f0g + (1 - f0g) * fr))
                    cb += lc[2] * (db / PI.toFloat() * ndl + spec * (f0b + (1 - f0b) * fr))
                }
                // hemisphere ambient (diffuse) + environment reflection (spec), the room stand-in
                val hd = 0.5f + 0.5f * n[1]
                cr += dr * (GROUND[0] + (SKY[0] - GROUND[0]) * hd); cg += dg * (GROUND[1] + (SKY[1] - GROUND[1]) * hd); cb += db * (GROUND[2] + (SKY[2] - GROUND[2]) * hd)
                // environment specular, roughness-aware Fresnel (Karis): a rough dielectric must
                // NOT turn into a mirror at grazing angles (it did, and a charcoal slab read as grey)
                val ndv = max(0f, dot(n, vdir))
                val rx = 2 * ndv * n[0] - vdir[0]; val ry = 2 * ndv * n[1] - vdir[1]
                val hr = (0.5f + 0.5f * ry + 0.12f * rx).coerceIn(0f, 1f)
                val fw = (1 - ndv).pow(5)
                val fmr = max(1 - rough, f0r); val fmg = max(1 - rough, f0g); val fmb = max(1 - rough, f0b)
                val er = f0r + (fmr - f0r) * fw; val eg = f0g + (fmg - f0g) * fw; val eb = f0b + (fmb - f0b) * fw
                val blur = 1f - 0.5f * rough
                cr += er * blur * (GROUND[0] + (SKY[0] * 1.3f - GROUND[0]) * hr)
                cg += eg * blur * (GROUND[1] + (SKY[1] * 1.3f - GROUND[1]) * hr)
                cb += eb * blur * (GROUND[2] + (SKY[2] * 1.3f - GROUND[2]) * hr)
                m.emissive?.let { e -> cr += lin(e.r); cg += lin(e.g); cb += lin(e.b) }
                // Reinhard (the device tone-maps PBR by default)
                cr = cr / (1 + cr * 0.6f); cg = cg / (1 + cg * 0.6f); cb = cb / (1 + cb * 0.6f)
                blend(c.col, o, cr, cg, cb, a, if (a < 0.999f) 1 else 0)
            }
        }
    }

    private fun blend(col: FloatArray, o: Int, r: Float, g: Float, b: Float, a: Float, mode: Int) {
        when (mode) {
            0 -> { col[o] = r; col[o + 1] = g; col[o + 2] = b }
            1 -> { col[o] = col[o] * (1 - a) + r * a; col[o + 1] = col[o + 1] * (1 - a) + g * a; col[o + 2] = col[o + 2] * (1 - a) + b * a }
            else -> { col[o] += r * a; col[o + 1] += g * a; col[o + 2] += b * a }
        }
    }

    /** Bilinear, U wraps, V clamps; row 0 = top = v 0 (MeshGen's convention). Linear RGB + alpha. */
    private fun sample(t: Image, u0: Float, v0: Float): FloatArray {
        val u = u0 - floor(u0); val v = v0.coerceIn(0f, 1f)
        val fx = u * t.w - 0.5f; val fy = v * (t.h - 1)
        val x0 = floor(fx).toInt(); val y0 = floor(fy).toInt().coerceIn(0, t.h - 1)
        val tx = fx - x0; val ty = fy - floor(fy)
        val y1 = min(t.h - 1, y0 + 1)
        val out = FloatArray(4)
        for ((wy, yy) in listOf(1 - ty to y0, ty to y1)) for ((wx, xx) in listOf(1 - tx to x0, tx to x0 + 1)) {
            val p = t.px[yy * t.w + ((xx % t.w) + t.w) % t.w]
            val wgt = wx * wy
            out[0] += lin(((p ushr 16) and 0xFF) / 255f) * wgt
            out[1] += lin(((p ushr 8) and 0xFF) / 255f) * wgt
            out[2] += lin((p and 0xFF) / 255f) * wgt
            out[3] += ((p ushr 24) and 0xFF) / 255f * wgt
        }
        return out
    }

    // ---------------------------------------------------------------- transforms

    /** Affine world transform: p' = M p + t. */
    class Aff(val m: FloatArray, val t: FloatArray) {
        fun apply(x: Float, y: Float, z: Float) = floatArrayOf(
            m[0] * x + m[1] * y + m[2] * z + t[0], m[3] * x + m[4] * y + m[5] * z + t[1], m[6] * x + m[7] * y + m[8] * z + t[2])
        fun mul(x: Float, y: Float, z: Float) = floatArrayOf(m[0] * x + m[1] * y + m[2] * z, m[3] * x + m[4] * y + m[5] * z, m[6] * x + m[7] * y + m[8] * z)
        fun then(child: Aff): Aff {
            val r = FloatArray(9)
            for (i in 0 until 3) for (j in 0 until 3) r[i * 3 + j] = m[i * 3] * child.m[j] + m[i * 3 + 1] * child.m[3 + j] + m[i * 3 + 2] * child.m[6 + j]
            return Aff(r, apply(child.t[0], child.t[1], child.t[2]))
        }
        /** inverse-transpose of M, for normals under non-uniform scale */
        fun normalMatrix(): Aff {
            val a = m
            val det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6])
            val id = if (abs(det) < 1e-20f) 0f else 1f / det
            // cofactor matrix / det == inverse-transpose
            val c = floatArrayOf(
                (a[4] * a[8] - a[5] * a[7]) * id, -(a[3] * a[8] - a[5] * a[6]) * id, (a[3] * a[7] - a[4] * a[6]) * id,
                -(a[1] * a[8] - a[2] * a[7]) * id, (a[0] * a[8] - a[2] * a[6]) * id, -(a[0] * a[7] - a[1] * a[6]) * id,
                (a[1] * a[5] - a[2] * a[4]) * id, -(a[0] * a[5] - a[2] * a[3]) * id, (a[0] * a[4] - a[1] * a[3]) * id)
            return Aff(c, floatArrayOf(0f, 0f, 0f))
        }
        companion object {
            fun local(n: Node): Aff {
                val q = n.rot
                val xx = q.x * q.x; val yy = q.y * q.y; val zz = q.z * q.z
                val xy = q.x * q.y; val xz = q.x * q.z; val yz = q.y * q.z; val wx = q.w * q.x; val wy = q.w * q.y; val wz = q.w * q.z
                val r = floatArrayOf(
                    1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
                    2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
                    2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy))
                val s = n.scale
                for (i in 0 until 3) { r[i * 3] *= s.x; r[i * 3 + 1] *= s.y; r[i * 3 + 2] *= s.z }
                return Aff(r, floatArrayOf(n.pos.x, n.pos.y, n.pos.z))
            }
            val ID = Aff(floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f), floatArrayOf(0f, 0f, 0f))
        }
    }

    /** World (= demo-root / volume) transform of [n], composed from its parents (billboards ignored). */
    fun worldOf(n: Node): Aff {
        val chain = ArrayList<Node>(); var p: Node? = n
        while (p != null) { chain += p; p = p.parent }
        var a = Aff.ID
        for (k in chain.indices.reversed()) a = a.then(Aff.local(chain[k]))
        return a
    }

    fun forEachMesh(root: Node, eye: P3? = null, f: (Node, Aff) -> Unit) {
        fun rec(n: Node, parent: Aff) {
            if (!n.visible) return
            val local = if (n.billboard && eye != null) {
                // turn +Z toward the eye (the device does this with LookAtComponent(viewer))
                val p = parent.apply(n.pos.x, n.pos.y, n.pos.z)
                val inv = parent.normalMatrix()      // inverse-transpose; for rotation*scale its transpose ~ inverse direction map
                val d = floatArrayOf(eye.x - p[0], eye.y - p[1], eye.z - p[2])
                // direction into the parent frame: M^-1 d = (inverse-transpose)^T d
                val m = inv.m
                val dl = floatArrayOf(m[0] * d[0] + m[3] * d[1] + m[6] * d[2], m[1] * d[0] + m[4] * d[1] + m[7] * d[2], m[2] * d[0] + m[5] * d[1] + m[8] * d[2])
                val q = com.bytedance.ailab.lab.Q.fromTo(0f, 0f, 1f, dl[0], dl[1], dl[2])
                val tmp = Node("_bb").also { it.pos = n.pos; it.rot = q; it.scale = n.scale }
                Aff.local(tmp)
            } else Aff.local(n)
            val w = parent.then(local)
            if (n.mesh != null && n.mat != null) f(n, w)
            for (ch in n.children) rec(ch, w)
        }
        rec(root, Aff.ID)
    }

    // ---------------------------------------------------------------- small maths

    private fun lin(c: Float) = ProcTex.toLinear(c.coerceIn(0f, 1f))
    private fun enc(l0: Float): Float { val l = l0.coerceIn(0f, 1f); return if (l <= 0.0031308f) l * 12.92f else 1.055f * l.pow(1f / 2.4f) - 0.055f }
    private fun sub(a: P3, b: P3) = floatArrayOf(a.x - b.x, a.y - b.y, a.z - b.z)
    private fun dot(a: FloatArray, b: FloatArray) = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    private fun len(a: FloatArray) = sqrt(dot(a, a))
    private fun norm(a: FloatArray): FloatArray { val l = len(a).coerceAtLeast(1e-12f); return floatArrayOf(a[0] / l, a[1] / l, a[2] / l) }
    private fun cross(a: FloatArray, b: FloatArray) = floatArrayOf(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
    @Suppress("unused") private fun soft(x: Float) = 1f - exp(-x)

    // ---------------------------------------------------------------- PNG

    /** Minimal RGBA PNG writer (works on JVM tests without javax.imageio). */
    fun writePng(img: Image, f: File) {
        val raw = java.io.ByteArrayOutputStream(img.w * img.h * 4 + img.h)
        for (y in 0 until img.h) {
            raw.write(0)
            for (x in 0 until img.w) {
                val c = img.px[y * img.w + x]
                raw.write((c ushr 16) and 0xFF); raw.write((c ushr 8) and 0xFF); raw.write(c and 0xFF); raw.write((c ushr 24) and 0xFF)
            }
        }
        f.parentFile?.mkdirs()
        java.io.DataOutputStream(java.io.FileOutputStream(f)).use { out ->
            out.write(byteArrayOf(0x89.toByte(), 'P'.code.toByte(), 'N'.code.toByte(), 'G'.code.toByte(), 13, 10, 26, 10))
            fun chunk(type: String, data: ByteArray) {
                out.writeInt(data.size)
                val td = type.toByteArray() + data
                out.write(td)
                val crc = java.util.zip.CRC32(); crc.update(td); out.writeInt(crc.value.toInt())
            }
            chunk("IHDR", java.nio.ByteBuffer.allocate(13).putInt(img.w).putInt(img.h).put(8).put(6).put(0).put(0).put(0).array())
            val z = java.io.ByteArrayOutputStream()
            java.util.zip.DeflaterOutputStream(z).use { it.write(raw.toByteArray()) }
            chunk("IDAT", z.toByteArray())
            chunk("IEND", ByteArray(0))
        }
    }
}
