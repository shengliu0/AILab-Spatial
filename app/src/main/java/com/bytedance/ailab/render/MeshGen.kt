// MeshGen — procedural geometry as plain arrays (no SDK types), so it is unit-testable on the JVM.
// The scene converts a MeshData into an SDK MeshModel in one place (SolarScene.toModel).
//
// Every triangle is wound counter-clockwise seen from OUTSIDE (right-handed, outward normal), which
// the tests check numerically. The scene still renders the shells / orbits / rings double-sided,
// so a wrong guess about the engine's front-face rule can cost a little fill but never a hole.

package com.bytedance.ailab.render

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

class MeshData(
    val pos: FloatArray,      // xyz per vertex
    val nrm: FloatArray,      // xyz per vertex
    val uv: FloatArray,       // uv per vertex
    val idx: IntArray,        // 3 per triangle
) {
    val vertexCount get() = pos.size / 3
    val triangleCount get() = idx.size / 3
}

object MeshGen {

    enum class SphereUv {
        /** u = longitude / 2pi, v = latitude fraction from +Y: equirectangular albedo maps. */
        EQUIRECT,
        /** u = angle from +Y / pi, v = longitude / 2pi: for the 1-D shell strips that vary on U. */
        POLAR_U,
    }

    /** UV sphere of radius [r] about +Y. [lat] rings x [lon] segments. */
    fun sphere(r: Float, lat: Int, lon: Int, layout: SphereUv): MeshData {
        val nv = (lat + 1) * (lon + 1)
        val pos = FloatArray(nv * 3); val nrm = FloatArray(nv * 3); val uv = FloatArray(nv * 2)
        var k = 0
        for (i in 0..lat) {
            val theta = PI * i / lat                 // 0 at +Y
            val st = sin(theta).toFloat(); val ct = cos(theta).toFloat()
            for (j in 0..lon) {
                val phi = 2.0 * PI * j / lon
                val x = st * cos(phi).toFloat(); val z = st * sin(phi).toFloat()
                pos[k * 3] = x * r; pos[k * 3 + 1] = ct * r; pos[k * 3 + 2] = z * r
                nrm[k * 3] = x; nrm[k * 3 + 1] = ct; nrm[k * 3 + 2] = z
                when (layout) {
                    SphereUv.EQUIRECT -> { uv[k * 2] = j.toFloat() / lon; uv[k * 2 + 1] = i.toFloat() / lat }
                    SphereUv.POLAR_U -> { uv[k * 2] = i.toFloat() / lat; uv[k * 2 + 1] = j.toFloat() / lon }
                }
                k++
            }
        }
        val idx = IntArray(lat * lon * 6)
        var t = 0
        for (i in 0 until lat) for (j in 0 until lon) {
            val a = i * (lon + 1) + j; val b = a + lon + 1
            // a(i,j) b(i+1,j) a+1(i,j+1) b+1(i+1,j+1); outward CCW for phi increasing toward +z
            idx[t++] = a; idx[t++] = a + 1; idx[t++] = b
            idx[t++] = a + 1; idx[t++] = b + 1; idx[t++] = b
        }
        return MeshData(pos, nrm, uv, idx)
    }

    /** A tube of radius [r] with [sides] facets swept along [path] (xyz triples). */
    fun tube(path: FloatArray, r: Float, sides: Int, closed: Boolean): MeshData {
        val n = path.size / 3
        val rings = if (closed) n + 1 else n
        val pos = FloatArray(rings * (sides + 1) * 3)
        val nrm = FloatArray(rings * (sides + 1) * 3)
        val uv = FloatArray(rings * (sides + 1) * 2)
        // parallel-transport a normal along the curve so the tube never twists
        var px = 0f; var py = 1f; var pz = 0f
        var k = 0
        for (ri in 0 until rings) {
            val i = ri % n
            val ip = if (closed) (i + 1) % n else minOf(i + 1, n - 1)
            val im = if (closed) (i - 1 + n) % n else maxOf(i - 1, 0)
            var tx = path[ip * 3] - path[im * 3]
            var ty = path[ip * 3 + 1] - path[im * 3 + 1]
            var tz = path[ip * 3 + 2] - path[im * 3 + 2]
            val tl = sqrt(tx * tx + ty * ty + tz * tz).coerceAtLeast(1e-12f)
            tx /= tl; ty /= tl; tz /= tl
            // remove the tangent component from the carried normal, renormalise
            val d = px * tx + py * ty + pz * tz
            px -= d * tx; py -= d * ty; pz -= d * tz
            var pl = sqrt(px * px + py * py + pz * pz)
            if (pl < 1e-6f) { // tangent parallel to the carried normal: pick any perpendicular
                px = -ty; py = tx; pz = 0f; pl = sqrt(px * px + py * py)
                if (pl < 1e-6f) { px = 0f; py = -tz; pz = ty; pl = sqrt(py * py + pz * pz) }
            }
            px /= pl; py /= pl; pz /= pl
            // binormal = t x n
            val bx = ty * pz - tz * py; val by = tz * px - tx * pz; val bz = tx * py - ty * px
            for (s in 0..sides) {
                val a = 2.0 * PI * s / sides
                val ca = cos(a).toFloat(); val sa = sin(a).toFloat()
                val nx = px * ca + bx * sa; val ny = py * ca + by * sa; val nz = pz * ca + bz * sa
                pos[k * 3] = path[i * 3] + nx * r
                pos[k * 3 + 1] = path[i * 3 + 1] + ny * r
                pos[k * 3 + 2] = path[i * 3 + 2] + nz * r
                nrm[k * 3] = nx; nrm[k * 3 + 1] = ny; nrm[k * 3 + 2] = nz
                uv[k * 2] = ri.toFloat() / (rings - 1); uv[k * 2 + 1] = s.toFloat() / sides
                k++
            }
        }
        val segs = rings - 1
        val idx = IntArray(segs * sides * 6)
        var t = 0
        for (ri in 0 until segs) for (s in 0 until sides) {
            val a = ri * (sides + 1) + s; val b = a + sides + 1
            idx[t++] = a; idx[t++] = a + 1; idx[t++] = b
            idx[t++] = a + 1; idx[t++] = b + 1; idx[t++] = b
        }
        return MeshData(pos, nrm, uv, idx)
    }

    /** Flat annulus in the XZ plane, u = radial fraction (rIn -> rOut), normal +Y. */
    fun annulus(rIn: Float, rOut: Float, seg: Int): MeshData {
        val nv = (seg + 1) * 2
        val pos = FloatArray(nv * 3); val nrm = FloatArray(nv * 3); val uv = FloatArray(nv * 2)
        for (s in 0..seg) {
            val a = 2.0 * PI * s / seg
            val ca = cos(a).toFloat(); val sa = sin(a).toFloat()
            for (k in 0..1) {
                val v = s * 2 + k
                val rr = if (k == 0) rIn else rOut
                pos[v * 3] = ca * rr; pos[v * 3 + 1] = 0f; pos[v * 3 + 2] = sa * rr
                nrm[v * 3 + 1] = 1f
                uv[v * 2] = k.toFloat(); uv[v * 2 + 1] = s.toFloat() / seg
            }
        }
        val idx = IntArray(seg * 6)
        var t = 0
        for (s in 0 until seg) {
            val a0 = s * 2; val a1 = a0 + 1; val b0 = a0 + 2; val b1 = a0 + 3
            // CCW seen from +Y
            idx[t++] = a0; idx[t++] = b0; idx[t++] = a1
            idx[t++] = a1; idx[t++] = b0; idx[t++] = b1
        }
        return MeshData(pos, nrm, uv, idx)
    }

    /** Closed cylinder along +Y, centred at the origin: radius [r], height [h]. */
    fun cylinder(r: Float, h: Float, seg: Int = 24): MeshData {
        val pos = ArrayList<Float>(); val nrm = ArrayList<Float>(); val uv = ArrayList<Float>(); val idx = ArrayList<Int>()
        fun v(x: Float, y: Float, z: Float, nx: Float, ny: Float, nz: Float, u: Float, w: Float): Int {
            pos += x; pos += y; pos += z; nrm += nx; nrm += ny; nrm += nz; uv += u; uv += w
            return pos.size / 3 - 1
        }
        val hh = h / 2
        // side
        val base = pos.size / 3
        for (s in 0..seg) {
            val a = 2.0 * PI * s / seg; val c = cos(a).toFloat(); val sn = sin(a).toFloat()
            v(c * r, -hh, sn * r, c, 0f, sn, s.toFloat() / seg, 1f)
            v(c * r, hh, sn * r, c, 0f, sn, s.toFloat() / seg, 0f)
        }
        for (s in 0 until seg) {
            val a0 = base + s * 2; val a1 = a0 + 1; val b0 = a0 + 2; val b1 = a0 + 3
            // outward CCW: (a0, a1, b0) faces -phi... wind so normal points out
            idx += a0; idx += a1; idx += b0; idx += b0; idx += a1; idx += b1
        }
        // caps
        for (top in listOf(false, true)) {
            val y = if (top) hh else -hh; val ny = if (top) 1f else -1f
            val ctr = v(0f, y, 0f, 0f, ny, 0f, 0.5f, 0.5f)
            val ring = pos.size / 3
            for (s in 0..seg) {
                val a = 2.0 * PI * s / seg; val c = cos(a).toFloat(); val sn = sin(a).toFloat()
                v(c * r, y, sn * r, 0f, ny, 0f, 0.5f + 0.5f * c, 0.5f + 0.5f * sn)
            }
            for (s in 0 until seg) {
                if (top) { idx += ctr; idx += ring + s + 1; idx += ring + s }
                else { idx += ctr; idx += ring + s; idx += ring + s + 1 }
            }
        }
        return MeshData(pos.toFloatArray(), nrm.toFloatArray(), uv.toFloatArray(), idx.toIntArray())
    }

    /** Axis-aligned box centred at the origin, flat-shaded faces, outward CCW. */
    fun box(sx: Float, sy: Float, sz: Float): MeshData {
        val hx = sx / 2; val hy = sy / 2; val hz = sz / 2
        // each face: normal, then 4 corners CCW seen from outside
        val faces = arrayOf(
            floatArrayOf(1f, 0f, 0f, hx, -hy, hz, hx, -hy, -hz, hx, hy, -hz, hx, hy, hz),
            floatArrayOf(-1f, 0f, 0f, -hx, -hy, -hz, -hx, -hy, hz, -hx, hy, hz, -hx, hy, -hz),
            floatArrayOf(0f, 1f, 0f, -hx, hy, hz, hx, hy, hz, hx, hy, -hz, -hx, hy, -hz),
            floatArrayOf(0f, -1f, 0f, -hx, -hy, -hz, hx, -hy, -hz, hx, -hy, hz, -hx, -hy, hz),
            floatArrayOf(0f, 0f, 1f, -hx, -hy, hz, hx, -hy, hz, hx, hy, hz, -hx, hy, hz),
            floatArrayOf(0f, 0f, -1f, hx, -hy, -hz, -hx, -hy, -hz, -hx, hy, -hz, hx, hy, -hz),
        )
        val pos = FloatArray(24 * 3); val nrm = FloatArray(24 * 3); val uv = FloatArray(24 * 2); val idx = IntArray(36)
        val uvs = floatArrayOf(0f, 1f, 1f, 1f, 1f, 0f, 0f, 0f)
        for (f in 0 until 6) {
            val d = faces[f]
            for (k in 0 until 4) {
                val vi = f * 4 + k
                pos[vi * 3] = d[3 + k * 3]; pos[vi * 3 + 1] = d[4 + k * 3]; pos[vi * 3 + 2] = d[5 + k * 3]
                nrm[vi * 3] = d[0]; nrm[vi * 3 + 1] = d[1]; nrm[vi * 3 + 2] = d[2]
                uv[vi * 2] = uvs[k * 2]; uv[vi * 2 + 1] = uvs[k * 2 + 1]
            }
            val b = f * 4
            idx[f * 6] = b; idx[f * 6 + 1] = b + 1; idx[f * 6 + 2] = b + 2
            idx[f * 6 + 3] = b; idx[f * 6 + 4] = b + 2; idx[f * 6 + 5] = b + 3
        }
        return MeshData(pos, nrm, uv, idx)
    }

    /**
     * Rewrite an existing tube's vertices for a new [path] with the SAME point count, sides and
     * closedness (same topology) — for traces that change every frame. Allocation-free; afterwards
     * tell the adapter (SpecDemo.meshEdited) so the GPU copy is refreshed.
     */
    fun tubeInto(md: MeshData, path: FloatArray, r: Float, sides: Int, closed: Boolean) {
        val t = tube(path, r, sides, closed)
        require(t.pos.size == md.pos.size) { "tubeInto: topology changed" }
        System.arraycopy(t.pos, 0, md.pos, 0, t.pos.size)
        System.arraycopy(t.nrm, 0, md.nrm, 0, t.nrm.size)
    }
}
