// Small mesh primitives the shared MeshGen does not have (kept in this package on purpose):
// a cone for arrow heads, a helix for the spring scale, and a tapered tube for the water jet.
// Same conventions as render/MeshGen: plain arrays, CCW outward winding, no SDK types.

package com.bytedance.ailab.render

import com.bytedance.ailab.render.MeshData
import com.bytedance.ailab.render.MeshGen
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

object BuoyMesh {

    /** Unit cone: base radius 1 at y = 0 (capped), apex at y = 1. */
    fun cone(seg: Int = 20): MeshData {
        val pos = ArrayList<Float>(); val nrm = ArrayList<Float>(); val uv = ArrayList<Float>(); val idx = ArrayList<Int>()
        fun v(x: Float, y: Float, z: Float, nx: Float, ny: Float, nz: Float): Int {
            pos += x; pos += y; pos += z; nrm += nx; nrm += ny; nrm += nz; uv += 0f; uv += 0f
            return pos.size / 3 - 1
        }
        val k = 1f / sqrt(2f)                     // slant normal for radius 1 / height 1
        for (s in 0 until seg) {
            val a0 = 2 * PI * s / seg; val a1 = 2 * PI * (s + 1) / seg; val am = (a0 + a1) / 2
            val b0 = v(cos(a0).toFloat(), 0f, sin(a0).toFloat(), cos(a0).toFloat() * k, k, sin(a0).toFloat() * k)
            val b1 = v(cos(a1).toFloat(), 0f, sin(a1).toFloat(), cos(a1).toFloat() * k, k, sin(a1).toFloat() * k)
            val ap = v(0f, 1f, 0f, cos(am).toFloat() * k, k, sin(am).toFloat() * k)
            idx += b0; idx += ap; idx += b1                       // outward CCW (phi increases toward +z)
            val c = v(0f, 0f, 0f, 0f, -1f, 0f)
            val d0 = v(cos(a0).toFloat(), 0f, sin(a0).toFloat(), 0f, -1f, 0f)
            val d1 = v(cos(a1).toFloat(), 0f, sin(a1).toFloat(), 0f, -1f, 0f)
            idx += c; idx += d0; idx += d1
        }
        return MeshData(pos.toFloatArray(), nrm.toFloatArray(), uv.toFloatArray(), idx.toIntArray())
    }

    /** Coil spring along -Y from y = 0 to y = -len: [turns] turns of radius [r], wire radius [wire]. */
    fun helix(r: Float, len: Float, turns: Int, wire: Float, perTurn: Int = 16): MeshData {
        val n = turns * perTurn + 1
        val p = FloatArray(n * 3)
        for (i in 0 until n) {
            val a = 2 * PI * i / perTurn
            p[i * 3] = (r * cos(a)).toFloat(); p[i * 3 + 1] = -len * i / (n - 1); p[i * 3 + 2] = (r * sin(a)).toFloat()
        }
        return MeshGen.tube(p, wire, 6, false)
    }

    /**
     * Tube whose radius varies along the path (one radius per point). Same vertex layout as
     * MeshGen.tube, so [taperInto] can rewrite it in place every frame.
     */
    fun taper(path: FloatArray, radii: FloatArray, sides: Int): MeshData {
        val md = MeshGen.tube(path, 1f, sides, false)
        applyRadii(md, path, radii, sides)
        return md
    }

    fun taperInto(md: MeshData, path: FloatArray, radii: FloatArray, sides: Int) {
        MeshGen.tubeInto(md, path, 1f, sides, false)
        applyRadii(md, path, radii, sides)
    }

    /** a unit-radius tube's ring vertices sit at centre + normal: rescale each ring by its radius */
    private fun applyRadii(md: MeshData, path: FloatArray, radii: FloatArray, sides: Int) {
        val n = path.size / 3
        for (i in 0 until n) for (s in 0..sides) {
            val k = (i * (sides + 1) + s) * 3
            for (c in 0 until 3) md.pos[k + c] = path[i * 3 + c] + md.nrm[k + c] * radii[i]
        }
    }
}
