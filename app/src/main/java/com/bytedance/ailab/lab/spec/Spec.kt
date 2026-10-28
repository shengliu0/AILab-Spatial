// Spec — a tiny, pure-Kotlin scene graph that every lab demo describes itself in.
//
// WHY. The same description is consumed twice:
//   * on the headset, lab/SpecAdapter turns it into SpatialSDK entities and keeps them in sync;
//   * on the Mac (JVM unit test), lab/spec/SoftRender rasterises it to PNG for the automated
//     visual review (score_shots.py) — no headset, no emulator, no network.
// So a demo's look can be iterated and scored headlessly, and SDK specifics live in ONE adapter.
//
// RULES FOR DEMOS
//   * Build the tree once; per frame only mutate pos/rot/scale/visible, or call setMesh/setMat.
//   * Share MeshData / Image instances between nodes (the adapter caches by identity).
//   * Metres, right-handed, +Y up, +Z toward the viewer. Volume = 2.16 x 1.2 x 2.16 m, origin at
//     its centre; keep content within |x|,|z| < 0.9 and -0.25 < y < 0.45 (panels live below/above).

package com.bytedance.ailab.lab.spec

import com.bytedance.ailab.core.P3
import com.bytedance.ailab.lab.Q
import com.bytedance.ailab.render.Image
import com.bytedance.ailab.render.MeshData
import com.bytedance.ailab.render.Rgb

enum class Blend { OPAQUE, TRANSPARENT, ADD }

/** Materials, colours in sRGB (as an artist would pick them). */
sealed class Mat {
    /** Self-lit: emitters, beams, glows, UI-like overlays, baked-lighting surfaces. */
    data class Unlit(
        val color: Rgb = Rgb(1f, 1f, 1f),
        val alpha: Float = 1f,
        val tex: Image? = null,
        val blend: Blend = Blend.OPAQUE,
    ) : Mat()

    /** Lit by the room (IBL) + scene key light: lab hardware, liquids, glass. */
    data class Pbr(
        val color: Rgb,
        val metallic: Float = 0f,
        val roughness: Float = 0.5f,
        val alpha: Float = 1f,            // < 1 -> transparent (glass, water)
        val tex: Image? = null,
        val emissive: Rgb? = null,        // glow on top of lighting (hot metal, LEDs)
    ) : Mat()
}

/** A floating text label (rendered as a SpatialUI chip on the headset). */
data class LabelSpec(val text: String, val pos: P3, val visible: Boolean = true)

class Node(val name: String) {
    var mesh: MeshData? = null; private set
    var mat: Mat? = null; private set
    var pos = P3(0f, 0f, 0f); set(v) { field = v; tfVersion++ }
    var rot = Q.identity(); set(v) { field = v; tfVersion++ }
    var scale = P3(1f, 1f, 1f); set(v) { field = v; tfVersion++ }
    var visible = true; set(v) { if (field != v) { field = v; visVersion++ } }
    /** > 0: gaze+pinch target with a sphere collider of this radius (node-local units). */
    var tapRadius = 0f; set(v) { field = v; tapVersion++ }
    /** Keep local +Z pointing at the viewer (engine LookAt on device). `rot` is ignored when set.
     *  Use for view-dependent stand-ins: limb-darkening shells, glows, coronas, sprites. */
    var billboard = false; set(v) { field = v; tapVersion++ }

    val children = ArrayList<Node>()
    var parent: Node? = null; private set

    // change counters the adapter compares against
    var tfVersion = 0; private set
    var meshVersion = 0; private set
    var matVersion = 0; private set
    var visVersion = 0; private set
    var tapVersion = 0; private set

    fun setMesh(m: MeshData?) { mesh = m; meshVersion++ }
    fun setMat(m: Mat?) { if (m != mat) { mat = m; matVersion++ } }
    fun uniformScale(s: Float) { scale = P3(s, s, s) }

    fun add(child: Node): Node { child.parent = this; children += child; return child }

    /** Convenience: child with mesh + material. */
    fun add(name: String, mesh: MeshData?, mat: Mat?, pos: P3 = P3(0f, 0f, 0f)): Node =
        add(Node(name).also { it.setMesh(mesh); it.setMat(mat); it.pos = pos })

    fun walk(f: (Node) -> Unit) { f(this); for (c in children) c.walk(f) }
    fun find(n: String): Node? { if (name == n) return this; for (c in children) c.find(n)?.let { return it }; return null }
}

fun rgb(hex: Int) = Rgb(((hex shr 16) and 0xFF) / 255f, ((hex shr 8) and 0xFF) / 255f, (hex and 0xFF) / 255f)
