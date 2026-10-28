// SpecAdapter — Spec scene graph -> SpatialSDK entities, kept in sync each frame by comparing the
// nodes' change counters. Every SDK call a lab demo makes goes through here (and Gfx), so a device
// quirk is fixed once for all demos.

package com.bytedance.ailab.lab

import com.bytedance.ailab.lab.spec.Blend
import com.bytedance.ailab.lab.spec.Mat
import com.bytedance.ailab.lab.spec.Node
import com.bytedance.ailab.render.Image
import com.bytedance.ailab.render.MeshData
import com.pico.spatial.core.ecs.Entity
import com.pico.spatial.core.ecs.ModelComponent
import com.pico.spatial.core.ecs.resource.BlendingMode
import com.pico.spatial.core.ecs.resource.Material
import com.pico.spatial.core.ecs.resource.MeshResource
import com.pico.spatial.core.ecs.resource.PhysicallyBasedMaterial
import com.pico.spatial.core.ecs.resource.TextureResource
import com.pico.spatial.core.ecs.resource.UnlitMaterial
import android.util.Log
import java.util.IdentityHashMap

class SpecAdapter {
    private class Seen(var tf: Int = -1, var mesh: Int = -1, var mat: Int = -1, var vis: Int = -1, var tap: Int = -1)

    private val ent = IdentityHashMap<Node, Entity>()
    private val seen = IdentityHashMap<Node, Seen>()
    private val mats = IdentityHashMap<Node, Material>()
    private val matKind = IdentityHashMap<Node, String>()
    private val meshCache = IdentityHashMap<MeshData, MeshResource>()
    private val texCache = IdentityHashMap<Image, TextureResource>()
    private var meshSeq = 0

    /**
     * Entities for the whole tree, but only SMALL meshes are uploaded now: creating a mesh costs ~17 us per
     * vertex on the Swan (solar 54k vertices = 0.9 s of frozen main thread). Bigger meshes are queued and
     * uploaded by [sync] over the next frames within [STREAM_BUDGET_NS], smallest first — the scene and its
     * panel appear at once and the heavy geometry fades in over ~1 s instead of freezing the volume.
     */
    fun build(root: Node): Entity { streaming = true; sync(root); return ent[root]!! }

    private var streaming = false
    private val pending = ArrayList<Node>()
    /** Meshes still waiting to be uploaded (0 = scene complete). */
    val pendingCount: Int get() = pending.size

    private fun flushPending() {
        if (pending.isEmpty()) return
        val t0 = System.nanoTime()
        pending.sortByDescending { it.mesh?.vertexCount ?: 0 }          // smallest at the end: pop from there
        var first = true
        while (pending.isNotEmpty()) {
            if (!first && System.nanoTime() - t0 > STREAM_BUDGET_NS) break
            val n = pending.removeAt(pending.size - 1); first = false
            val e = ent[n] ?: continue
            applyModel(n, e, meshChanged = true, defer = false)
        }
        if (pending.isEmpty()) Log.i("SpecAdapter", "streamed in: ${takeStats()}")
    }

    /** Create entities for new nodes and push every changed property. Main thread. */
    // per-frame sync statistics: transform / visibility writes cross into native code (device log every 4 s)
    private var syTf = 0; private var syVis = 0; private var syFrames = 0; private var syWorstNs = 0L; private var syWorstTf = 0
    private var syTfTotal = 0L; private var syT0 = 0L

    /** Entities still to be created (a demo assembles over several frames; see [CREATE_BUDGET_NS]). */
    private var assembling = true
    val complete: Boolean get() = !assembling && pending.isEmpty()

    fun sync(root: Node) {
        val sy0 = System.nanoTime(); syTf = 0; syVis = 0
        var skipped = false
        fun rec(n: Node, parent: Entity?) {
            var e = ent[n]
            // a hidden subtree that already exists: nothing it does is visible, so skip its transform/mesh writes
            // (pooled particles keep moving while hidden); everything changed is written when it shows again
            if (e != null && !n.visible) {
                val s0 = seen[n]!!
                if (s0.vis != n.visVersion) { e.enabled = false; s0.vis = n.visVersion; syVis++ }
                return
            }
            if (e == null) {
                // creating an entity costs ~1.3 ms on the Swan (a 500-node lab = 0.7 s): only within the budget,
                // the rest next frame (rec revisits uncreated nodes every sync)
                if (assembling && parent != null && System.nanoTime() - sy0 > CREATE_BUDGET_NS) { skipped = true; return }
                e = Entity().apply { setName(safeName(n.name)) }
                ent[n] = e; seen[n] = Seen()
                parent?.addChild(e)
            }
            val s = seen[n]!!
            if (s.tf != n.tfVersion && !finite(n)) {
                if (warned.add(n)) Log.w("SpecAdapter", "non-finite transform on '${n.name}' — skipped (fix the demo)")
                s.tf = n.tfVersion
            }
            if (s.tf != n.tfVersion) {
                val t = tf(e)
                t.position = com.pico.spatial.core.math.Vector3(n.pos.x, n.pos.y, n.pos.z)
                t.quaternion = com.pico.spatial.core.math.Quat(n.rot.x, n.rot.y, n.rot.z, n.rot.w)
                t.scaleVector = com.pico.spatial.core.math.Vector3(n.scale.x, n.scale.y, n.scale.z)
                s.tf = n.tfVersion; syTf++
            }
            if (s.mesh != n.meshVersion || s.mat != n.matVersion) {
                applyModel(n, e, meshChanged = s.mesh != n.meshVersion)
                s.mesh = n.meshVersion; s.mat = n.matVersion
            }
            if (s.vis != n.visVersion) { e.enabled = n.visible; s.vis = n.visVersion; syVis++ }
            if (s.tap != n.tapVersion) {
                if (n.tapRadius > 0f) makeTappable(e, maxOf(n.tapRadius, MIN_TAP_M / worldScale(n)))
                if (n.billboard && !e.components.has(com.pico.spatial.core.ecs.LookAtComponent::class.java))
                    e.components.set(com.pico.spatial.core.ecs.LookAtComponent().apply {
                        lookAtForwardDirection = com.pico.spatial.core.ecs.LookAtForwardDirection.POSITIVE_Z
                        setViewerAsTarget()
                    })
                s.tap = n.tapVersion
            }
            for (c in n.children) rec(c, e)
        }
        rec(root, null)
        if (assembling && !skipped) { assembling = false; Log.i("SpecAdapter", "entities complete") }
        val buildPass = streaming
        streaming = false
        if (!buildPass) flushPending()                         // the build frame itself only creates what is cheap
        flushMeshes()
        val dt = System.nanoTime() - sy0
        syFrames++; syTfTotal += syTf
        if (dt > syWorstNs) { syWorstNs = dt; syWorstTf = syTf }
        if (syT0 == 0L) syT0 = sy0
        if (sy0 - syT0 > 4_000_000_000L) {
            Log.i("SpecSync", "%d frames: %.0f transform writes/frame avg, worst sync %.1f ms (%d transform + %d visibility writes that frame)".format(
                syFrames, syTfTotal.toDouble() / syFrames, syWorstNs / 1e6, syWorstTf, syVis))
            syFrames = 0; syTfTotal = 0; syWorstNs = 0; syT0 = sy0
        }
    }

    /**
     * TEXT QUADS READ RIGHT FROM BOTH SIDES. Overlays (TRANSPARENT / ADD) are double-sided, so a text decal
     * seen from behind — a billboard the SDK turned the other way, a board the user walked around — showed
     * its text MIRRORED (users saw it in the chemistry and algebra labs). A textured overlay quad is drawn as
     * a two-faced twin instead: the original front face plus a back face with u mirrored, each back-face
     * culled, so whichever side faces the viewer reads correctly.
     */
    private val twins = IdentityHashMap<MeshData, MeshData>()
    private fun isTextQuad(md: MeshData, m: Mat) = md.vertexCount == 4 && md.idx.size == 6 && m is Mat.Unlit && m.tex != null && m.blend != Blend.OPAQUE
    private fun twinOf(md: MeshData): MeshData = twins.getOrPut(md) { MeshData(FloatArray(24), FloatArray(24), FloatArray(16), IntArray(12)) }.also { fillTwin(md, it) }
    private fun fillTwin(md: MeshData, t: MeshData) {
        for (k in 0 until 4) for (c in 0 until 3) {
            t.pos[k * 3 + c] = md.pos[k * 3 + c]; t.pos[(k + 4) * 3 + c] = md.pos[k * 3 + c]
            t.nrm[k * 3 + c] = md.nrm[k * 3 + c]; t.nrm[(k + 4) * 3 + c] = -md.nrm[k * 3 + c]
        }
        for (k in 0 until 4) {
            t.uv[k * 2] = md.uv[k * 2]; t.uv[k * 2 + 1] = md.uv[k * 2 + 1]
            t.uv[(k + 4) * 2] = 1f - md.uv[k * 2]; t.uv[(k + 4) * 2 + 1] = md.uv[k * 2 + 1]
        }
        for (i in 0 until 6) t.idx[i] = md.idx[i]
        for (tri in 0 until 2) {                                   // back face: same triangles, reversed winding
            t.idx[6 + tri * 3] = md.idx[tri * 3] + 4; t.idx[6 + tri * 3 + 1] = md.idx[tri * 3 + 2] + 4; t.idx[6 + tri * 3 + 2] = md.idx[tri * 3 + 1] + 4
        }
    }

    private fun applyModel(n: Node, e: Entity, meshChanged: Boolean, defer: Boolean = streaming || assembling) {
        val m = n.mat
        val twin = n.mesh != null && m != null && isTextQuad(n.mesh!!, m)
        val md = n.mesh?.let { if (twin) twinOf(it) else it }
        if (md == null || m == null) { e.components.remove(ModelComponent::class.java); mats.remove(n); pending.remove(n); return }
        // stream: big meshes and not-yet-created textures (each TextureResource is a synchronous GPU upload)
        val newTex = when (m) { is Mat.Unlit -> m.tex; is Mat.Pbr -> m.tex }?.let { !texCache.containsKey(it) } ?: false
        if (defer && ((md.vertexCount > STREAM_MIN_VERTS && !meshCache.containsKey(md)) || newTex)) { if (n !in pending) pending += n; return }
        if (!defer) pending.remove(n)
        val mr = meshCache[md] ?: run {
            val t0 = System.nanoTime()
            com.pico.spatial.core.ecs.resource.MeshResource.createWithMeshModel(toModel(md), boundsFor(md), "m${meshSeq++}").also {
                meshCache[md] = it; statMeshN++; statMeshNs += System.nanoTime() - t0
            }
        }
        val kind = kindOf(m) + if (twin) "T" else ""
        val existing = mats[n]
        val mat: Material = if (existing != null && matKind[n] == kind) { update(existing, m); existing }
                            else create(m).also { mats[n] = it; matKind[n] = kind
                                // each face of a two-faced text quad shows only to its own side
                                if (twin) (it as UnlitMaterial).setCullingMode(com.pico.spatial.core.ecs.resource.MaterialCullingMode.BACK) }
        val mc = e.components[ModelComponent::class.java]
        if (mc == null || existing !== mat) e.components.set(ModelComponent(mr, mat))
        else if (meshChanged) mc.mesh = mr
    }

    /**
     * A mesh whose MeshData was mutated in place (same identity) needs re-uploading. NOT done here:
     * `replaceWithMeshModel` only takes boxed per-vertex Lists that the SDK converts again before the
     * native call — measured on the Swan at ~20 us per vertex, so the pendulum's three 960-vertex trace
     * tubes cost ~60 ms per frame (29 Hz), Young 23 Hz, prism 63 Hz. Edits are COALESCED (only the
     * latest state of each mesh is uploaded) and uploaded in [sync] within [UPLOAD_BUDGET_NS] per frame,
     * round-robin, at least one per frame: a heavy demo refreshes its geometry less often instead of
     * dropping frames.
     */
    fun refreshMesh(md: MeshData) {
        twins[md]?.let { t -> fillTwin(md, t); if (meshCache.containsKey(t)) dirty.add(t) }
        if (meshCache.containsKey(md)) dirty.add(md)
    }

    private val dirty = LinkedHashSet<MeshData>()          // insertion order = round-robin fairness
    private var upN = 0; private var upVerts = 0L; private var upModelNs = 0L; private var upReplaceNs = 0L
    private var upWindowStart = 0L; private var upBacklogMax = 0

    private fun flushMeshes() {
        if (dirty.isEmpty()) return
        val t0 = System.nanoTime()
        upBacklogMax = maxOf(upBacklogMax, dirty.size)
        val it = dirty.iterator()
        var first = true
        while (it.hasNext()) {
            if (!first && System.nanoTime() - t0 > UPLOAD_BUDGET_NS) break
            val md = it.next(); it.remove(); first = false
            val mr = meshCache[md] ?: continue
            val a = System.nanoTime()
            val model = toModel(md)
            val b = System.nanoTime()
            try { mr.replaceWithMeshModel(model, boundsFor(md)) } catch (e: Exception) { Log.w("SpecAdapter", "mesh upload failed", e) }
            val c = System.nanoTime()
            upN++; upVerts += md.vertexCount; upModelNs += b - a; upReplaceNs += c - b
        }
        if (upWindowStart == 0L) upWindowStart = t0
        if (t0 - upWindowStart > 4_000_000_000L && upN > 0) {
            Log.i("MeshUpload", "%d uploads in 4 s, %d verts each (avg): toModel %.2f ms + replace %.2f ms per upload = %.1f us/vert; backlog max %d".format(
                upN, upVerts / upN, upModelNs / 1e6 / upN, upReplaceNs / 1e6 / upN, (upModelNs + upReplaceNs) / 1e3 / upVerts, upBacklogMax))
            upN = 0; upVerts = 0; upModelNs = 0; upReplaceNs = 0; upWindowStart = t0; upBacklogMax = 0
        }
    }

    private val warned = java.util.Collections.newSetFromMap(IdentityHashMap<Node, Boolean>())

    companion object {
        /** Smallest pinch sphere in WORLD metres. Eye tracking is good to ~1 deg = 4.4 cm at 2.5 m; 4.5 cm
         *  spheres made users aim twice, so 5.5 cm (~1.3 deg). Neighbours may overlap: the ray then
         *  picks the sphere whose surface it meets first, i.e. roughly the nearest centre. */
        const val MIN_TAP_M = 0.055f
        /** Initial build uploads meshes up to this many vertices at once; bigger ones stream in. */
        const val STREAM_MIN_VERTS = 0                          // device: ~8 ms per mesh whatever its size
        /** Entity creation allowed per frame while a demo assembles. */
        const val CREATE_BUDGET_NS = 5_000_000L
        /** Streamed mesh creation allowed per frame. */
        const val STREAM_BUDGET_NS = 4_000_000L

        private fun worldScale(n: Node): Float {
            var s = 1f; var c: Node? = n
            while (c != null) { s *= maxOf(kotlin.math.abs(c.scale.x), kotlin.math.abs(c.scale.y), kotlin.math.abs(c.scale.z)); c = c.parent }
            return if (s > 1e-6f) s else 1f
        }

        /** Mesh re-upload time allowed per frame (90 Hz frame = 11.1 ms). */
        const val UPLOAD_BUDGET_NS = 3_000_000L
        private val BAD = Regex("[^A-Za-z0-9_]")

        /** Entity.setName throws "Invalid name!" on anything but [A-Za-z0-9_] (found: "FinPost-0.33"). */
        fun safeName(n: String): String {
            val s = BAD.replace(n, "_")
            return if (s.isEmpty() || s[0].isDigit()) "n$s" else s
        }

        private fun finite(n: Node) = n.pos.x.isFinite() && n.pos.y.isFinite() && n.pos.z.isFinite() &&
            n.rot.x.isFinite() && n.rot.y.isFinite() && n.rot.z.isFinite() && n.rot.w.isFinite() &&
            n.scale.x.isFinite() && n.scale.y.isFinite() && n.scale.z.isFinite()

        /**
         * Explicit bounds for every mesh. createWithMeshModel throws "Empty bounding box" for a
         * zero-extent mesh — and demos legitimately park unused beams/spots as a collapsed point.
         * A minimum half-extent of 1 mm keeps those valid; later in-place edits recompute it.
         */
        fun boundsFor(md: MeshData): com.pico.spatial.core.ecs.BoundingBox {
            var lx = Float.MAX_VALUE; var ly = Float.MAX_VALUE; var lz = Float.MAX_VALUE
            var hx = -Float.MAX_VALUE; var hy = -Float.MAX_VALUE; var hz = -Float.MAX_VALUE
            for (k in 0 until md.vertexCount) {
                val x = md.pos[k * 3]; val y = md.pos[k * 3 + 1]; val z = md.pos[k * 3 + 2]
                if (!x.isFinite() || !y.isFinite() || !z.isFinite()) continue
                if (x < lx) lx = x; if (y < ly) ly = y; if (z < lz) lz = z
                if (x > hx) hx = x; if (y > hy) hy = y; if (z > hz) hz = z
            }
            if (lx > hx) { lx = 0f; ly = 0f; lz = 0f; hx = 0f; hy = 0f; hz = 0f }
            val e = 0.001f
            return com.pico.spatial.core.ecs.BoundingBox(
                com.pico.spatial.core.math.Vector3((lx + hx) / 2, (ly + hy) / 2, (lz + hz) / 2),
                com.pico.spatial.core.math.Vector3(maxOf((hx - lx) / 2, e), maxOf((hy - ly) / 2, e), maxOf((hz - lz) / 2, e)))
        }
    }

    private fun kindOf(m: Mat) = when (m) {
        is Mat.Unlit -> "U${m.blend}${m.tex != null}"
        is Mat.Pbr -> "P${m.alpha < 0.999f}${m.tex != null}"
    }

    /** Cached texture for [img]. Textures are made GLOBAL in [tex]: a local one is closed by the SDK when its
     *  material switches texture, and reusing the cached handle crashed setBaseColorTexture. */
    private fun texOf(img: Image?) = img?.let { texCache.getOrPut(it) { val t0 = System.nanoTime(); tex(it).also { statTexN++; statTexNs += System.nanoTime() - t0 } } }

    // build-cost statistics (device log: where does a demo's load time go?)
    private var statMeshN = 0; private var statMeshNs = 0L; private var statTexN = 0; private var statTexNs = 0L
    fun takeStats(): String {
        val s = "meshes %d (%.0f ms), textures %d (%.0f ms)".format(statMeshN, statMeshNs / 1e6, statTexN, statTexNs / 1e6)
        statMeshN = 0; statMeshNs = 0; statTexN = 0; statTexNs = 0; return s
    }

    private fun create(m: Mat): Material = when (m) {
        is Mat.Unlit -> unlit(
            when (m.blend) { Blend.OPAQUE -> BlendingMode.OPAQUE; Blend.TRANSPARENT -> BlendingMode.TRANSPARENT; Blend.ADD -> BlendingMode.ADD },
            texOf(m.tex), lin(m.color), m.alpha)
        is Mat.Pbr -> pbr(lin(m.color), m.metallic, m.roughness,
            if (m.alpha < 0.999f) BlendingMode.TRANSPARENT else BlendingMode.OPAQUE, m.alpha,
            texOf(m.tex), m.emissive?.let { lin(it) })
    }

    private fun update(mat: Material, m: Mat) {
        when (m) {
            is Mat.Unlit -> (mat as UnlitMaterial).apply {
                setBaseColor(lin(m.color)); if (m.blend != Blend.OPAQUE) setOpacity(m.alpha)
                texOf(m.tex)?.let { setBaseColorTexture(it) }
            }
            is Mat.Pbr -> (mat as PhysicallyBasedMaterial).apply {
                setBaseColor(lin(m.color)); setMetallic(m.metallic); setRoughness(m.roughness)
                if (m.alpha < 0.999f) setOpacity(m.alpha)
                setEmissiveColor(m.emissive?.let { lin(it) } ?: lin(0f, 0f, 0f))
                texOf(m.tex)?.let { setBaseColorTexture(it) }
            }
        }
    }
}
