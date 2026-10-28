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

    fun build(root: Node): Entity { sync(root); return ent[root]!! }

    /** Create entities for new nodes and push every changed property. Main thread. */
    fun sync(root: Node) {
        fun rec(n: Node, parent: Entity?) {
            var e = ent[n]
            if (e == null) {
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
                s.tf = n.tfVersion
            }
            if (s.mesh != n.meshVersion || s.mat != n.matVersion) {
                applyModel(n, e, meshChanged = s.mesh != n.meshVersion)
                s.mesh = n.meshVersion; s.mat = n.matVersion
            }
            if (s.vis != n.visVersion) { e.enabled = n.visible; s.vis = n.visVersion }
            if (s.tap != n.tapVersion) {
                if (n.tapRadius > 0f) makeTappable(e, n.tapRadius)
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
        flushMeshes()
    }

    private fun applyModel(n: Node, e: Entity, meshChanged: Boolean) {
        val md = n.mesh; val m = n.mat
        if (md == null || m == null) { e.components.remove(ModelComponent::class.java); mats.remove(n); return }
        val mr = meshCache.getOrPut(md) {
            com.pico.spatial.core.ecs.resource.MeshResource.createWithMeshModel(toModel(md), boundsFor(md), "m${meshSeq++}")
        }
        val kind = kindOf(m)
        val existing = mats[n]
        val mat: Material = if (existing != null && matKind[n] == kind) { update(existing, m); existing }
                            else create(m).also { mats[n] = it; matKind[n] = kind }
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
    fun refreshMesh(md: MeshData) { if (meshCache.containsKey(md)) dirty.add(md) }

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

    private fun texOf(img: Image?) = img?.let { texCache.getOrPut(it) { tex(it) } }

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
