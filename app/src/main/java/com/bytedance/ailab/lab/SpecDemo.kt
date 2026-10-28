package com.bytedance.ailab.lab

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.bytedance.ailab.lab.spec.LabelSpec
import com.bytedance.ailab.lab.spec.Node
import com.pico.spatial.core.ecs.Entity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Base class for lab demos. A subclass implements two PURE functions (no Android, no SDK):
 *   buildSpec(cacheDir) -> Node   build the scene graph (bakes allowed; runs on a worker thread)
 *   update(dt)                    advance the physics and mutate nodes / labels / readout
 * plus a Compose Panel(). The same subclass is driven by the JVM review test (SoftRender) and by
 * the headset (SpecAdapter), so what is scored on the Mac is what ships.
 */
abstract class SpecDemo : Demo {
    /** The live scene root (null until built). */
    var root: Node? = null
        private set
    private var adapter: SpecAdapter? = null

    /** 2-3 lines shown at the top of the control panel. Set it from update(); published at 10 Hz. */
    var readoutText = ""
    var readout by mutableStateOf("")
        private set
    private var readoutT = 0f

    /** Up to 6 floating labels (world = volume space). Rebuild/modify freely in update(). */
    override val labels: MutableList<LabelSpec> = ArrayList()

    abstract fun buildSpec(cacheDir: File): Node
    abstract fun update(dt: Float)

    /** Extra states the headless review should also render (name -> put the demo in that state). */
    open fun reviewVariants(): Map<String, () -> Unit> = emptyMap()

    /** For the JVM review test: build + simulate [seconds] at 60 Hz without any SDK. */
    fun runHeadless(cacheDir: File, seconds: Float): Node {
        val r = buildSpec(cacheDir); root = r
        halos.clear(); haloBase.clear(); flashNode = null; haloClock = 0f; haloAlphaShown = -1f
        val n = (seconds * 60).toInt()
        repeat(n) { update(1f / 60f) }
        ensureHalos(); pulseHalos(1f / 60f)
        return r
    }

    // ---------------------------------------------------------------- pinch affordance halos
    // Every node with a tapRadius gets a light-blue ring around its pinch zone, facing the viewer:
    // users could not tell which objects are interactive. Breathing for the first 15 s after the demo
    // opens, then a faint steady ring. Rings are added lazily (some demos set tapRadius in update()).
    private val halos = java.util.IdentityHashMap<Node, Node>()
    private var haloClock = 0f
    private var haloScan = 1f
    private var haloAlphaShown = -1f
    private val haloMesh by lazy {
        val n = 72; val rIn = 0.86f; val rOut = 1.0f
        val pos = FloatArray((n + 1) * 2 * 3); val nrm = FloatArray((n + 1) * 2 * 3); val uv = FloatArray((n + 1) * 2 * 2)
        for (k in 0..n) for (j in 0..1) {
            val a = 2.0 * Math.PI * k / n; val r = if (j == 0) rIn else rOut; val v = k * 2 + j
            pos[v * 3] = (r * kotlin.math.cos(a)).toFloat(); pos[v * 3 + 1] = (r * kotlin.math.sin(a)).toFloat()
            nrm[v * 3 + 2] = 1f; uv[v * 2] = j.toFloat(); uv[v * 2 + 1] = k.toFloat() / n
        }
        val idx = IntArray(n * 6); var t = 0
        for (k in 0 until n) { val a0 = k * 2; idx[t++] = a0; idx[t++] = a0 + 1; idx[t++] = a0 + 2; idx[t++] = a0 + 1; idx[t++] = a0 + 3; idx[t++] = a0 + 2 }
        com.bytedance.ailab.render.MeshData(pos, nrm, uv, idx)
    }
    private fun haloMat(a: Float) = com.bytedance.ailab.lab.spec.Mat.Unlit(
        com.bytedance.ailab.render.Rgb(0.45f, 0.80f, 1.0f), alpha = a, blend = com.bytedance.ailab.lab.spec.Blend.ADD)

    /** Add/resize halos for every pinchable node (call after the tree exists; cheap to repeat). */
    fun ensureHalos() {
        val r = root ?: return
        val want = ArrayList<Node>()
        r.walk { if (it.tapRadius > 0f && !it.name.startsWith("Halo_")) want += it }
        // a node that stopped being pinchable (tapRadius back to 0) must lose its ring too
        for ((n, h) in halos) { val on = n.tapRadius > 0f; if (h.visible != on) h.visible = on }
        for (n in want) {
            val h = halos.getOrPut(n) { n.add("Halo_${n.name}", haloMesh, haloMat(0.3f)).also { it.billboard = true } }
            // the ring hugs what you SEE (<= 1.6x the drawn radius) — the pinch zone itself stays big;
            // rings at the full gaze radius overlapped into a tangle around the inner planets
            val drawn = n.mesh?.let { meshRadius(it) } ?: 0f
            val want = if (drawn > 0f) minOf(n.tapRadius, drawn * 1.6f) else n.tapRadius
            haloBase[n] = want
            if (n !== flashNode && kotlin.math.abs(h.scale.x - want) > 1e-6f) h.uniformScale(want)
        }
    }

    private val meshR = java.util.IdentityHashMap<com.bytedance.ailab.render.MeshData, Float>()
    private fun meshRadius(m: com.bytedance.ailab.render.MeshData): Float = meshR.getOrPut(m) {
        var r = 0f
        for (k in 0 until m.vertexCount) { val x = m.pos[k * 3]; val y = m.pos[k * 3 + 1]; val z = m.pos[k * 3 + 2]; r = maxOf(r, x * x + y * y + z * z) }
        kotlin.math.sqrt(r)
    }

    private fun pulseHalos(dt: Float) {
        haloClock += dt
        val a = if (haloClock < 15f) 0.16f + 0.20f * (0.5f + 0.5f * kotlin.math.sin(haloClock * 2f * Math.PI.toFloat() / 1.8f))
                else 0.07f
        val q = (a * 40f).toInt() / 40f                     // quantised: no per-frame material churn
        if (q != haloAlphaShown) { haloAlphaShown = q; val m = haloMat(q); halos.values.forEach { if (it !== halos[flashNode]) it.setMat(m) } }
        // tap confirmation: the tapped object's ring "pings" — bright, expanding 1.0 -> 1.6x, fading
        val fn = flashNode ?: return
        val h = halos[fn] ?: run { flashNode = null; return }
        val base = haloBase[fn] ?: fn.tapRadius
        flashT -= dt
        if (flashT <= 0f) { flashNode = null; h.uniformScale(base); h.setMat(haloMat(haloAlphaShown.coerceAtLeast(0.07f))); return }
        val k = flashT / FLASH_S                             // 1 -> 0
        h.uniformScale(base * (1f + 0.6f * (1f - k)))
        h.setMat(haloMat(((0.2f + 0.75f * k) * 20f).toInt() / 20f))
    }

    private val haloBase = java.util.IdentityHashMap<Node, Float>()
    private var flashNode: Node? = null
    private var flashT = 0f
    private val FLASH_S = 0.5f

    override fun flash(name: String) {
        val n = root?.find(name) ?: return
        ensureHalos()
        flashNode?.let { o -> halos[o]?.let { it.uniformScale(haloBase[o] ?: o.tapRadius); it.setMat(haloMat(haloAlphaShown.coerceAtLeast(0.07f))) } }
        flashNode = n; flashT = FLASH_S
    }

    /** Just above the object's ring, in volume space (where the tap note appears). */
    override fun anchorOf(name: String): FloatArray? {
        val n = root?.find(name) ?: return null
        val a = com.bytedance.ailab.lab.spec.SoftRender.worldOf(n)
        val r = haloBase[n] ?: n.tapRadius
        val up = a.mul(0f, r, 0f)
        val wr = kotlin.math.sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2])
        return floatArrayOf(a.t[0], a.t[1] + wr + 0.06f, a.t[2] + 0.03f)
    }

    override suspend fun build(ctx: DemoContext): Entity {
        halos.clear(); haloBase.clear(); flashNode = null; haloClock = 0f; haloScan = 1f; haloAlphaShown = -1f
        val t0 = System.nanoTime()
        val r = withContext(Dispatchers.Default) { buildSpec(ctx.cacheDir) }
        val t1 = System.nanoTime()
        root = r
        val a = SpecAdapter(); adapter = a
        update(0f)
        val e = a.build(r)
        android.util.Log.i("LabHost", "BUILD $id: spec %.0f ms (background), first frame %.0f ms: ${a.takeStats()}, %d streaming".format(
            (t1 - t0) / 1e6, (System.nanoTime() - t1) / 1e6, a.pendingCount))
        return e
    }

    override fun tick(dt: Float) {
        val r = root ?: return
        update(dt.coerceIn(0f, 0.05f))
        haloScan += dt
        if (haloScan > 0.5f) { haloScan = 0f; ensureHalos() }
        pulseHalos(dt)
        adapter?.sync(r)
        readoutT += dt
        if (readoutT >= 0.1f) { readoutT = 0f; if (readout != readoutText) readout = readoutText }
    }

    /** Publish [readoutText] immediately (UI previews; on device [tick] does it at 10 Hz). */
    fun publishReadoutNow() { readout = readoutText }

    /** Push an in-place edit of a MeshData (same instance) to the GPU. */
    protected fun meshEdited(md: com.bytedance.ailab.render.MeshData) { adapter?.refreshMesh(md) }

    override fun dispose() { root = null; adapter = null; labels.clear() }

    /**
     * TAP-TARGET SELF-CHECK (device only, once, 2 s after build). For every gaze+pinch target, cast
     * a ray from 1.5 m in front of its VISIBLE centre straight back (stage = volume space) and log
     * whether the engine hits that target. A miss means the collider is not where the object is
     * drawn — e.g. a parent rotation/scale the physics did not apply — which is exactly what a user
     * experiences as "I have to look above it to select it".
     */
    override fun selfCheck(scene: com.pico.spatial.core.ecs.Scene) {
        val r = root ?: return
        val stage = LabHost.stageEntity ?: return
        val targets = ArrayList<Node>()
        fun visibleChain(n: Node): Boolean { var p: Node? = n; while (p != null) { if (!p.visible) return false; p = p.parent }; return true }
        r.walk { if (it.tapRadius > 0f && visibleChain(it)) targets += it }
        if (targets.isEmpty()) return
        val group = com.pico.spatial.core.ecs.simulation.CollisionGroup(com.pico.spatial.core.ecs.simulation.CollisionGroup.COLLISION_GROUP_ALL)
        var ok = 0
        val sb = StringBuilder()
        for (t in targets) {
            val w = com.bytedance.ailab.lab.spec.SoftRender.worldOf(t).t
            val hits = try {
                scene.rayCast(com.pico.spatial.core.math.Vector3(w[0], w[1], w[2] + 1.5f), com.pico.spatial.core.math.Vector3(0f, 0f, -1f),
                    3f, com.pico.spatial.core.ecs.simulation.CollisionCastHitMode.ALL, group, stage).results.map { it.entity.getName() }
            } catch (e: Exception) { listOf("threw:${e.javaClass.simpleName}") }
            val want = SpecAdapter.safeName(t.name)
            if (want in hits) ok++ else sb.append(" MISS ").append(t.name).append("@(%.2f,%.2f,%.2f)->".format(w[0], w[1], w[2])).append(hits.take(3))
        }
        android.util.Log.i("TapCheck", "$id: $ok/${targets.size} gaze targets hit at their visible centre.$sb")
    }
}
