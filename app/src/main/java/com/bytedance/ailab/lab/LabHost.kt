// LabHost — one volume, many demos. Owns the persistent "Stage" entity inside the SpatialView and
// swaps demo roots in and out of it. Each Demo is a self-contained object (physics in a pure-Kotlin
// core with JVM tests, entities built under one root, one SpatialUI control panel).

package com.bytedance.ailab.lab

import android.util.Log
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.bytedance.ailab.lab.spec.LabelSpec
import com.pico.spatial.core.ecs.TransformComponent
import com.pico.spatial.core.math.Vector3
import com.pico.spatial.core.ecs.Entity
import com.pico.spatial.core.ecs.Scene
import java.io.File

class DemoContext(val cacheDir: File)

/**
 * A gaze+pinch target ALSO offered as a panel button. The button calls LabHost.onTap(node) — the very
 * same code path as pinching the object — so the panel and the object can never disagree. [isOn]
 * (optional) lights the button up while that object's state is "on" (lamp burning, slit covered).
 */
data class TapTarget(val node: String, val zh: String, val en: String, val isOn: (() -> Boolean)? = null)

interface Demo {
    /** Stable id (used by the debug broadcast and the bake cache), e.g. "young". */
    val id: String
    /** Menu title, Chinese first: "杨氏双缝干涉". */
    val title: String
    /** Who / when: "Young · 1801". */
    val credit: String

    /** Build every entity under ONE new root and return it (main thread; bake on Dispatchers.Default). */
    suspend fun build(ctx: DemoContext): Entity
    /** Per frame, main thread. dt is real seconds (clamp it yourself if you integrate). */
    fun tick(dt: Float)
    /** Which entity names are gaze+pinch targets for this demo. */
    fun isTarget(name: String): Boolean = false
    fun onTap(name: String) {}
    /** The demo's control panel (SpatialUI only, inside PicoTheme). */
    @Composable fun Panel()
    /** After the root has been destroyed: drop entity references so a rebuild starts clean. */
    fun dispose() {}
    /** Optional one-shot self test with access to the Scene (raycasts etc.). */
    fun selfCheck(scene: Scene) {}
    /** Floating labels, volume coordinates; the host shows the first [LabHost.LABEL_SLOTS]. */
    val labels: List<LabelSpec> get() = emptyList()
    /** Objects the user can pinch, mirrored as panel buttons ("✋ 也可以直接捏合"). */
    val tapTargets: List<TapTarget> get() = emptyList()

    // ---- pinch-and-drag (hand motion while pinched; deltas already in VOLUME metres) ----
    fun isDragTarget(name: String): Boolean = false
    fun onDragStart(name: String) {}
    fun onDrag(name: String, dx: Float, dy: Float, dz: Float) {}
    fun onDragEnd(name: String) {}

    // ---- tap feedback: what the user sees right after a pinch / panel tap ----
    /** One line saying what the tap on [name] just did (state-dependent); null = use the TapTarget label. */
    fun tapMessage(name: String): String? = null
    /** Volume-space point to show that message at (just above the object); null = no floating note. */
    fun anchorOf(name: String): FloatArray? = null
    /** Flash the object's pinch halo (confirms the pinch registered). */
    fun flash(name: String) {}

    /** The guided track (导学) for this demo, or null. Build it ONCE (`by lazy`): steps capture demo state. */
    val lesson: Lesson? get() = null
}

object LabHost {
    private const val TAG = "LabHost"

    lateinit var demos: List<Demo>
        private set

    /** Compose-observed state. */
    var currentId by mutableStateOf("")
        private set
    var loading by mutableStateOf(false)
        private set
    /** Set by the debug broadcast; LabVolume observes it and calls [switchTo]. */
    var requested by mutableStateOf<String?>(null)

    /** Catalogue open: the scene is hidden so the cards have the user's whole attention. */
    var catalogOpen by mutableStateOf(false)
        private set
    fun showCatalog(open: Boolean) {
        catalogOpen = open
        stage?.enabled = !open
        // hide the control panel through its ENTITY, never by emptying its content: an attachment
        // whose content went to zero size did not come back (other demos showed no panel at all)
        controlsEntity?.enabled = !open
        lessonEntity?.enabled = !open && LessonRunner.phase != LessonRunner.Phase.NONE
        if (open) hideLabels()
    }
    var controlsEntity: Entity? = null
    var lessonEntity: Entity? = null

    // ---- drag routing ----
    private var dragName: String? = null
    /** A drag happened since the last pinch-down: that pinch is a drag (comet throw), not a tap. */
    var draggedSincePinch = false
        private set
    fun pinchBegan() { draggedSincePinch = false }
    fun isDragTarget(name: String) = current?.isDragTarget(name) == true
    fun onDrag(name: String?, dx: Float, dy: Float, dz: Float) {
        draggedSincePinch = true
        val d = current ?: return
        if (name != null && dragName == null) { dragName = name; d.onDragStart(name) }
        dragName?.let { d.onDrag(it, dx, dy, dz) }
    }
    fun onDragEnd() { dragName?.let { current?.onDragEnd(it) }; dragName = null }

    /** Previous / next demo in catalogue order (wraps). */
    fun neighbour(step: Int): String? {
        if (demos.isEmpty()) return null
        val i = demos.indexOfFirst { it.id == currentId }.coerceAtLeast(0)
        return demos[((i + step) % demos.size + demos.size) % demos.size].id
    }

    private var stage: Entity? = null
    private var ctx: DemoContext? = null
    private var current: Demo? = null
    private var currentRoot: Entity? = null
    private var checked = false
    private var sinceBuild = 0f
    private var workWorst = 0f

    const val LABEL_SLOTS = 6
    /** Compose-observed label texts, one per attachment slot. */
    val labelText = mutableStateListOf(*Array(LABEL_SLOTS) { "" })
    private var labelEntities: List<Entity?> = emptyList()
    fun attachLabels(list: List<Entity?>) { labelEntities = list; hideLabels() }
    private fun hideLabels() { labelEntities.forEach { it?.enabled = false } }

    fun register(list: List<Demo>) { demos = list }

    fun attach(stageEntity: Entity, context: DemoContext) { stage = stageEntity; ctx = context }

    val demo: Demo? get() = current
    /** The persistent entity all demo roots hang under (= volume space). */
    val stageEntity: Entity? get() = stage

    /** Main thread. Tears down the current demo, builds and shows the next one. */
    suspend fun switchTo(id: String) {
        val s = stage ?: return
        val c = ctx ?: return
        val next = demos.firstOrNull { it.id == id } ?: return
        if (loading || (current === next && currentRoot != null)) return
        loading = true
        try {
            hideLabels(); toastLeft = 0f; toastEntity?.enabled = false
            currentRoot?.let { r -> s.removeChild(r); r.destroy(true) }
            current?.dispose()
            current = null; currentRoot = null
            val t0 = System.nanoTime()
            val root = next.build(c)
            s.addChild(root)
            current = next; currentRoot = root; checked = false; sinceBuild = 0f
            fpsN = 0; fpsT = 0f; fpsWorst = 0f; workWorst = 0f
            currentId = next.id
            LessonRunner.start(next.lesson) { node -> current?.flash(node) }
            lessonEntity?.enabled = !catalogOpen && LessonRunner.phase != LessonRunner.Phase.NONE
            Log.i(TAG, "demo ${next.id} up in ${"%.0f".format((System.nanoTime() - t0) / 1e6)} ms")
        } catch (e: Exception) {
            Log.e(TAG, "demo $id failed to build", e)
        } finally {
            loading = false
        }
    }

    // frame-rate probe for every demo: engine tick rate and worst frame over 4 s windows
    private var fpsN = 0; private var fpsT = 0f; private var fpsWorst = 0f

    fun tick(dt: Float, scene: Scene) {
        val d = current ?: return
        if (loading) return
        tickToast(dt)
        LessonRunner.tick(dt)
        val t0 = System.nanoTime()
        d.tick(dt)
        val work = (System.nanoTime() - t0) / 1e6f
        fpsN++; fpsT += dt; if (dt > fpsWorst) fpsWorst = dt
        if (work > workWorst) workWorst = work
        if (fpsT >= 4f) {
            Log.i(TAG, "FPS ${d.id} %.1f Hz, worst frame %.1f ms, worst demo update %.2f ms".format(fpsN / fpsT, fpsWorst * 1000, workWorst))
            fpsN = 0; fpsT = 0f; fpsWorst = 0f; workWorst = 0f
        }
        val ls = if (catalogOpen) emptyList() else d.labels
        for (i in labelEntities.indices) {
            val e = labelEntities[i] ?: continue
            val l = ls.getOrNull(i)
            val on = l != null && l.visible && l.text.isNotEmpty()
            if (e.enabled != on) e.enabled = on
            if (l != null && on) {
                e.components[TransformComponent::class.java]?.position = Vector3(l.pos.x, l.pos.y, l.pos.z)
                if (labelText[i] != l.text) labelText[i] = l.text
            }
        }
        // self-check after 2 s: colliders created this frame are not in the physics scene yet
        if (!checked) {
            sinceBuild += dt
            if (sinceBuild > 2f) { checked = true; try { d.selfCheck(scene) } catch (e: Exception) { Log.w(TAG, "selfCheck", e) } }
        }
    }

    /** UI previews (JVM): make [d] the current demo without building any entity. */
    fun previewAs(d: Demo) { current = d; currentId = d.id }

    fun isTarget(name: String) = current?.isTarget(name) == true
    /** Bumped on every pinch / panel target, so the panel re-reads the demo's state immediately. */
    var uiEpoch by mutableStateOf(0)
        private set

    /**
     * A pinch on an object ([fromPanel] = false) or its panel twin. Besides running the demo's action,
     * every tap gets VISIBLE confirmation where the user is looking: the object's halo flashes and a
     * note ("✓ 铜球 A → 换释放角 30°") appears just above it for [TOAST_S] s. Without it a pinch that
     * changed something subtle was indistinguishable from a pinch that missed.
     */
    fun onTap(name: String?, fromPanel: Boolean = false) {
        val d = current ?: return
        if (name == null) return
        d.onTap(name)
        uiEpoch++
        d.flash(name)
        val msg = d.tapMessage(name) ?: d.tapTargets.firstOrNull { it.node == name }?.let { "✓ " + tr(it.zh, it.en) }
        val at = d.anchorOf(name)
        Log.i("LabTap", "${d.id} tap $name via ${if (fromPanel) "panel" else "pinch"} -> ${msg ?: "-"}")
        if (msg != null && at != null) {
            toastText = msg; toastLeft = TOAST_S
            toastEntity?.let { e ->
                e.components[TransformComponent::class.java]?.position = Vector3(at[0], at[1], at[2])
                e.enabled = !catalogOpen
            }
        }
    }

    const val TOAST_S = 2.6f
    var toastText by mutableStateOf("")
        private set
    private var toastLeft = 0f
    var toastEntity: Entity? = null
        set(v) { field = v; v?.enabled = false }
    private fun tickToast(dt: Float) {
        if (toastLeft <= 0f) return
        toastLeft -= dt
        if (toastLeft <= 0f || catalogOpen) { toastLeft = 0f; toastEntity?.enabled = false }
    }
}
