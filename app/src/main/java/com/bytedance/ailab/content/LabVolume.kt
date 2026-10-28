package com.bytedance.ailab.content

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.key
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.bytedance.ailab.lab.DemoContext
import com.bytedance.ailab.lab.LabHost
import com.bytedance.ailab.lab.LabAssets
import com.bytedance.ailab.lab.LabSettings
import com.bytedance.ailab.lab.Narrator
import androidx.compose.runtime.remember
import com.bytedance.ailab.lab.LabTickSystem
import com.bytedance.ailab.lab.group
import com.bytedance.ailab.lab.tr
import com.bytedance.ailab.lab.ui.LabPanel
import com.pico.spatial.core.ecs.TransformComponent
import com.pico.spatial.core.math.Vector3
import com.pico.spatial.ui.design.PicoTheme
import com.pico.spatial.ui.design.Text
import com.pico.spatial.ui.foundation.content.SpatialView
import com.pico.spatial.ui.foundation.dsl.registerSystem
import com.pico.spatial.ui.foundation.dsl.unregisterSystem
import com.pico.spatial.ui.foundation.gesture.TargetEntity
import com.pico.spatial.ui.foundation.gesture.detectSpatialDragGesture
import com.pico.spatial.ui.foundation.gesture.detectSpatialPointerEvent
import com.pico.spatial.ui.geometry.Offset3D
import android.util.Log
import com.pico.spatial.ui.foundation.material.backgroundMaterial
import com.pico.spatial.ui.platform.LengthUnit
import com.pico.spatial.ui.platform.LocalPhysicalLengthConverter
import com.pico.spatial.ui.platform.Material
import kotlinx.coroutines.launch

/** Volume layout (metres from the volume centre; volume = 2700 x 1500 x 2700 dp = 2.16 x 1.2 x 2.16 m). */
private val MENU_POS = Vector3(0f, 0.52f, 0.95f)
private val CATALOG_POS = Vector3(0f, 0.02f, 0.98f)
private val PANEL_POS = Vector3(0f, -0.36f, 0.92f)
/**
 * Where the control panel can sit (LabSettings.panelPos): front-bottom (wide), or on the left / right,
 * narrow and turned 30 deg toward the viewer so it reads straight on from the default eye point.
 */
private fun panelPose(i: Int): Pair<Vector3, com.pico.spatial.core.math.Quat> {
    val yaw = when (i) { com.bytedance.ailab.lab.LabSettings.PANEL_LEFT -> 30f; com.bytedance.ailab.lab.LabSettings.PANEL_RIGHT -> -30f; else -> 0f }
    val h = Math.toRadians(yaw / 2.0)
    val q = com.pico.spatial.core.math.Quat(0f, kotlin.math.sin(h).toFloat(), 0f, kotlin.math.cos(h).toFloat())
    val p = when (i) {
        com.bytedance.ailab.lab.LabSettings.PANEL_LEFT -> Vector3(-0.78f, -0.12f, 0.62f)
        com.bytedance.ailab.lab.LabSettings.PANEL_RIGHT -> Vector3(0.78f, -0.12f, 0.62f)
        else -> PANEL_POS
    }
    return p to q
}
/** Background card: upper right (upper left when the panel is on the right). */
private fun aboutPos(panel: Int) = if (panel == com.bytedance.ailab.lab.LabSettings.PANEL_RIGHT) Vector3(-0.66f, 0.26f, 0.95f) else Vector3(0.66f, 0.26f, 0.95f)
/** Guided-track card: upper left, beside the top bar (x -0.90 … -0.48 for its 520 dp). */
private val LESSON_POS = Vector3(-0.69f, 0.43f, 0.95f)
/** A pinch released within this long is a tap (the OS verdict's timeout is unknown and was too strict). */
private const val TAP_MAX_MS = 1500L

@Composable
fun LabVolume(firstDemo: String) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    DisposableEffect(Unit) {
        registerSystem<LabTickSystem>()
        onDispose { unregisterSystem<LabTickSystem>() }
    }
    // debug broadcast (adb) -> switch demo, for the automated screenshot review. Collected in ONE
    // long-lived coroutine: keying a LaunchedEffect on `requested` and clearing it inside cancelled
    // the very switchTo() it had started (LeftCompositionCancellationException on the headset).
    LaunchedEffect(Unit) {
        snapshotFlow { LabHost.requested }.collect { id ->
            if (id != null) { LabHost.switchTo(id); LabHost.requested = null }
        }
    }
    // layout choices (top bar): scene scale, panel pose, background card position — applied live
    LaunchedEffect(Unit) { snapshotFlow { LabSettings.scaleIdx }.collect { LabHost.applyScale() } }
    LaunchedEffect(Unit) {
        snapshotFlow { LabSettings.panelPos }.collect { i ->
            val (p, q) = panelPose(i)
            LabHost.controlsEntity?.components?.get(TransformComponent::class.java)?.apply { position = p; quaternion = q }
            LabHost.aboutEntity?.components?.get(TransformComponent::class.java)?.position = aboutPos(i)
        }
    }
    remember(context) { Narrator.init(context); 0 }
    // demos read shipped files (trained weights) through LabAssets: here, from the APK
    remember(context) { LabAssets.open = { path -> runCatching { context.assets.open(path).use { it.readBytes() } }.getOrNull() }; 0 }

    val density = LocalDensity.current
    val lengths = LocalPhysicalLengthConverter.current
    /** view px -> volume metres (the documented path: px -> dp -> LengthUnit.Meters). */
    fun pxToM(px: Float): Float = lengths.dpToLength(with(density) { px.toDp() }, LengthUnit.Meters)

    SpatialView(
        modifier = Modifier
            .pointerInput(Unit) {
                // TAP = our own, on the raw pinch events. The SDK's detectSpatialTapGesture (read from its
                // bytecode) re-hit-tests at RELEASE, requires the same entity at down and up, and asks the
                // OS for a TAP verdict with its own slop/timeout: a swinging bob or orbiting planet that
                // moved out from under the gaze, a glance away before letting go, or a slow pinch all
                // failed SILENTLY while the hover highlight had shown. Here the entity is LOCKED at the
                // pinch-down (detectSpatialPointerEvent keeps it per pointer) and any release within
                // TAP_MAX_MS is a tap — unless the pinch became a drag (comet throw).
                val downs = HashMap<Long, Triple<String, Long, Offset3D>>()
                detectSpatialPointerEvent(context, TargetEntity.any { e -> LabHost.isTarget(e.getName()) }) { infos ->
                    for (p in infos) {
                        val id = p.pointerId.value
                        if (p.isDownEvent()) {
                            val name = p.targetedEntity?.getName() ?: continue
                            downs[id] = Triple(name, p.uptimeMillis, p.position3D)
                            LabHost.pinchBegan()
                        } else if (p.isUpEvent()) {
                            val (name, t0, pos0) = downs.remove(id) ?: continue
                            val ms = p.uptimeMillis - t0
                            val movedPx = (p.position3D - pos0).getDistance()
                            val tap = ms <= TAP_MAX_MS && !LabHost.draggedSincePinch
                            Log.i("LabTap", "pinch $name ${ms} ms, moved ${"%.0f".format(movedPx)} px = ${"%.3f".format(pxToM(movedPx))} m, " +
                                "${p.kind} -> ${if (tap) "TAP" else if (LabHost.draggedSincePinch) "drag" else "too long"}")
                            if (tap) LabHost.onTap(name)
                        }
                    }
                    false                                    // never consume: the drag detector needs the events
                }
            }
            .pointerInput(Unit) {
                // pinch-and-drag (one recognizer per pointerInput, as the SDK requires). Hand motion
                // arrives as px deltas in view space (+Y down): converted to volume metres, Y flipped.
                detectSpatialDragGesture(
                    context,
                    TargetEntity.any { e -> LabHost.isDragTarget(e.getName()) },
                    onDragStart = { },
                    onDragEnd = { LabHost.onDragEnd() },
                    onDragCancel = { LabHost.onDragEnd() },
                ) { v ->
                    LabHost.onDrag(v.targetEntity?.getName(), pxToM(v.dragAmount.x), pxToM(-v.dragAmount.y), pxToM(v.dragAmount.z))
                }
            },
        initial = { content, attachments ->
            val stage = group("Stage")
            content.addEntity(stage)
            LabHost.attach(stage, DemoContext(context.cacheDir))
            attachments.entity(id = "menu")?.apply {
                components[TransformComponent::class.java]?.position = MENU_POS
                content.addEntity(this)
            }
            attachments.entity(id = "catalog")?.apply {
                components[TransformComponent::class.java]?.position = CATALOG_POS
                content.addEntity(this)
            }
            attachments.entity(id = "controls")?.apply {
                val (p, q) = panelPose(LabSettings.panelPos)
                components[TransformComponent::class.java]?.apply { position = p; quaternion = q }
                content.addEntity(this)
                LabHost.controlsEntity = this
            }
            attachments.entity(id = "about")?.apply {
                components[TransformComponent::class.java]?.position = aboutPos(LabSettings.panelPos)
                content.addEntity(this)
                LabHost.aboutEntity = this
            }
            attachments.entity(id = "lesson")?.apply {
                components[TransformComponent::class.java]?.position = LESSON_POS
                content.addEntity(this)
                LabHost.lessonEntity = this
                enabled = false
            }
            attachments.entity(id = "toast")?.apply {
                content.addEntity(this)
                LabHost.toastEntity = this
            }
            LabHost.attachLabels(List(LabHost.LABEL_SLOTS) { i ->
                attachments.entity(id = "label$i")?.also { content.addEntity(it) }
            })
            LabHost.switchTo(firstDemo)
        },
        attachments = {
            AttachmentPanel(id = "menu") { LabTopBar { id -> scope.launch { LabHost.switchTo(id) } } }
            AttachmentPanel(id = "catalog") { LabCatalog { id -> scope.launch { LabHost.switchTo(id) } } }
            AttachmentPanel(id = "controls") {
                // never empty: a zero-size attachment did not come back on the headset
                key(LabHost.currentId, LabHost.uiEpoch) {
                    val d = LabHost.demo
                    if (d != null && LabHost.currentId.isNotEmpty()) d.Panel() else LabPanel(tr("加载中…", "Loading…")) {}
                }
            }
            for (i in 0 until LabHost.LABEL_SLOTS) AttachmentPanel(id = "label$i") { LabelChip(LabHost.labelText[i]) }
            AttachmentPanel(id = "toast") { ToastChip(LabHost.toastText) }
            AttachmentPanel(id = "lesson") { LessonCard() }
            AttachmentPanel(id = "about") { AboutCard() }
        },
    )
}

/** What a tap just did, shown above the tapped object for a few seconds (LabHost.onTap). */
@Composable
private fun ToastChip(text: String) {
    Text(
        text = text.ifEmpty { " " },
        color = Color.White,
        style = PicoTheme.typography.bodyLarge.copy(fontSize = 24.sp, lineHeight = 30.sp, fontWeight = FontWeight.SemiBold),
        modifier = Modifier
            .clip(RoundedCornerShape(22.dp))
            .background(Color(0xE63B82F6))
            .padding(horizontal = 18.dp, vertical = 8.dp),
    )
}

@Composable
private fun LabelChip(text: String) {
    Text(
        text = text,
        color = PicoTheme.colorScheme.labelPrimary,
        style = PicoTheme.typography.bodyLarge.copy(fontSize = 20.sp, lineHeight = 26.sp),
        modifier = Modifier
            .clip(RoundedCornerShape(18.dp))
            .backgroundMaterial(true, Material.Regular)
            .padding(horizontal = 14.dp, vertical = 6.dp),
    )
}
