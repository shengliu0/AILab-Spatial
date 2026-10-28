// GazeUi — drop-in, gaze-sized replacements for the SpatialUI controls the demo panels use.
//
// WHY. On the headset the catalogue cards (24 x 20 cm) were easy, but SpatialUI's default
// SegmentControl items are ~40 dp tall (3.2 cm = 0.7 deg at 2.5 m) and the user could not hit the
// subject tabs at all; in the solar panel they had to look ABOVE a control to select it, i.e. eye
// tracking carries a vertical offset of a centimetre or two. A target only survives that if it is
// several times larger than the error, so every control here is >= 76 dp (~6 cm, ~1.4 deg) tall
// and uses the SYSTEM hover effect (scale + highlight, rendered by PICO OS) so the user sees which
// control their eyes are on before they pinch.
//
// Same names and call shapes as com.pico.spatial.ui.design.{SegmentControl, SegmentItem, Button}:
// a demo switches by changing its imports, nothing else.

package com.bytedance.ailab.lab.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.composed
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.pico.spatial.ui.design.PicoTheme
import com.pico.spatial.ui.design.ProvideTextStyle
import com.pico.spatial.ui.foundation.hover.spatialHoverEffect
import com.pico.spatial.ui.foundation.hover.tween
import com.pico.spatial.ui.graphics.SpatialHoverStyle

/** Minimum height of every interactive control (dp). */
const val GAZE_TARGET_DP = 76
/** Invisible extra hit area above and below each chip (dp); rows space chips by 12 - 2 x this less. */
const val HIT_SLOP_DP = 6
val ACCENT = Color(0xFF4F9DFF)
private val CHIP = Color(0x2EFFFFFF)
private val CHIP_ON = Color(0xFF3B82F6)

/**
 * Click for gaze + pinch. `Modifier.clickable` CANCELS when the pointer leaves the bounds — and in
 * PICO's Compose the pointer follows the HAND during a pinch, so a little drift near the edge of a
 * 6 cm chip (the user's gaze sits 1-2 cm off) silently dropped the click while the highlight showed.
 * Compose routes every event of a pointer to the node that got its DOWN, so: down here + release
 * within [maxMs] = click, wherever the hand wandered.
 */
fun Modifier.gazeClick(maxMs: Long = 1500L, onClick: () -> Unit): Modifier = composed {
    val latest by rememberUpdatedState(onClick)
    pointerInput(Unit) {
        awaitEachGesture {
            val down = awaitFirstDown(requireUnconsumed = false)
            while (true) {
                val c = awaitPointerEvent().changes.firstOrNull { it.id == down.id } ?: break
                if (!c.pressed) {
                    val ms = c.uptimeMillis - down.uptimeMillis
                    android.util.Log.i("LabTap", "panel click ${ms} ms -> ${if (ms <= maxMs) "CLICK" else "too long"}")
                    if (ms <= maxMs) { c.consume(); latest() }
                    break
                }
            }
        }
    }
}

/**
 * System-side gaze feedback: grow on look + highlight layer. [dim] < 1 dims it while NOT looked at.
 * The grow is rendered by the OS and does NOT enlarge the hit area — keep it small (<= 1.05) so the
 * part that looks pinchable is pinchable; contrast comes from [dim] and the highlight.
 */
fun Modifier.gazeFeedback(grow: Float = 1.04f, dim: Float = 1f, shape: RoundedCornerShape? = null): Modifier =
    this.spatialHoverEffect { ctx ->
        // the OS highlight is a RECTANGLE unless the hover layer is clipped: give it the chip's own
        // rounded shape (this clips only the hover effect — the hit area stays the full rectangle)
        if (shape != null) clipShape(shape, ctx.size, androidx.compose.ui.unit.IntOffset.Zero)
        animation(tween(140)) { scale(if (ctx.isActive) grow else 1f) }
        if (dim < 1f) alpha(if (ctx.isActive) 1f else dim)
    }.spatialHoverEffect(SpatialHoverStyle.Highlight)

/**
 * Container for [SegmentItem]s (drop-in for the SpatialUI SegmentControl). A FLOW row: when the items
 * do not fit (English is ~1.6x wider than Chinese) the next item starts a new line. A plain Row gave
 * the last item whatever width was left, and its text wrapped letter by letter into a vertical stack.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun SegmentControl(modifier: Modifier = Modifier, content: @Composable RowScope.() -> Unit) {
    FlowRow(modifier = modifier, horizontalArrangement = Arrangement.spacedBy(10.dp), verticalArrangement = Arrangement.spacedBy(0.dp),
        itemVerticalAlignment = Alignment.CenterVertically) { content() }
}

/** One big selectable chip (drop-in for the SpatialUI SegmentItem). */
@Composable
fun RowScope.SegmentItem(
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    title: @Composable () -> Unit = {},
) {
    Chip(selected, onClick, modifier, title)
}

/** A big button (drop-in for the SpatialUI Button's `Button(onClick) { content }` form). */
@Composable
fun Button(onClick: () -> Unit, modifier: Modifier = Modifier, enabled: Boolean = true, content: @Composable () -> Unit) {
    Chip(selected = false, onClick = { if (enabled) onClick() }, modifier = modifier, content = content, outlined = true, enabled = enabled)
}

@Composable
internal fun Chip(selected: Boolean, onClick: () -> Unit, modifier: Modifier = Modifier,
                  content: @Composable () -> Unit, outlined: Boolean = false, enabled: Boolean = true) {
    val shape = RoundedCornerShape((GAZE_TARGET_DP / 2).dp)
    Box(
        modifier = modifier
            // the hit area is the full RECTANGLE (+6 dp above and below for the vertical gaze offset):
            // no clip layer — the PICO hit test checks a clip's rounded outline, which shaved the ends
            .gazeClick(onClick = onClick)
            .padding(vertical = HIT_SLOP_DP.dp)
            .heightIn(min = GAZE_TARGET_DP.dp)
            .defaultMinSize(minWidth = 112.dp)
            .gazeFeedback(shape = shape)
            // unavailable: faded (still clickable, so a pinch can say WHY instead of silently doing nothing)
            .alpha(if (enabled) 1f else 0.38f)
            .background(if (selected) CHIP_ON else CHIP, shape)
            .then(if (outlined && !selected) Modifier.border(2.dp, Color(0x55FFFFFF), shape) else Modifier)
            .padding(horizontal = 26.dp, vertical = 10.dp),
        contentAlignment = Alignment.Center,
    ) {
        ProvideTextStyle(PicoTheme.typography.bodyLarge.copy(fontSize = 24.sp, color = if (selected) Color.White else PicoTheme.colorScheme.labelPrimary)) {
            content()
        }
    }
}
