// Shared SpatialUI building blocks so all demo panels look like one product: same width, same
// type scale, same label column. SpatialUI only (com.pico.spatial.ui.design), inside PicoTheme.

package com.bytedance.ailab.lab.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.FlowRowScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.runtime.remember
import com.bytedance.ailab.lab.LabHost
import com.bytedance.ailab.lab.tr
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.pico.spatial.ui.design.PicoTheme
import com.pico.spatial.ui.design.Slider
import com.pico.spatial.ui.design.SliderDefaults
import com.pico.spatial.ui.design.Text
import com.pico.spatial.ui.foundation.material.backgroundMaterial
import com.pico.spatial.ui.platform.Material

/**
 * Panel width. 1040 dp was fine in Chinese, but once rows WRAP (English, more controls) panels grew to
 * 880 dp = 0.70 m tall (device log matched UiPreviewTest to 0.5%): centred at y = -0.36 m that is
 * clipped by the volume floor (-0.6 m) and hides the bottom of the experiment. Wide and low instead:
 * 1560 dp = 1.25 m (the volume is 2.16 m wide), controls FLOW side by side, height capped by the test.
 */
const val PANEL_WIDTH_DP = 1560
/** Tallest allowed panel (dp): 0.42 m, i.e. y from -0.57 to -0.15 m around PANEL_POS. */
const val PANEL_MAX_H_DP = 520

/** Height of the last laid-out panel in px (UiPreviewTest, mdpi: px = dp). */
var lastPanelHeightPx = 0

/** A demo's control panel: a live readout (2-3 lines of physics) above the controls, which flow. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun LabPanel(readout: String, controls: @Composable FlowRowScope.() -> Unit) {
    val lastSize = remember { intArrayOf(0, 0) }
    Column(
        modifier = Modifier
            .onGloballyPositioned { c ->
                // device diagnostic: is the panel actually laid out, and how big?
                lastPanelHeightPx = c.size.height
                if (c.size.width != lastSize[0] || c.size.height != lastSize[1]) {
                    lastSize[0] = c.size.width; lastSize[1] = c.size.height
                    android.util.Log.i("LabPanel", "${LabHost.currentId} panel ${c.size.width}x${c.size.height} px")
                }
            }
            .width(PANEL_WIDTH_DP.dp)
            .clip(RoundedCornerShape(40.dp))
            .backgroundMaterial(true, Material.Regular)
            .padding(horizontal = 28.dp, vertical = 20.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(
            text = readout,
            color = PicoTheme.colorScheme.labelPrimary,
            style = PicoTheme.typography.bodyLarge.copy(fontSize = 22.sp, lineHeight = 30.sp),
            minLines = 3,
            // every readout line must fit the panel: a wrapped line makes the panel grow and jump
            onTextLayout = { r -> readoutOverflow = r.lineCount - maxOf(3, readout.count { it == '\n' } + 1) },
        )
        // sliders and chip groups flow side by side (two sliders per line at this width)
        FlowRow(horizontalArrangement = Arrangement.spacedBy(36.dp), verticalArrangement = Arrangement.spacedBy(0.dp),
            itemVerticalAlignment = Alignment.CenterVertically, content = controls)
        // the demo's pinchable objects, as buttons that run the SAME action as pinching the object
        val targets = LabHost.demo?.tapTargets.orEmpty()
        if (targets.isNotEmpty()) LabRow {
            Text("✋ " + tr("捏场景里的物体，或点：", "Pinch the object, or tap:"), color = PicoTheme.colorScheme.labelSecondary,
                style = PicoTheme.typography.bodyLarge.copy(fontSize = 22.sp), maxLines = 1)
            targets.forEach { t ->
                val on = t.isOn?.invoke() == true
                SegmentItem(selected = on, onClick = { LabHost.onTap(t.node, fromPanel = true) }, title = { Text((if (on) "✓ " else "") + tr(t.zh, t.en)) })
            }
        }
    }
}

/** Extra lines the last laid-out readout wrapped onto (0 = every line fits). Read by UiPreviewTest. */
var readoutOverflow = 0

/** A row of controls that WRAPS instead of squeezing its last item (see [SegmentControl]). */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun LabRow(content: @Composable FlowRowScope.() -> Unit) {
    FlowRow(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalArrangement = Arrangement.spacedBy(0.dp),
        itemVerticalAlignment = Alignment.CenterVertically, content = content)
}

/** Slider with a gaze-sized thumb (72 dp target): the default thumb was too small to grab by gaze.
 *  Default track 400 dp so two sliders share a line of the 1560 dp panel. */
@Composable
fun LabSlider(label: String, value: Float, range: ClosedFloatingPointRange<Float>, trailing: String,
              width: Int = 400, onChange: (Float) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.heightIn(min = GAZE_TARGET_DP.dp)) {
        Text(label, color = PicoTheme.colorScheme.labelSecondary, modifier = Modifier.widthIn(min = 130.dp).padding(end = 12.dp),
            style = PicoTheme.typography.bodyLarge.copy(fontSize = 22.sp), maxLines = 1)
        Slider(value = value, valueRange = range, onValueChange = onChange, modifier = Modifier.width(width.dp),
            sliderSpec = SliderDefaults.sliderSpec(thumbAreaSize = 72.dp, thumbSize = 34.dp, thumbPressedSize = 40.dp,
                thumbHoverSize = 42.dp, trackHeight = 12.dp))
        Spacer(Modifier.width(14.dp))
        Text(trailing, color = PicoTheme.colorScheme.labelSecondary, maxLines = 1, overflow = TextOverflow.Ellipsis,
            style = PicoTheme.typography.bodyLarge.copy(fontSize = 22.sp), modifier = Modifier.widthIn(min = 110.dp))
    }
}

/** On/off as one big toggle chip (✓ when on): the SpatialUI Switch is a small target for gaze. */
@Composable
fun LabSwitch(label: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Chip(selected = checked, onClick = { onChange(!checked) }, content = { Text(if (checked) "✓ $label" else label) })
}
