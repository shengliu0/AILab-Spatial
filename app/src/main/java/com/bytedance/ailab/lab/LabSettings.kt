// LabSettings — the viewer's layout choices, remembered across launches: content size, where the
// control panel sits, whether the background ("wiki") card is open.

package com.bytedance.ailab.lab

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

object LabSettings {
    /** Content scale steps. The scene is built for |x| < 0.9 m in a 2.16 m wide volume, so 1.2x still fits. */
    val SCALES = floatArrayOf(1f, 1.2f)
    /** Panel positions: front-bottom (wide, default), left side, right side (narrow, turned to the viewer). */
    const val PANEL_BOTTOM = 0
    const val PANEL_LEFT = 1
    const val PANEL_RIGHT = 2

    private var prefs: android.content.SharedPreferences? = null

    var scaleIdx by mutableIntStateOf(0)
        private set
    var panelPos by mutableIntStateOf(PANEL_BOTTOM)
        private set
    var aboutOpen by mutableStateOf(false)
        private set

    val scale: Float get() = SCALES[scaleIdx.coerceIn(0, SCALES.size - 1)]

    fun init(context: Context) {
        prefs = context.getSharedPreferences("lab", Context.MODE_PRIVATE)
        scaleIdx = (prefs?.getInt("scale", 0) ?: 0).coerceIn(0, SCALES.size - 1)
        panelPos = (prefs?.getInt("panel", PANEL_BOTTOM) ?: PANEL_BOTTOM).coerceIn(0, 2)
    }

    fun cycleScale() { scaleIdx = (scaleIdx + 1) % SCALES.size; prefs?.edit()?.putInt("scale", scaleIdx)?.apply() }
    fun cyclePanel() { panelPos = (panelPos + 1) % 3; prefs?.edit()?.putInt("panel", panelPos)?.apply() }
    fun setAbout(open: Boolean) { aboutOpen = open; if (!open) Narrator.stop() }
}
