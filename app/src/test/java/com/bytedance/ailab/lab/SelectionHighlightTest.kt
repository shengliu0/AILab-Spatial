package com.bytedance.ailab.lab

import android.os.Looper
import androidx.activity.ComponentActivity
import androidx.compose.runtime.key
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.ui.platform.ComposeView
import com.bytedance.ailab.lab.ui.ChipProbe
import com.pico.spatial.ui.design.PicoTheme
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode
import java.io.File

/**
 * "I chose 10⁶ but the button did not light up": renders every demo's REAL panel (in every review state),
 * clicks each option chip (SegmentItem) and on/off switch the way a pinch does, lets the panel recompose as on
 * the device (LabVolume keys it on LabHost.uiEpoch), and checks that the clicked option now shows selected / the
 * switch flipped. Failures name the demo, state and the chip's position in the panel.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35], qualifiers = "w1800dp-h1500dp-mdpi", application = android.app.Application::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
class SelectionHighlightTest {
    @Test
    fun clickedOptionsLightUp() {
        val act = Robolectric.buildActivity(ComponentActivity::class.java).setup().get()
        LabHost.register(DemoRegistry.all)
        val cache = File("build/review-cache").apply { mkdirs() }
        val bad = ArrayList<String>()
        var checked = 0
        var live: ComposeView? = null
        fun snapshot(d: Demo): List<ChipProbe.Entry> {
            live?.disposeComposition()                                          // old panels must not keep laying out
            ChipProbe.entries.clear(); ChipProbe.recording = true
            val view = ComposeView(act); act.setContentView(view); live = view   // fresh view: a full composition every time
            view.setContent { androidx.compose.foundation.layout.Box(androidx.compose.ui.Modifier.wrapContentSize()) { PicoTheme { key(LabHost.currentId, LabHost.uiEpoch) { d.Panel() } } } }
            shadowOf(Looper.getMainLooper()).idle()
            ChipProbe.recording = false
            return ChipProbe.entries.toList()
        }
        for (d in DemoRegistry.all.filterIsInstance<SpecDemo>()) {
            val states = listOf<Pair<String, () -> Unit>>("default" to {}) + d.reviewVariants().toList()
            for ((st, apply) in states) {
                fun prepare() { d.runHeadless(cache, 0.3f); apply(); repeat(5) { d.update(1f / 60f) }; LabHost.previewAs(d); d.publishReadoutNow() }
                prepare()
                val first = snapshot(d); val n = first.size
                for (i in 0 until n) {
                    prepare()
                    val before = snapshot(d)
                    if (before.size != n) continue
                    val e = before[i]
                    if (!e.enabled || (e.kind != "segment" && e.kind != "switch")) continue
                    if (e.kind == "segment" && e.selected) continue
                    // click inside the SAME live composition (as on the device) and let it recompose by itself
                    ChipProbe.entries.clear(); ChipProbe.recording = true
                    e.onClick(); repeat(3) { d.update(1f / 60f) }; d.publishReadoutNow()
                    // the device's frame loop sends snapshot apply-notifications every frame; in a shared test JVM
                    // nothing guarantees it, so send them explicitly before letting the composition catch up
                    androidx.compose.runtime.snapshots.Snapshot.sendApplyNotifications()
                    shadowOf(Looper.getMainLooper()).idle()
                    ChipProbe.recording = false
                    val after = ChipProbe.entries.toList()
                    checked++
                    if (after.isEmpty()) { bad += "${d.id} [$st] ${e.kind} #${i + 1}/$n: clicked but the panel did not redraw (stale highlight)"; continue }
                    if (after.size != before.size) continue                 // partial redraw or new layout: cannot index, fine
                    val ok = if (e.kind == "segment") after[i].selected else after[i].selected != e.selected
                    if (!ok) bad += "${d.id} [$st] ${e.kind} #${i + 1}/$n: clicked but not ${if (e.kind == "segment") "highlighted" else "toggled"}"
                }
            }
        }
        live?.disposeComposition()
        println("HIGHLIGHT $checked option clicks checked, ${bad.size} problems")
        bad.forEach { println("HIGHLIGHT   $it") }
        assertTrue(bad.joinToString("\n"), bad.isEmpty())
    }
}
