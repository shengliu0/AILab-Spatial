package com.bytedance.ailab.lab

import android.graphics.Bitmap
import android.graphics.Canvas
import android.os.Looper
import androidx.activity.ComponentActivity
import androidx.compose.ui.platform.ComposeView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.positionInRoot
import androidx.compose.ui.unit.dp
import com.bytedance.ailab.content.LabCatalog
import com.bytedance.ailab.content.LabTopBar
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
import java.io.FileOutputStream

/**
 * UI PREVIEW (headless): renders the REAL Compose panels, top bar and catalogue in both display
 * languages to build/review/ui/<lang>_<what>.png, so layout overflow (English is ~1.6x wider than
 * Chinese) is caught on the Mac instead of in the headset. Robolectric native graphics, mdpi
 * (1 dp = 1 px). The PICO glass material is a system effect and does not render here; the backdrop
 * is a flat light grey (preview text colours are the light-theme ones). Select one demo with -Dreview.only=<id>.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35], qualifiers = "w1800dp-h1500dp-mdpi", application = android.app.Application::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
class UiPreviewTest {
    private val out = File("build/review/ui").apply { mkdirs() }

    @Test
    fun renderPanelsInBothLanguages() {
        val only = System.getProperty("review.only").orEmpty()
        val act = Robolectric.buildActivity(ComponentActivity::class.java).setup().get()
        LabHost.register(DemoRegistry.all)
        val cache = File("build/review-cache").apply { mkdirs() }
        var n = 0
        val wrapped = ArrayList<String>()
        val tall = ArrayList<String>()
        for (en in listOf(false, true)) {
            Lang.set(en)
            val tag = if (en) "en" else "zh"
            for (d in DemoRegistry.all.filterIsInstance<SpecDemo>()) {
                if (only.isNotEmpty() && d.id != only) continue
                d.runHeadless(cache, 2f); d.publishReadoutNow()
                LabHost.previewAs(d)
                capture(act, File(out, "${tag}_panel_${d.id}.png")) { d.Panel() }; n++
                val h = com.bytedance.ailab.lab.ui.lastPanelHeightPx
                if (h > com.bytedance.ailab.lab.ui.PANEL_MAX_H_DP) tall += "$tag/${d.id} ${h} dp"
                if (com.bytedance.ailab.lab.ui.readoutOverflow > 0) wrapped += "$tag/${d.id} (+${com.bytedance.ailab.lab.ui.readoutOverflow})"
                for ((name, apply) in d.reviewVariants()) {
                    apply(); repeat(30) { d.update(1f / 60f) }; d.publishReadoutNow()
                    capture(act, File(out, "${tag}_panel_${d.id}-$name.png")) { d.Panel() }; n++
                    if (com.bytedance.ailab.lab.ui.readoutOverflow > 0) wrapped += "$tag/${d.id}-$name (+${com.bytedance.ailab.lab.ui.readoutOverflow})"
                }
            }
            // the guided-track card at its first step (and at a PREDICT step, if any) for every demo with a lesson
            for (d in DemoRegistry.all.filterIsInstance<SpecDemo>()) {
                if (only.isNotEmpty() && d.id != only) continue
                val l = d.lesson ?: continue
                LabHost.previewAs(d); LessonRunner.start(l)
                capture(act, File(out, "${tag}_lesson_${d.id}.png")) { com.bytedance.ailab.content.LessonCard() }; n++
                // every step's instruction and insight, and the challenge, so long later cards get looked at too
                for (i in l.steps.indices) {
                    LessonRunner.previewAt(i, false)
                    capture(act, File(out, "${tag}_lesson_${d.id}_s${i + 1}.png")) { com.bytedance.ailab.content.LessonCard() }; n++
                    if (l.steps[i].insight != null) { LessonRunner.previewAt(i, true)
                        capture(act, File(out, "${tag}_lesson_${d.id}_s${i + 1}_insight.png")) { com.bytedance.ailab.content.LessonCard() }; n++ }
                }
                if (l.challenge != null) { LessonRunner.previewAt(l.steps.size, false)
                    capture(act, File(out, "${tag}_lesson_${d.id}_challenge.png")) { com.bytedance.ailab.content.LessonCard() }; n++ }
            }
            // the background card (📖): every page, both languages, must fit without scrolling
            for (d in DemoRegistry.all) {
                if (only.isNotEmpty() && d.id != only) continue
                val ab = d.about ?: continue
                LabHost.previewAs(d)
                for (i in ab.pages.indices) {
                    capture(act, File(out, "${tag}_about_${d.id}_p${i + 1}.png")) { com.bytedance.ailab.content.AboutCard(previewPage = i) }; n++
                    val h = com.bytedance.ailab.content.lastAboutHeightPx
                    if (h > com.bytedance.ailab.content.ABOUT_MAX_H_DP) tall += "$tag/${d.id} about p${i + 1} ${h} dp"
                }
            }
            if (only.isEmpty()) {
                capture(act, File(out, "${tag}_topbar.png")) { LabTopBar { } }; n++
                LabHost.showCatalog(true)
                capture(act, File(out, "${tag}_catalog.png")) { LabCatalog { } }; n++
                LabHost.showCatalog(false)
            }
        }
        Lang.set(false)
        println("READOUT WRAPS: ${if (wrapped.isEmpty()) "none" else wrapped.joinToString()}")
        assertTrue(n > 0)
        println("PANELS TALLER THAN ${com.bytedance.ailab.lab.ui.PANEL_MAX_H_DP} dp: ${if (tall.isEmpty()) "none" else tall.joinToString()}")
        assertTrue("readout lines wrap (panel grows and jumps): $wrapped", wrapped.isEmpty())
        assertTrue("panels too tall (clipped by the volume floor, hide the experiment): $tall", tall.isEmpty())
    }

    private fun capture(act: ComponentActivity, file: File, content: @Composable () -> Unit) {
        val box = IntArray(4)
        val view = ComposeView(act)
        act.setContentView(view)
        view.setContent {
            Box(Modifier.background(Color(0xFFC9CED6)).padding(24.dp)) {
                Box(Modifier.onGloballyPositioned { c ->
                    val p = c.positionInRoot(); box[0] = p.x.toInt(); box[1] = p.y.toInt(); box[2] = c.size.width; box[3] = c.size.height
                }) { PicoTheme { content() } }
            }
        }
        shadowOf(Looper.getMainLooper()).idle()
        val root = act.window.decorView
        val full = Bitmap.createBitmap(root.width, root.height, Bitmap.Config.ARGB_8888)
        root.draw(Canvas(full))
        val m = 24
        val x0 = (box[0] - m).coerceAtLeast(0); val y0 = (box[1] - m).coerceAtLeast(0)
        val w = (box[2] + 2 * m).coerceAtMost(full.width - x0); val h = (box[3] + 2 * m).coerceAtMost(full.height - y0)
        val crop = Bitmap.createBitmap(full, x0, y0, maxOf(1, w), maxOf(1, h))
        FileOutputStream(file).use { crop.compress(Bitmap.CompressFormat.PNG, 100, it) }
        println("UI ${file.name}: content ${box[2]}x${box[3]} dp")
    }
}
