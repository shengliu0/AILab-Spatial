// The background card ("📖 背景"): a few short pages about the current experiment — what it is, the
// story, the idea in plain words, where it matters — so a viewer new to the topic has the context the
// scene assumes. Paged (‹ ›) rather than scrolled: scrolling by gaze is tiring. Optional read-aloud.

package com.bytedance.ailab.content

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.bytedance.ailab.lab.LabHost
import com.bytedance.ailab.lab.LabSettings
import com.bytedance.ailab.lab.Narrator
import com.bytedance.ailab.lab.tr
import com.bytedance.ailab.lab.ui.Button
import com.pico.spatial.ui.design.PicoTheme
import com.pico.spatial.ui.design.Text
import com.pico.spatial.ui.foundation.material.backgroundMaterial
import com.pico.spatial.ui.platform.Material

/** Card width (dp); the body is sized so every page fits without scrolling. */
const val ABOUT_WIDTH_DP = 640
/** Tallest allowed page (dp): the card sits at y = 0.26 m, 700 dp = 0.56 m keeps it inside the volume. */
const val ABOUT_MAX_H_DP = 700
/** Height of the last laid-out card in px (UiPreviewTest, mdpi: px = dp). */
var lastAboutHeightPx = 0

@Composable
fun AboutCard(previewPage: Int? = null) {
    val d = LabHost.demo
    val about = d?.about
    if ((previewPage == null && !LabSettings.aboutOpen) || d == null || about == null || about.pages.isEmpty()) { Box(Modifier.size(1.dp)); return }
    var page by remember(d.id) { mutableIntStateOf(0) }
    if (previewPage != null) page = previewPage
    if (page >= about.pages.size) page = 0
    val p = about.pages[page]
    val n = about.pages.size
    Column(
        modifier = Modifier
            .onGloballyPositioned { lastAboutHeightPx = it.size.height }
            .width(ABOUT_WIDTH_DP.dp)
            .clip(RoundedCornerShape(36.dp))
            .backgroundMaterial(true, Material.Thick)
            .padding(horizontal = 26.dp, vertical = 20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("📖 " + tr("背景", "Background") + " · ${page + 1}/$n", color = PicoTheme.colorScheme.labelSecondary,
                style = PicoTheme.typography.bodyLarge.copy(fontSize = 22.sp), modifier = Modifier.weight(1f), maxLines = 1)
            Button(onClick = { LabSettings.setAbout(false) }) { Text(tr("收起", "Hide")) }
        }
        Text(p.title.text, color = PicoTheme.colorScheme.labelPrimary, maxLines = 2,
            style = PicoTheme.typography.bodyLarge.copy(fontSize = 28.sp, lineHeight = 36.sp, fontWeight = FontWeight.SemiBold))
        Text(p.body.text, color = PicoTheme.colorScheme.labelPrimary, modifier = Modifier.heightIn(min = 300.dp),
            style = PicoTheme.typography.bodyLarge.copy(fontSize = 22.sp, lineHeight = 33.sp))
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(onClick = { Narrator.stop(); page = (page + n - 1) % n }) { Text("‹ " + tr("上一页", "Back")) }
            Button(onClick = { Narrator.stop(); page = (page + 1) % n }) { Text(tr("下一页", "Next") + " ›") }
            Spacer(Modifier.weight(1f))
            if (Narrator.available) Button(onClick = { if (Narrator.speaking) Narrator.stop() else Narrator.speak(p.title.text + tr("。", ". ") + p.body.text) }) {
                Text(if (Narrator.speaking) tr("■ 停止", "■ Stop") else tr("🔊 朗读", "🔊 Read"))
            }
        }
    }
}

