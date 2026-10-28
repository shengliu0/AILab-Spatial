// LabLesson — the guided-track card (导学), floating at the upper left of the volume, beside the top bar:
// "导学 2/4", the instruction, prediction options, the insight when a step completes, the challenge.
// Never blocks free play: skip, collapse to a small pill, or restart at any time.

package com.bytedance.ailab.content

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.bytedance.ailab.lab.LessonRunner
import com.bytedance.ailab.lab.LessonRunner.Phase
import com.bytedance.ailab.lab.tr
import com.bytedance.ailab.lab.ui.Button
import com.bytedance.ailab.lab.ui.LabRow
import com.bytedance.ailab.lab.ui.SegmentItem
import com.pico.spatial.ui.design.PicoTheme
import com.pico.spatial.ui.design.Text
import com.pico.spatial.ui.foundation.material.backgroundMaterial
import com.pico.spatial.ui.platform.Material

const val LESSON_CARD_DP = 520
private val ACCENT_BG = Color(0xCC2F6FDB)
private val INSIGHT_BG = Color(0xCC1E8F6A)

@Composable
fun LessonCard() {
    @Suppress("UNUSED_VARIABLE") val e = LessonRunner.epoch          // recompose on every change
    val phase = LessonRunner.phase
    if (phase == Phase.NONE) { Spacer(Modifier.width(1.dp)); return }
    if (LessonRunner.collapsed) {
        Button(onClick = { LessonRunner.collapsed = false }) {
            Text(tr("📘 导学 ", "📘 Guide ") + progressLabel() + "  ›", style = PicoTheme.typography.bodyLarge.copy(fontSize = 24.sp))
        }
        return
    }
    Column(
        modifier = Modifier.width(LESSON_CARD_DP.dp).clip(RoundedCornerShape(32.dp))
            .backgroundMaterial(true, Material.Regular).padding(horizontal = 24.dp, vertical = 18.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(tr("📘 导学 ", "📘 Guide ") + progressLabel(), color = PicoTheme.colorScheme.labelPrimary,
                style = PicoTheme.typography.bodyLarge.copy(fontSize = 24.sp, fontWeight = FontWeight.SemiBold),
                modifier = Modifier.weight(1f))
            if (phase == Phase.STEP || phase == Phase.INSIGHT || phase == Phase.CHALLENGE)
                Button(onClick = { LessonRunner.skip() }) { Text(tr("跳过 ›", "Skip ›")) }
            Spacer(Modifier.width(10.dp))
            Button(onClick = { LessonRunner.collapsed = true }) { Text(tr("收起", "Hide")) }
        }
        val body = PicoTheme.typography.bodyLarge.copy(fontSize = 23.sp, lineHeight = 31.sp)
        when (phase) {
            Phase.STEP -> {
                val s = LessonRunner.step ?: return@Column
                Text(s.say.text, color = PicoTheme.colorScheme.labelPrimary, style = body)
                if (s.isPredict) LabRow {
                    s.options.forEachIndexed { i, o -> SegmentItem(selected = false, onClick = { LessonRunner.choose(i) }, title = { Text(o.text) }) }
                }
            }
            Phase.INSIGHT -> Text("✓ " + LessonRunner.insightText, color = Color.White, style = body,
                modifier = Modifier.clip(RoundedCornerShape(18.dp)).background(INSIGHT_BG).padding(horizontal = 16.dp, vertical = 10.dp))
            Phase.CHALLENGE -> {
                val c = LessonRunner.lesson?.challenge ?: return@Column
                Text(tr("🏆 挑战：", "🏆 Challenge: ") + c.say.text, color = PicoTheme.colorScheme.labelPrimary, style = body)
                if (LessonRunner.challengeText.isNotEmpty())
                    Text(LessonRunner.challengeText, color = Color.White, style = body,
                        modifier = Modifier.clip(RoundedCornerShape(18.dp)).background(ACCENT_BG).padding(horizontal = 16.dp, vertical = 8.dp))
            }
            Phase.FINISHED -> {
                Text(tr("导学完成 ✓  现在自由探索吧", "Guide complete ✓  Now explore freely"), color = PicoTheme.colorScheme.labelPrimary, style = body)
                Button(onClick = { LessonRunner.restart() }) { Text(tr("↺ 再来一遍", "↺ Start over")) }
            }
            Phase.NONE -> {}
        }
    }
}

private fun progressLabel(): String = when (LessonRunner.phase) {
    Phase.CHALLENGE -> tr("挑战", "challenge")
    Phase.FINISHED -> "✓"
    else -> "${(LessonRunner.index + 1).coerceAtMost(LessonRunner.total)}/${LessonRunner.total}"
}
