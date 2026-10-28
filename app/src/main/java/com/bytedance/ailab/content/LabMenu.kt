// LabMenu — choosing an experiment by gaze + pinch.
//
// WHY THIS SHAPE. The first menu packed 11 long titles into two SegmentControl rows: each target was
// ~40 dp tall, i.e. 3.2 cm = 0.7 deg at the Shared-Space viewing distance of 2.5 m — below the
// ~1-1.5 deg error of eye tracking, so picking one was a lottery, and it only gets worse as demos
// are added. Now:
//   * a slim TOP BAR (always there): [ 实验目录 ] [ ‹ ] current title [ › ] — every target 96 dp
//     (7.7 cm, 1.8 deg);
//   * a CATALOGUE (on demand): subject tabs + a 5 x 3 grid of picture cards, 225 x 188 dp each
//     (18 x 15 cm, 4.1 x 3.4 deg — the user asked for 25% smaller than the first 300 x 250 cut; still
//     ~3x the eye-tracking error) — 15 per page scales to any number of demos.
// Gaze feedback is the SYSTEM hover effect (rendered by PICO OS, the app never sees where the eyes
// are): the looked-at card grows 7% and gets the highlight layer; the running demo has an accent ring.

package com.bytedance.ailab.content

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.bytedance.ailab.lab.Demo
import com.bytedance.ailab.lab.DemoRegistry
import com.bytedance.ailab.lab.LabHost
import com.bytedance.ailab.lab.Lang
import com.bytedance.ailab.lab.tr
import com.bytedance.ailab.lab.Subject
import com.bytedance.ailab.lab.ui.Button
import com.pico.spatial.ui.design.PicoTheme
import com.bytedance.ailab.lab.ui.SegmentControl
import com.bytedance.ailab.lab.ui.SegmentItem
import com.pico.spatial.ui.design.Text
import com.bytedance.ailab.lab.ui.gazeClick
import com.bytedance.ailab.lab.ui.gazeFeedback
import com.pico.spatial.ui.foundation.material.backgroundMaterial
import com.pico.spatial.ui.platform.Material

private const val CARD_W = 225           // 18 cm = 4.1 deg at 2.5 m (was 300; still 2x the gaze error)
private const val CARD_H = 188
private const val THUMB_H = 134           // ~16:9.5 thumbnail; leaves 54 dp for title + credit
private const val COLS = 6
private const val PER_PAGE = 18
private val ACCENT = Color(0xFF5AB2FF)
private val BIG = 96.dp

/** Always-visible slim bar: open the catalogue, step through demos, see what is running. */
@Composable
fun LabTopBar(onPick: (String) -> Unit) {
    val current = LabHost.demos.firstOrNull { it.id == LabHost.currentId }
    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(48.dp))
            .backgroundMaterial(true, Material.Regular)
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Button(onClick = { LabHost.showCatalog(!LabHost.catalogOpen) }, modifier = Modifier.height(BIG).widthIn(min = 230.dp)) {
            Text(if (LabHost.catalogOpen) tr("✕ 关闭目录", "✕ Close") else tr("☰ 实验目录", "☰ Experiments"), maxLines = 1, style = PicoTheme.typography.bodyLarge.copy(fontSize = 28.sp))
        }
        Button(onClick = { LabHost.neighbour(-1)?.let(onPick) }, modifier = Modifier.size(BIG)) {
            Text("‹", style = PicoTheme.typography.bodyLarge.copy(fontSize = 44.sp))
        }
        Column(modifier = Modifier.width(420.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Text(current?.let { DemoRegistry.titleOf(it) } ?: "…", color = PicoTheme.colorScheme.labelPrimary, maxLines = 1, overflow = TextOverflow.Ellipsis,
                style = PicoTheme.typography.bodyLarge.copy(fontSize = 30.sp, fontWeight = FontWeight.SemiBold))
            Text(current?.credit ?: "", color = PicoTheme.colorScheme.labelSecondary,
                style = PicoTheme.typography.bodyLarge.copy(fontSize = 20.sp))
        }
        Button(onClick = { LabHost.neighbour(+1)?.let(onPick) }, modifier = Modifier.size(BIG)) {
            Text("›", style = PicoTheme.typography.bodyLarge.copy(fontSize = 44.sp))
        }
        // display language: one big two-state chip; the label names the language you will GET
        Button(onClick = { Lang.set(!Lang.en) }, modifier = Modifier.height(BIG).width(150.dp)) {
            Text(if (Lang.en) "中文" else "EN", style = PicoTheme.typography.bodyLarge.copy(fontSize = 28.sp, fontWeight = FontWeight.SemiBold))  // i18n-ok (language names)
        }
    }
}

/** The on-demand catalogue: subject tabs, picture cards, pages. Renders nothing while closed. */
@Composable
fun LabCatalog(onPick: (String) -> Unit) {
    if (!LabHost.catalogOpen) { Box(Modifier.size(1.dp)); return }
    var subject by remember { mutableStateOf<Subject?>(null) }          // null = 全部
    var page by remember { mutableIntStateOf(0) }
    val shown = DemoRegistry.catalog.filter { subject == null || it.second == subject }.map { it.first }
    val pages = maxOf(1, (shown.size + PER_PAGE - 1) / PER_PAGE)
    if (page >= pages) page = 0
    val onPage = shown.drop(page * PER_PAGE).take(PER_PAGE)

    Column(
        modifier = Modifier
            .clip(RoundedCornerShape(48.dp))
            .backgroundMaterial(true, Material.Thick)
            .padding(28.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp),
    ) {
        // tabs: only subjects that have demos, so empty tabs never appear
        val subjects = Subject.values().filter { s -> DemoRegistry.catalog.any { it.second == s } }
        SegmentControl {
            SegmentItem(selected = subject == null, onClick = { subject = null; page = 0 },
                title = { Text(tr("全部", "All") + " ${DemoRegistry.catalog.size}", style = PicoTheme.typography.bodyLarge.copy(fontSize = 26.sp)) })
            subjects.forEach { s ->
                val n = DemoRegistry.catalog.count { it.second == s }
                SegmentItem(selected = subject == s, onClick = { subject = s; page = 0 },
                    title = { Text("${s.label} $n", style = PicoTheme.typography.bodyLarge.copy(fontSize = 26.sp)) })
            }
        }
        for (row in onPage.chunked(COLS)) {
            Row(horizontalArrangement = Arrangement.spacedBy(18.dp)) {
                row.forEach { d -> DemoCard(d) { onPick(d.id); LabHost.showCatalog(false) } }
            }
        }
        if (pages > 1) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                Button(onClick = { page = (page + pages - 1) % pages }, modifier = Modifier.size(BIG)) { Text("‹", fontSize = 44.sp) }
                Text("${page + 1} / $pages", style = PicoTheme.typography.bodyLarge.copy(fontSize = 26.sp))
                Button(onClick = { page = (page + 1) % pages }, modifier = Modifier.size(BIG)) { Text("›", fontSize = 44.sp) }
            }
        }
    }
}

@Composable
private fun DemoCard(d: Demo, onClick: () -> Unit) {
    val context = LocalContext.current
    val thumb = remember(d.id) { context.resources.getIdentifier("thumb_${d.id}", "drawable", context.packageName) }
    val running = LabHost.currentId == d.id
    val shape = RoundedCornerShape(22.dp)
    Column(
        modifier = Modifier
            .width(CARD_W.dp).height(CARD_H.dp)
            // gaze (system-side): the looked-at card grows 5% and gets the highlight layer, every
            // OTHER card dims to 72% (the contrast cue). Grow is kept small: the OS draws it but the
            // hit area stays 1.0x. The click survives hand drift (gazeClick) — no Modifier.clickable.
            .gazeClick(onClick = onClick)
            .gazeFeedback(grow = 1.05f, dim = 0.72f)
            .clip(shape)
            .background(Color(0x33000000))
            .border(if (running) 4.dp else 1.dp, if (running) ACCENT else Color(0x33FFFFFF), shape),
    ) {
        Box {
            if (thumb != 0) Image(painter = painterResource(thumb), contentDescription = DemoRegistry.titleOf(d), contentScale = ContentScale.Crop,
                modifier = Modifier.width(CARD_W.dp).height(THUMB_H.dp))
            else Box(Modifier.width(CARD_W.dp).height(THUMB_H.dp).background(Color(0xFF1B1F27)))
            if (running) Text(tr("正在运行", "Running"), color = Color.White, style = PicoTheme.typography.bodyLarge.copy(fontSize = 18.sp),
                modifier = Modifier.padding(10.dp).clip(RoundedCornerShape(12.dp)).background(ACCENT).padding(horizontal = 10.dp, vertical = 2.dp))
        }
        Spacer(Modifier.height(4.dp))
        // explicit line heights: CJK glyphs are taller and pushed the credit line out of the card
        Text(DemoRegistry.cardTitleOf(d), color = PicoTheme.colorScheme.labelPrimary, maxLines = 1, overflow = TextOverflow.Ellipsis,
            style = PicoTheme.typography.bodyLarge.copy(fontSize = 20.sp, lineHeight = 26.sp, fontWeight = FontWeight.SemiBold),
            modifier = Modifier.padding(horizontal = 12.dp))
        Text(d.credit, color = PicoTheme.colorScheme.labelSecondary, maxLines = 1,
            style = PicoTheme.typography.bodyLarge.copy(fontSize = 15.sp, lineHeight = 18.sp), modifier = Modifier.padding(horizontal = 12.dp))
    }
}
