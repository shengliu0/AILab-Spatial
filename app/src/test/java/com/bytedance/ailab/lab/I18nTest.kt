package com.bytedance.ailab.lab

import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * The display-language switch, enforced.
 *  1. RUNTIME: in English every demo's readout and 3D labels contain no CJK character, and in Chinese
 *     the readout is still Chinese (the switch really switches).
 *  2. SOURCE: every CJK string literal in demos/content/lab is the FIRST argument of tr(...), so no
 *     panel text can be left untranslated (Compose panels cannot be rendered on the JVM, this is how
 *     they are checked). Comments and Log calls are exempt; a line can opt out with `// i18n-ok`.
 */
class I18nTest {
    private val cjk = Regex("[\\u4e00-\\u9fff\\u3000-\\u303f\\uff00-\\uffef]")

    @After fun back() = Lang.set(false)

    @Test
    fun readoutsAndLabelsFollowTheLanguage() {
        val cache = File("build/review-cache").apply { mkdirs() }
        val bad = ArrayList<String>()
        for (d in DemoRegistry.all.filterIsInstance<SpecDemo>()) {
            Lang.set(true)
            d.buildSpec(cache); d.update(0f); repeat(120) { d.update(1f / 60f) }
            if (d.readoutText.isBlank()) bad += "${d.id}: empty readout (EN)"
            if (cjk.containsMatchIn(d.readoutText)) bad += "${d.id}: CJK in EN readout: ${d.readoutText.lines().firstOrNull { cjk.containsMatchIn(it) }}"
            d.labels.filter { cjk.containsMatchIn(it.text) }.forEach { bad += "${d.id}: CJK in EN label '${it.text}'" }
            if (cjk.containsMatchIn(DemoRegistry.titleOf(d))) bad += "${d.id}: CJK in EN title"
            d.lesson?.let { l ->
                val texts = l.steps.flatMap { s -> listOfNotNull(s.say.en, s.insight?.en) + s.options.map { it.en } } + listOfNotNull(l.challenge?.say?.en)
                texts.filter { cjk.containsMatchIn(it) }.forEach { bad += "${d.id}: CJK in EN lesson text '${it.take(30)}'" }
            }
            d.about?.let { ab ->
                ab.pages.flatMap { listOf(it.title.en, it.body.en) }.filter { cjk.containsMatchIn(it) }
                    .forEach { bad += "${d.id}: CJK in EN background text '${it.take(30)}'" }
                ab.pages.flatMap { listOf(it.title.zh, it.body.zh) }.filter { it.isBlank() }.forEach { bad += "${d.id}: empty background text" }
            }
            Lang.set(false)
            d.update(1f / 60f)
            if (!cjk.containsMatchIn(d.readoutText)) bad += "${d.id}: ZH readout has no Chinese — switch did not switch back"
            println("I18N ${d.id}: EN ok=${bad.none { it.startsWith(d.id) }}")
        }
        bad.forEach { println("I18N-BAD $it") }
        assertTrue("i18n problems:\n" + bad.take(25).joinToString("\n"), bad.isEmpty())
    }

    @Test
    fun everyChineseLiteralGoesThroughTr() {
        val roots = listOf("src/main/java/com/bytedance/ailab/demos", "src/main/java/com/bytedance/ailab/content",
            "src/main/java/com/bytedance/ailab/lab").map { File(it) }
        val lit = Regex("\"(?:[^\"\\\\\\n]|\\\\.)*\"")
        val bad = ArrayList<String>()
        for (f in roots.flatMap { it.walkTopDown().filter { f -> f.extension == "kt" }.toList() }) {
            if (f.name == "I18n.kt") continue
            f.readLines().forEachIndexed { i, line ->
                val code = line.substringBefore("//").trim()
                if (code.startsWith("*") || code.startsWith("/*") || line.trimStart().startsWith("//") || line.contains("i18n-ok") || line.contains("TapTarget(") || line.contains("Bi(") || code.contains("Log.")) return@forEachIndexed
                for (m in lit.findAll(line.substringBefore(" //"))) {
                    if (!cjk.containsMatchIn(m.value)) continue
                    val before = line.substring(0, m.range.first).trimEnd()
                    if (!before.endsWith("tr(")) bad += "${f.name}:${i + 1}: ${m.value.take(40)}"
                }
            }
        }
        bad.forEach { println("I18N-SRC $it") }
        assertTrue("${bad.size} Chinese literals not wrapped in tr(\"中文\", \"English\"):\n" + bad.take(30).joinToString("\n"), bad.isEmpty())
    }
}
