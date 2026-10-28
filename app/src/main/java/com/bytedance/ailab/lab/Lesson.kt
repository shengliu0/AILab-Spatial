// Lesson — the guided track ("导学") every demo can carry: 3-5 steps of predict -> do -> observe ->
// explain, then an optional challenge. The user can skip, collapse or restart it at any time; free play
// is never blocked.
//
// A step completes by itself: PREDICT steps when an option is picked, every other step when its [done]
// predicate turns true (polled at 10 Hz). While a step waits, its [target] node's halo pings every
// 2.5 s so the eye finds where to act. [autoplay] is what a user would do — LessonFlowTest drives every
// lesson end to end with it, so no lesson can get stuck.
//
// Text is written Bi("中文", "English") — one literal pair per line, like TapTarget (I18nTest exempts
// `Bi(` lines and checks the English halves at runtime).

package com.bytedance.ailab.lab

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

/** A bilingual string; [text] follows the display language. */
data class Bi(val zh: String, val en: String) { val text: String get() = tr(zh, en) }

class LessonStep(
    /** The instruction on the card. */
    val say: Bi,
    /** Node whose halo pings while the step waits (null = none). */
    val target: String? = null,
    /** Non-empty = a PREDICT step: the user picks one option, which completes the step. */
    val options: List<Bi> = emptyList(),
    /** Index of the correct option (for the later reveal); -1 = an open question. */
    val answer: Int = -1,
    /** Shown for [LessonRunner.INSIGHT_S] when the step completes. */
    val insight: Bi? = null,
    /** Prefix the insight with the verdict on the most recent prediction ("你的预测：…"). */
    val reveal: Boolean = false,
    /** Completion predicate (ignored for PREDICT steps). */
    val done: () -> Boolean = { false },
    /** What a user would do to complete it — used by LessonFlowTest. */
    val autoplay: () -> Unit = {},
    /** Called when the step becomes the current one (e.g. remember a value to compare against). */
    val onEnter: () -> Unit = {},
) { val isPredict get() = options.isNotEmpty() }

class Challenge(
    val say: Bi,
    /** Live progress / score line, e.g. "误差 0.08 mL". */
    val progress: () -> String,
    val done: () -> Boolean,
    val autoplay: () -> Unit = {},
    /** Called when the challenge begins (e.g. remember a baseline). */
    val onEnter: () -> Unit = {},
)

class Lesson(val steps: List<LessonStep>, val challenge: Challenge? = null)

/** Runs the current demo's lesson. Compose-observed state; ticked by LabHost (main thread). */
object LessonRunner {
    enum class Phase { STEP, INSIGHT, CHALLENGE, FINISHED, NONE }

    const val INSIGHT_S = 4.5f
    private const val PING_S = 2.5f

    var lesson: Lesson? = null
        private set
    var index by mutableIntStateOf(0)
        private set
    var phase by mutableStateOf(Phase.NONE)
        private set
    var collapsed by mutableStateOf(false)
    var insightText by mutableStateOf("")
        private set
    var challengeText by mutableStateOf("")
        private set
    /** Bumped when anything shown changes, so the card recomposes. */
    var epoch by mutableIntStateOf(0)
        private set

    /** (picked, answer) of the most recent PREDICT step. */
    private var lastPrediction: Pair<Int, Int>? = null
    private var lastOptions: List<Bi> = emptyList()
    private var timer = 0f
    private var poll = 0f
    private var ping = 0f
    private var onPing: (String) -> Unit = {}

    val step: LessonStep? get() = lesson?.steps?.getOrNull(index)
    val total: Int get() = lesson?.steps?.size ?: 0

    fun start(l: Lesson?, ping: (String) -> Unit = {}) {
        lesson = l; onPing = ping
        index = 0; timer = 0f; poll = 0f; this.ping = PING_S - 0.8f
        lastPrediction = null; lastOptions = emptyList(); insightText = ""; challengeText = ""
        phase = if (l == null || l.steps.isEmpty()) Phase.NONE else Phase.STEP
        if (phase == Phase.STEP) runCatching { l!!.steps[0].onEnter() }
        epoch++
    }

    fun restart() { start(lesson, onPing); collapsed = false }

    /** Review only (UiPreviewTest): show step [i] — its instruction, or its insight when [insight] — or the challenge (i = total). */
    internal fun previewAt(i: Int, insight: Boolean) {
        val l = lesson ?: return
        index = i.coerceAtMost(l.steps.size); collapsed = false
        val s = l.steps.getOrNull(i)
        when {
            s == null -> { phase = if (l.challenge != null) Phase.CHALLENGE else Phase.FINISHED; challengeText = runCatching { l.challenge?.progress?.invoke() }.getOrNull() ?: "" }
            insight && s.insight != null -> { phase = Phase.INSIGHT; insightText = (if (s.reveal) verdict().let { if (it.isEmpty()) "" else it + "\n" } else "") + s.insight.text }
            else -> phase = Phase.STEP
        }
        epoch++
    }

    /** The user picked option [i] of the current PREDICT step. */
    fun choose(i: Int) {
        val s = step ?: return
        if (phase != Phase.STEP || !s.isPredict) return
        lastPrediction = i to s.answer; lastOptions = s.options
        complete(s)
    }

    fun skip() {
        when (phase) {
            Phase.STEP, Phase.INSIGHT -> advance()
            Phase.CHALLENGE -> { phase = Phase.FINISHED; epoch++ }
            else -> {}
        }
    }

    fun tick(dt: Float) {
        val l = lesson ?: return
        when (phase) {
            Phase.STEP -> {
                val s = step ?: return
                poll += dt
                if (poll >= 0.1f) {
                    poll = 0f
                    if (!s.isPredict && runCatching { s.done() }.getOrDefault(false)) { complete(s); return }
                }
                ping += dt
                if (ping >= PING_S && s.target != null && !collapsed) { ping = 0f; onPing(s.target) }
            }
            Phase.INSIGHT -> { timer += dt; if (timer >= INSIGHT_S) advance() }
            Phase.CHALLENGE -> {
                val c = l.challenge ?: return
                poll += dt
                if (poll >= 0.1f) {
                    poll = 0f
                    val p = runCatching { c.progress() }.getOrDefault("")
                    if (p != challengeText) { challengeText = p; epoch++ }
                    if (runCatching { c.done() }.getOrDefault(false)) { phase = Phase.FINISHED; epoch++ }
                }
            }
            else -> {}
        }
    }

    private fun complete(s: LessonStep) {
        val ins = s.insight
        if (ins == null) { advance(); return }
        insightText = (if (s.reveal) verdict().let { if (it.isEmpty()) "" else it + "\n" } else "") + ins.text
        phase = Phase.INSIGHT; timer = 0f; epoch++
    }

    private fun verdict(): String {
        val (picked, answer) = lastPrediction ?: return ""
        val p = lastOptions.getOrNull(picked)?.text ?: "?"
        if (answer < 0) return tr("你的预测：$p", "Your prediction: $p")
        val a = lastOptions.getOrNull(answer)?.text ?: "?"
        return if (picked == answer) tr("你的预测：$p —— 猜对了 ✓", "Your prediction: $p — right ✓")
               else tr("你的预测：$p —— 实际是：$a", "Your prediction: $p — actually: $a")
    }

    private fun advance() {
        val l = lesson ?: return
        index++
        timer = 0f; poll = 0f; ping = PING_S - 0.8f
        phase = when {
            index < l.steps.size -> Phase.STEP
            l.challenge != null -> Phase.CHALLENGE
            else -> Phase.FINISHED
        }
        if (phase == Phase.STEP) runCatching { l.steps[index].onEnter() }
        if (phase == Phase.CHALLENGE) runCatching { l.challenge?.onEnter?.invoke() }
        epoch++
    }
}
