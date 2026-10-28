package com.bytedance.ailab.lab

import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Every guided track (导学) can be completed: for each demo with a lesson, play it end to end the way a user
 * would — pick the right answer on PREDICT steps, run each step's [LessonStep.autoplay], then simulate (60 Hz,
 * LessonRunner ticking) until the step completes — then the challenge. A step that does not complete within
 * [LIMIT_S] simulated seconds fails the test: a lesson must never get stuck.
 */
class LessonFlowTest {
    private val LIMIT_S = 90f

    @Test
    fun everyLessonCanBeCompleted() {
        val cache = File("build/review-cache").apply { mkdirs() }
        val bad = ArrayList<String>()
        var lessons = 0
        for (d in DemoRegistry.all.filterIsInstance<SpecDemo>()) {
            val l = d.lesson ?: continue
            lessons++
            d.runHeadless(cache, 1f)
            LessonRunner.start(l)
            val dt = 1f / 60f
            fun simulateUntil(cond: () -> Boolean): Boolean {
                var t = 0f
                while (t < LIMIT_S) { if (cond()) return true; d.update(dt); LessonRunner.tick(dt); t += dt }
                return cond()
            }
            for ((i, s) in l.steps.withIndex()) {
                check(LessonRunner.index == i)
                if (s.isPredict) LessonRunner.choose(if (s.answer >= 0) s.answer else 0)
                else {
                    s.autoplay()
                    if (!simulateUntil { LessonRunner.phase != LessonRunner.Phase.STEP || LessonRunner.index != i }) { bad += "${d.id}: step ${i + 1} '${s.say.en.take(40)}' never completes"; break }
                }
                // let the insight run out
                simulateUntil { LessonRunner.index != i || LessonRunner.phase == LessonRunner.Phase.CHALLENGE || LessonRunner.phase == LessonRunner.Phase.FINISHED }
            }
            val c = l.challenge
            if (c != null && bad.none { it.startsWith(d.id) }) {
                c.autoplay()
                if (!simulateUntil { LessonRunner.phase == LessonRunner.Phase.FINISHED }) bad += "${d.id}: challenge '${c.say.en.take(40)}' never completes"
            }
            println("LESSON ${d.id}: ${l.steps.size} steps${if (c != null) " + challenge" else ""} -> ${if (bad.none { it.startsWith(d.id) }) "completed" else "STUCK"}")
        }
        println("LESSONS $lessons, problems: ${if (bad.isEmpty()) "none" else bad.joinToString("; ")}")
        assertTrue(bad.joinToString("\n"), bad.isEmpty())
    }
}
