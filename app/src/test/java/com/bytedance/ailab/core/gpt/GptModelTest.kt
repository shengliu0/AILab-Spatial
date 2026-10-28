package com.bytedance.ailab.core.gpt

import com.bytedance.ailab.lab.LabAssets
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln

/**
 * The demo's GPT, checked against REAL numbers:
 *  - the Kotlin forward pass reproduces PyTorch (float64 forward of the shipped float32 weights, exported by
 *    tools/train_gpt.py to src/test/resources/gpt/ref.bin) for logits at every position and every attention
 *    weight, within 1e-4;
 *  - attention rows are probability distributions over the PAST only (causal mask);
 *  - temperature: entropy grows monotonically with T (dH/dT = Var(z)/T^3 >= 0), T -> 0 sampling = argmax;
 *  - task metric: validation perplexity on held-out generated text beats a unigram baseline by a wide
 *    margin, and equals what PyTorch printed.
 */
class GptModelTest {
    private val model by lazy { GptModel(LabAssets.bytes("gpt/model.bin")) }
    private fun res(name: String) = javaClass.classLoader!!.getResource("gpt/$name")!!.readBytes()

    private class Ref(val ids: IntArray, val logits: FloatArray, val att: FloatArray)
    private val refs by lazy {
        val b = ByteBuffer.wrap(res("ref.bin")).order(ByteOrder.LITTLE_ENDIAN)
        val m = ByteArray(4).also { b.get(it) }; check(String(m) == "GREF")
        val V = model.V; val nl = model.nLayer; val nh = model.nHead
        List(b.int) {
            val n = b.int
            val ids = IntArray(n) { b.int }
            Ref(ids, FloatArray(n * V) { b.float }, FloatArray(nl * nh * n * n) { b.float })
        }
    }

    @Test
    fun kotlinMatchesPyTorch() {
        var maxLogit = 0.0; var maxAtt = 0.0; var cases = 0
        for (r in refs) {
            val tr = model.forward(r.ids, allLogits = true)
            val n = r.ids.size
            for (t in 0 until n) for (v in 0 until model.V) maxLogit = maxOf(maxLogit, abs(tr.logits[t]!![v] - r.logits[t * model.V + v]))
            for (l in 0 until model.nLayer) for (h in 0 until model.nHead) for (i in 0 until n) for (j in 0 until n)
                maxAtt = maxOf(maxAtt, abs(tr.att[l][h][i][j] - r.att[((l * model.nHead + h) * n + i) * n + j]))
            cases++
        }
        println("GPT parity: $cases prompts, max |logit diff| = %.2e, max |attention diff| = %.2e".format(maxLogit, maxAtt))
        assertTrue(cases >= 5)
        assertTrue("logits differ from PyTorch by $maxLogit", maxLogit < 1e-4)
        assertTrue("attention differs from PyTorch by $maxAtt", maxAtt < 1e-4)
    }

    @Test
    fun attentionIsCausalAndNormalised() {
        val ids = model.encode("\n猫有三样东西：书、苹果、笔。猫最喜欢")
        val tr = model.forward(ids)
        var worstSum = 0.0; var future = 0.0
        for (l in 0 until model.nLayer) for (h in 0 until model.nHead) for (i in ids.indices) {
            val row = tr.att[l][h][i]
            worstSum = maxOf(worstSum, abs(row.sum() - 1.0))
            for (j in i + 1 until ids.size) future = maxOf(future, abs(row[j]))
            assertTrue(row.all { it >= 0.0 })
        }
        println("GPT attention: max |row sum - 1| = %.1e, max weight on a future token = %.1e".format(worstSum, future))
        assertTrue(worstSum < 1e-12)
        assertEquals(0.0, future, 0.0)
        // changing a FUTURE token cannot change the past (the mask, checked end to end)
        val a = model.forward(model.encode("\n猫把书给了狗。"), allLogits = true)
        val b = model.forward(model.encode("\n猫把书给了兔。"), allLogits = true)
        // "↵猫把书给了" (positions 0..5) is shared; the name at position 6 differs
        for (t in 0 until 6) for (v in 0 until model.V) assertEquals(a.logits[t]!![v], b.logits[t]!![v], 0.0)
        assertTrue((0 until model.V).any { a.logits[6]!![it] != b.logits[6]!![it] })
    }

    @Test
    fun temperatureFlattensMonotonically() {
        val ts = doubleArrayOf(0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 3.0, 5.0)
        for (prompt in listOf("\n狗问猫：「你有", "\n狗把", "\n猫有三样东西：书、苹果、笔。猫最喜欢", "\n今天是星期三，明天是星期")) {
            val z = model.forward(model.encode(prompt)).lastLogits
            val hs = ts.map { Sampling.entropy(Sampling.softmax(z, it)) }
            for (k in 1 until hs.size) assertTrue("$prompt: H(${ts[k]})=${hs[k]} < H(${ts[k - 1]})=${hs[k - 1]}", hs[k] >= hs[k - 1] - 1e-12)
            assertTrue(hs.last() > hs.first())
            // the entropy cannot exceed ln V (uniform)
            assertTrue(hs.last() <= ln(model.V.toDouble()) + 1e-12)
            println("GPT temperature ${prompt.trim().takeLast(8)}: " + ts.indices.joinToString { "H(%.2f)=%.3f".format(ts[it], hs[it]) })
        }
        // the closed form dH/dT = Var(z)/T^3, checked by central difference at T = 0.8
        val z = model.forward(model.encode("\n狗问猫：「你有")).lastLogits
        val t = 0.8; val e = 1e-5
        val num = (Sampling.entropy(Sampling.softmax(z, t + e)) - Sampling.entropy(Sampling.softmax(z, t - e))) / (2 * e)
        val p = Sampling.softmax(z, t); val mean = p.indices.sumOf { p[it] * z[it] }
        val varz = p.indices.sumOf { p[it] * (z[it] - mean) * (z[it] - mean) }
        assertEquals(varz / (t * t * t), num, 1e-5 * maxOf(1.0, num))
    }

    @Test
    fun zeroTemperatureSamplingIsArgmax() {
        val rng = java.util.Random(1)
        for (prompt in listOf("\n狗问猫：「你有", "\n", "\n数数：五六七", "\n狗说：「我喜欢花。")) {
            val z = model.forward(model.encode(prompt)).lastLogits
            val am = Sampling.argmax(z)
            for (t in doubleArrayOf(0.0, 1e-7, 1e-3)) repeat(50) { assertEquals(am, Sampling.sample(Sampling.softmax(z, t), rng.nextDouble())) }
            // and the T = 1 distribution is a probability vector whose top entry is the argmax
            val p = Sampling.softmax(z, 1.0)
            assertEquals(1.0, p.sum(), 1e-12)
            assertEquals(am, Sampling.topK(p, 3)[0])
        }
    }

    @Test
    fun perplexityBeatsUnigram() {
        val text = String(res("val.txt"), Charsets.UTF_8)
        val ids = model.encode(text)
        val cb = ByteBuffer.wrap(res("train_counts.bin")).order(ByteOrder.LITTLE_ENDIAN)
        val counts = LongArray(model.V) { cb.int.toLong() }
        val total = counts.sum()
        // the same scoring as train_gpt.py block_nll: consecutive blocks of ctx, each token sees its in-block prefix
        var nll = 0.0; var uni = 0.0; var n = 0
        var b = 0
        while (b < ids.size - 1) {
            val x = ids.copyOfRange(b, minOf(b + model.ctx, ids.size - 1))
            val tr = model.forward(x, allLogits = true)
            for (t in x.indices) {
                val y = ids[b + t + 1]
                val z = tr.logits[t]!!; val mx = z.max()
                var s = 0.0; for (v in z) s += exp(v - mx)
                nll -= z[y] - mx - ln(s)
                uni -= ln((counts[y] + 1.0) / (total + model.V))        // add-one smoothed unigram from the TRAIN text
                n++
            }
            b += model.ctx
        }
        val ppl = exp(nll / n); val uppl = exp(uni / n)
        println("GPT validation: %d held-out chars, model perplexity %.3f (%.3f nats/char) vs unigram %.2f; PyTorch reported %.3f / %.2f; %d params, trained on %d chars"
            .format(n, ppl, nll / n, uppl, model.reportedValPpl, model.reportedUnigramPpl, model.paramCount, model.trainChars))
        assertEquals(model.reportedValPpl.toDouble(), ppl, 1e-3 * ppl)
        assertEquals(model.reportedUnigramPpl.toDouble(), uppl, 1e-3 * uppl)
        assertTrue("model ppl $ppl not well below unigram $uppl", ppl < uppl / 10)
        // parameter count re-derived: embeddings V*d + ctx*d, per layer 12 d^2 + 13 d (dff = 4d), final LN 2d
        val d = model.d
        assertEquals(4 * d, model.dff)
        assertEquals(model.V * d + model.ctx * d + model.nLayer * (12 * d * d + 13 * d) + 2 * d, model.paramCount)
    }

    @Test
    fun forwardIsFastEnoughForOneStep() {
        val ids = model.encode("\n猫有三样东西：书、苹果、笔。猫最喜欢书。狗把")
        val full = IntArray(model.ctx) { ids[it % ids.size] }
        repeat(30) { model.forward(full) }                     // JIT warm-up
        val t0 = System.nanoTime(); val reps = 50
        repeat(reps) { model.forward(full) }
        val ms = (System.nanoTime() - t0) / 1e6 / reps
        println("GPT forward (ctx ${model.ctx}, all rows + attention): %.2f ms per step on this JVM".format(ms))
        assertTrue("forward $ms ms > 20 ms", ms < 20.0)
    }

    @Test
    fun glyphsCoverTheVocabulary() {
        val g = GlyphAtlas(LabAssets.bytes("gpt/glyphs.bin"))
        val missing = model.vocab.filter { !g.has(it) }
        assertTrue("no glyph for: $missing", missing.isEmpty())
        for (c in "0123456789.%<>T=H▶→…") assertTrue("no glyph for '$c'", g.has(c))
        var ink = 0; val cc = g.cell * g.cell; val k = g.indexOf('明')
        for (i in 0 until cc) if ((g.alpha[k * cc + i].toInt() and 0xFF) > 128) ink++
        println("GPT glyphs: ${g.size} cells of ${g.cell} px; '明' has $ink ink pixels")
        assertTrue(ink in 150..1500)
    }
}
