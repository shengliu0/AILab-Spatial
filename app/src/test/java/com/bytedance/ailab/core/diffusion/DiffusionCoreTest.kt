package com.bytedance.ailab.core.diffusion

import com.bytedance.ailab.lab.LabAssets
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * The mini DDPM (core/diffusion), checked against the PyTorch model it was trained as (tools/train_diffusion.py
 * exports assets/diffusion/testvec.bin from the SAME weights the app ships) and against the closed forms of
 * Ho, Jain & Abbeel 2020:
 *   - eps_hat(x, t, dataset): Kotlin == PyTorch float32 within 1e-4 (48 vectors, t = 1 .. T);
 *   - a whole 50-step DDIM run (deterministic, 50 network calls per point) ends where PyTorch's does (1e-4);
 *   - the schedule is the stated one and abar_t = prod (1 - beta_s);
 *   - the simulated forward Markov chain has the closed-form marginal N(sqrt(abar_t) x0, (1 - abar_t) I);
 *   - quality (min of precision and coverage, EPS 0.15) is high with enough steps and collapses with 2;
 *   - fixed seed -> identical samples; chunking the work differently -> identical samples.
 */
class DiffusionCoreTest {
    private val model by lazy { DiffusionModel(LabAssets.bytes("diffusion/model.bin")) }

    private fun testvec(): ByteBuffer = ByteBuffer.wrap(LabAssets.bytes("diffusion/testvec.bin")).order(ByteOrder.LITTLE_ENDIAN)

    @Test
    fun epsMatchesPyTorch() {
        val b = testvec()
        val nv = b.int
        var worst = 0f
        for (v in 0 until nv) {
            val d = b.int; val t = b.int
            val x = b.float; val y = b.float; val ex = b.float; val ey = b.float
            val e = model.epsAll(d, t, floatArrayOf(x, y))
            worst = maxOf(worst, abs(e[0] - ex), abs(e[1] - ey))
        }
        println("DIFFUSION parity: $nv eps vectors, max |Kotlin - PyTorch| = %.2e".format(worst))
        assertTrue("eps_hat differs from PyTorch by $worst", worst < 1e-4f)
    }

    @Test
    fun ddimRunMatchesPyTorch() {
        val b = testvec()
        val nv = b.int; repeat(nv) { b.int; b.int; repeat(4) { b.float } }
        val ns = b.int; val s = b.int
        var worst = 0f
        for (d in 0 until model.D) {
            val start = FloatArray(ns * 2) { b.float }
            val want = FloatArray(ns * 2) { b.float }
            val run = ReverseRun(model, d, s, ddim = true, n = ns, seed = 1L, start = start)
            run.runToEnd()
            val got = run.result()
            for (i in 0 until ns * 2) worst = maxOf(worst, abs(got[i] - want[i]))
        }
        println("DIFFUSION parity: DDIM $s steps x $ns points x ${model.D} datasets, max end-point |Kotlin - PyTorch| = %.2e".format(worst))
        assertTrue("DDIM end points differ from PyTorch by $worst", worst < 1e-4f)
    }

    @Test
    fun scheduleIsTheStatedOne() {
        // abar_t = 1 - (1 - 1e-3)(t/T)^2 (sigma_t linear in t); abar_t = prod_{s<=t} (1 - beta_s) (Ho et al. eq. 4)
        var prod = 1.0
        var worst = 0.0
        for (t in 0..model.T) {
            val want = 1.0 - (1.0 - 1e-3) * (t.toDouble() / model.T).let { it * it }
            worst = maxOf(worst, abs(model.abar[t] - want))
            if (t > 0) { prod *= 1 - model.beta(t); worst = maxOf(worst, abs(prod - model.abar[t])) }
        }
        assertEquals(1f, model.abar[0], 0f)
        assertTrue("schedule off by $worst", worst < 1e-6)
        assertEquals(1.0, model.sigma(model.T).toDouble(), 1e-3)
        assertEquals(0.5, model.sigma(model.T / 2).toDouble(), 1e-3)
        println("DIFFUSION schedule: T=${model.T}, sigma_T/2 = %.4f, abar_T = %.4f, max deviation %.1e".format(model.sigma(model.T / 2), model.abar[model.T], worst))
    }

    @Test
    fun forwardChainHasTheClosedFormMarginal() {
        // 20 000 independent random walks from one point; at t the cloud must be N(sqrt(abar_t) x0, (1 - abar_t) I)
        val n = 20000
        val x0 = FloatArray(n * 2) { if (it % 2 == 0) 1.2f else -0.7f }
        val ch = ForwardChain(model, x0, seed = 2020L)
        for (t in intArrayOf(10, 50, 100, 150, 200)) {
            var mx = 0.0; var my = 0.0
            for (i in 0 until n) { mx += ch.x(t, i); my += ch.y(t, i) }
            mx /= n; my /= n
            var vx = 0.0; var vy = 0.0
            for (i in 0 until n) { vx += (ch.x(t, i) - mx).let { it * it }; vy += (ch.y(t, i) - my).let { it * it } }
            vx /= n - 1; vy /= n - 1
            val sa = sqrt(model.abar[t].toDouble()); val v = 1.0 - model.abar[t]
            val se = sqrt(v / n)
            println("DIFFUSION forward t=%3d: mean (%.4f, %.4f) vs (%.4f, %.4f)   var (%.4f, %.4f) vs %.4f".format(t, mx, my, sa * 1.2, -sa * 0.7, vx, vy, v))
            assertEquals("mean x at t=$t", sa * 1.2, mx, 4 * se + 1e-6)
            assertEquals("mean y at t=$t", -sa * 0.7, my, 4 * se + 1e-6)
            // sample variance has relative standard error sqrt(2/n) = 1 %
            assertEquals("var x at t=$t", v, vx, 0.05 * v)
            assertEquals("var y at t=$t", v, vy, 0.05 * v)
        }
    }

    @Test
    fun qualityMetricBasics() {
        val ref = model.ref[0]
        val q = Quality(ref, model.nCov)
        // the reference points themselves: perfect
        assertEquals(1f, q.score(ref.copyOf(model.nCov * 2)).q, 0f)
        // the same points pushed 2 EPS off along x... are still near OTHER points of a curve, so use a far shift
        val far = FloatArray(600) { if (it % 2 == 0) 9f else 9f }
        assertEquals(0f, q.score(far).precision, 0f)
        // one point only: precision 1, coverage tiny -> quality tiny (collapse does not score)
        val one = q.score(floatArrayOf(ref[0], ref[1]))
        assertEquals(1f, one.precision, 0f)
        assertTrue(one.q < 0.05f)
    }

    private fun gen(d: Int, steps: Int, ddim: Boolean, seed: Long): Quality.Score {
        val run = ReverseRun(model, d, steps, ddim, 300, seed)
        run.runToEnd()
        return Quality(model.ref[d], model.nCov).score(run.result())
    }

    @Test
    fun qualityRisesWithSteps() {
        val names = listOf("spiral", "moons", "glyph")
        val table = HashMap<String, Float>()
        for (d in 0 until model.D) {
            val sb = StringBuilder("DIFFUSION quality %-6s".format(names[d]))
            val noise = Rng(9).let { r -> FloatArray(600) { r.gauss().toFloat() } }
            sb.append("  noise %3.0f%%".format(Quality(model.ref[d], model.nCov).score(noise).q * 100))
            for (ddim in listOf(false, true)) for (s in intArrayOf(2, 5, 10, 50, 200)) {
                // mean of 3 seeds (each run: 300 points)
                val qs = (0 until 3).map { gen(d, s, ddim, 100L + it).q }
                val m = qs.average().toFloat()
                table["$d/$ddim/$s"] = m
                sb.append("  %s%-3d %3.0f%%".format(if (ddim) "DDIM" else "DDPM", s, m * 100))
            }
            println(sb)
        }
        // the demo's claims (stated in its UI and lesson), spiral:
        assertTrue("200 DDPM steps should draw the spiral", table["0/false/200"]!! >= 0.95f)
        assertTrue("10 DDPM steps are nearly as good", table["0/false/10"]!! >= 0.90f)
        assertTrue("5 steps pass the 80 % challenge", table["0/false/5"]!! >= 0.80f && table["0/true/5"]!! >= 0.80f)
        assertTrue("2 steps collapse", table["0/false/2"]!! < 0.65f && table["0/true/2"]!! < 0.65f)
        for (d in 0 until model.D) assertTrue("dataset $d at 200 steps", table["$d/false/200"]!! >= 0.9f)
    }

    @Test
    fun deterministicAndChunkIndependent() {
        val a = ReverseRun(model, 0, 50, false, 300, 77L).also { it.runToEnd() }.result()
        val b = ReverseRun(model, 0, 50, false, 300, 77L).also { r -> while (!r.finished) r.advance(37) }.result()
        val c = ReverseRun(model, 0, 50, false, 300, 78L).also { it.runToEnd() }.result()
        assertTrue("same seed, different chunking -> same samples", a.contentEquals(b))
        assertTrue("different seed -> different samples", !a.contentEquals(c))
        val f1 = ForwardChain(model, model.ref[1].copyOf(600), 5L).chain
        val f2 = ForwardChain(model, model.ref[1].copyOf(600), 5L).chain
        assertTrue(f1.contentEquals(f2))
    }

    @Test
    fun budget() {
        val f = File("src/main/assets/diffusion/model.bin")
        assertTrue("asset over 2 MB", f.length() <= 2_000_000)
        // warm up, then time a full 200-step DDPM generation of 300 points (60 000 network calls)
        repeat(2) { ReverseRun(model, 0, 50, false, 300, 1L).runToEnd() }
        val t0 = System.nanoTime()
        val r = ReverseRun(model, 0, 200, false, 300, 1L); r.runToEnd()
        val ms = (System.nanoTime() - t0) / 1e6
        println("DIFFUSION cost: ${r.evals} network calls (${model.macsPerPoint} MAC each) in %.0f ms = %.2f us/call; asset %d bytes".format(ms, ms * 1000 / r.evals, f.length()))
    }
}
