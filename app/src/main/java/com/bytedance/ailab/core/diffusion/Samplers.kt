// Samplers — the forward (noising) chain and the reverse (denoising) samplers, pure Kotlin.
//
// FORWARD: the Markov chain of DDPM, x_t = sqrt(1 - beta_t) x_{t-1} + sqrt(beta_t) z_t, simulated step by step
// (a real random walk, not the closed form), for every displayed particle. Its marginal at t is the closed form
// N(sqrt(abar_t) x_0, (1 - abar_t) I) — DiffusionCoreTest checks this by Monte Carlo.
//
// REVERSE, on a respaced grid tau_i = round(T (S - i) / S), i = 0..S (S = number of sampling steps):
//   DDPM (Ho et al. 2020, ancestral, posterior variance), a = abar_tau_i, a' = abar_tau_{i+1}, b = 1 - a/a':
//       x' = (x - b / sqrt(1 - a) eps_hat) / sqrt(1 - b) + sqrt(b (1 - a') / (1 - a)) z
//   DDIM (Song, Meng & Ermon 2020, eta = 0, deterministic):
//       x0_hat = (x - sqrt(1 - a) eps_hat) / sqrt(a);   x' = sqrt(a') x0_hat + sqrt(1 - a') eps_hat
// Exactly the samplers of tools/train_diffusion.py. The work is INCREMENTAL ([ReverseRun.advance] does at most
// a given number of point-evaluations), so the app spreads a 200-step run over many frames; the result does
// not depend on how the work is chunked (each step's noise is drawn up front, in a fixed order).

package com.bytedance.ailab.core.diffusion

import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sqrt

/** SplitMix64 + Box-Muller: a tiny deterministic Gaussian source (same numbers on every JVM / ART). */
class Rng(seed: Long) {
    private var s = seed
    private var spare = Double.NaN
    fun nextLong(): Long {
        s += -0x61c8864680b583ebL
        var z = s
        z = (z xor (z ushr 30)) * -0x40a7b892e31b1a47L
        z = (z xor (z ushr 27)) * -0x6b2fb644ecceee15L
        return z xor (z ushr 31)
    }
    /** Uniform in (0, 1). */
    fun uniform(): Double = ((nextLong() ushr 11) + 0.5) * (1.0 / (1L shl 53))
    fun gauss(): Double {
        if (!spare.isNaN()) { val g = spare; spare = Double.NaN; return g }
        val r = sqrt(-2.0 * ln(uniform())); val a = 2.0 * Math.PI * uniform()
        spare = r * kotlin.math.sin(a)
        return r * cos(a)
    }
}

/** The respaced timestep grid T = tau_0 > tau_1 > ... > tau_S = 0 (as numpy round(linspace(T, 0, S+1))). */
fun timeGrid(T: Int, S: Int): IntArray = IntArray(S + 1) { i -> Math.rint(T.toDouble() * (S - i) / S).toInt() }

/**
 * One generation: [n] points start as pure noise N(0, I) at t = T and are denoised in [steps] steps.
 * Positions after every step are kept ([pos]) so the app can scrub the trajectory back and forth.
 */
class ReverseRun(val model: DiffusionModel, val dataset: Int, val steps: Int, val ddim: Boolean, val n: Int, seed: Long,
                 start: FloatArray? = null) {
    val tau: IntArray = timeGrid(model.T, steps)
    /** traj[(k * n + i) * 2] = point i after k steps (k = 0: the initial noise at t = T). */
    val traj = FloatArray((steps + 1) * n * 2)
    /** Steps fully computed (positions known for k = 0..stepsDone). */
    var stepsDone = 0; private set
    /** Network calls so far (one per point per step). */
    var evals = 0L; private set
    /** Wall time spent inside [advance] (ns) — what the app shows as the cost of the run. */
    var computeNs = 0L; private set
    val finished get() = stepsDone >= steps

    private val rng = Rng(seed)
    private val noise = FloatArray(n * 2)
    private val epsBuf = FloatArray(n * 2)
    private val cur = FloatArray(n * 2)
    private val scratch = model.Scratch()
    private var cond: FloatArray? = null
    private var cursor = 0          // points of the current step already evaluated

    init {
        // the starting noise x_T ~ N(0, I) (or a given one: the PyTorch parity test)
        for (i in 0 until n * 2) traj[i] = start?.get(i) ?: rng.gauss().toFloat()
    }

    /** Do at most [maxEvals] point-evaluations of work; returns how many were done. */
    fun advance(maxEvals: Int): Int {
        val t0 = System.nanoTime()
        var done = 0
        while (!finished && done < maxEvals) {
            val k = stepsDone
            if (cursor == 0) {
                System.arraycopy(traj, k * n * 2, cur, 0, n * 2)
                cond = model.cond(dataset, tau[k])
                if (!ddim) for (i in 0 until n * 2) noise[i] = rng.gauss().toFloat()
            }
            val m = minOf(DiffusionModel.CHUNK, n - cursor, maxEvals - done)
            model.eps(cond!!, cur, cursor, m, epsBuf, scratch)
            cursor += m; done += m; evals += m
            if (cursor == n) { finishStep(k); cursor = 0; stepsDone++ }
        }
        computeNs += System.nanoTime() - t0
        return done
    }

    /** Run to the end (tests, review). */
    fun runToEnd() { while (!finished) advance(Int.MAX_VALUE) }

    private fun finishStep(k: Int) {
        val a = model.abar[tau[k]].toDouble(); val ap = model.abar[tau[k + 1]].toDouble()
        val o = (k + 1) * n * 2
        if (ddim) {
            val sa = sqrt(a); val s1a = sqrt(1 - a); val sap = sqrt(ap); val s1ap = sqrt(1 - ap)
            for (i in 0 until n * 2) {
                val x0 = (cur[i] - s1a * epsBuf[i]) / sa
                traj[o + i] = (sap * x0 + s1ap * epsBuf[i]).toFloat()
            }
        } else {
            val b = 1 - a / ap
            val c1 = b / sqrt(1 - a); val c0 = 1 / sqrt(1 - b)
            val sd = sqrt((b * (1 - ap) / (1 - a)).coerceAtLeast(0.0))
            for (i in 0 until n * 2) traj[o + i] = ((cur[i] - c1 * epsBuf[i]) * c0 + sd * noise[i]).toFloat()
        }
    }

    /** Point i after k steps. */
    fun x(k: Int, i: Int) = traj[(k * n + i) * 2]
    fun y(k: Int, i: Int) = traj[(k * n + i) * 2 + 1]
    /** Final samples (valid when [finished]). */
    fun result(): FloatArray = traj.copyOfRange(steps * n * 2, (steps + 1) * n * 2)
}

/**
 * The forward Markov chain for [n] particles starting at [x0] (2n floats): positions at every t = 0..T.
 * chain[(t * n + i) * 2] = particle i at time t.
 */
class ForwardChain(val model: DiffusionModel, x0: FloatArray, seed: Long) {
    val n = x0.size / 2
    val chain = FloatArray((model.T + 1) * n * 2)
    init {
        System.arraycopy(x0, 0, chain, 0, n * 2)
        val rng = Rng(seed)
        for (t in 1..model.T) {
            val b = model.beta(t); val keep = sqrt(1 - b); val kick = sqrt(b)
            val o = t * n * 2; val p = (t - 1) * n * 2
            for (i in 0 until n * 2) chain[o + i] = (keep * chain[p + i] + kick * rng.gauss()).toFloat()
        }
    }
    fun x(t: Int, i: Int) = chain[(t * n + i) * 2]
    fun y(t: Int, i: Int) = chain[(t * n + i) * 2 + 1]
}
