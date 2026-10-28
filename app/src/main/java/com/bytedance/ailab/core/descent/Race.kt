// The race: three optimisers (SGD, momentum, Adam) start from the same point of the same landscape with the
// same learning rate and step in lockstep. Pure Kotlin; the demo only draws what this records.

package com.bytedance.ailab.core.descent

import kotlin.math.abs
import kotlin.math.hypot

/** One ball: an optimiser walking a landscape, with its full history (for the trail and the loss numbers). */
class Racer(val kind: Kind, private val seed: Long) {
    val opt = Optimizer(kind)
    private var rng = Rng(seed)
    /** Current parameters (x, y). After a divergence: the point where it left the drawn box. */
    val theta = DoubleArray(2)
    private val g = DoubleArray(2)
    private val prev = DoubleArray(2)

    var steps = 0; private set
    /** Left the drawn landscape (or overflowed): stops there. */
    var diverged = false; private set
    /** First step whose loss is within [HIT] of the minimum (−1 = not yet). */
    var firstHit = -1; private set
    /** Consecutive steps that moved less than 1e-10 (= standing still). */
    private var still = 0

    val hx = DoubleArray(MAX_STEPS + 1)
    val hy = DoubleArray(MAX_STEPS + 1)
    val hf = DoubleArray(MAX_STEPS + 1)
    /** Number of recorded points (= steps + 1). */
    val count get() = steps + 1
    val loss get() = hf[steps]
    /** Nothing more will happen: diverged, settled, or out of steps. */
    val done get() = diverged || still >= 5 || steps >= MAX_STEPS

    fun reset(s: LossSurface, x: Double, y: Double) {
        opt.reset(); rng = Rng(seed)
        theta[0] = x; theta[1] = y
        steps = 0; diverged = false; firstHit = -1; still = 0
        hx[0] = x; hy[0] = y; hf[0] = s.f(x, y)
        if (hf[0] - s.fMin < HIT) firstHit = 0
    }

    /** One optimiser step with learning rate [lr] and gradient noise [noise] (0..1). */
    fun step(s: LossSurface, lr: Double, noise: Double) {
        if (done) return
        prev[0] = theta[0]; prev[1] = theta[1]
        s.noisyGrad(theta[0], theta[1], noise, rng, g)
        opt.step(theta, g, lr)
        steps++
        if (!theta[0].isFinite() || !theta[1].isFinite() || !s.inside(theta[0], theta[1])) {
            diverged = true
            clipToBox(s, prev, theta)
        } else if (abs(theta[0] - prev[0]) + abs(theta[1] - prev[1]) < 1e-10) still++ else still = 0
        hx[steps] = theta[0]; hy[steps] = theta[1]; hf[steps] = s.f(theta[0], theta[1])
        if (firstHit < 0 && !diverged && hf[steps] - s.fMin < HIT) firstHit = steps
    }

    /** Lowest loss among steps 0..[n]. */
    fun bestWithin(n: Int): Double { var b = Double.MAX_VALUE; for (k in 0..minOf(n, steps)) b = minOf(b, hf[k]); return b }

    /** Exact gradient at the current point (the arrow). */
    fun gradHere(s: LossSurface, out: DoubleArray) = s.grad(theta[0], theta[1], out)

    companion object {
        const val MAX_STEPS = 1500
        /** "Reached the bottom": loss within 1e-3 of the minimum. */
        const val HIT = 1e-3

        /** Move [b] back along a→b to where the segment leaves the box (a is inside). */
        fun clipToBox(s: LossSurface, a: DoubleArray, b: DoubleArray) {
            var t = 1.0
            val dx = b[0] - a[0]; val dy = b[1] - a[1]
            if (!dx.isFinite() || !dy.isFinite()) { b[0] = a[0]; b[1] = a[1]; return }
            if (dx > 0) t = minOf(t, (s.x1 - a[0]) / dx) else if (dx < 0) t = minOf(t, (s.x0 - a[0]) / dx)
            if (dy > 0) t = minOf(t, (s.y1 - a[1]) / dy) else if (dy < 0) t = minOf(t, (s.y0 - a[1]) / dy)
            t = t.coerceIn(0.0, 1.0)
            b[0] = (a[0] + t * dx).coerceIn(s.x0, s.x1); b[1] = (a[1] + t * dy).coerceIn(s.y0, s.y1)
        }
    }
}

/** Three balls, one landscape, one start, one learning rate. */
class Race(seed: Long = 1847L) {
    val racers = Kind.entries.map { Racer(it, seed + 7919L * it.ordinal) }
    /** Which balls take part (pinch a ball to toggle). */
    val active = booleanArrayOf(true, true, true)
    var surface: LossSurface = Surfaces.BOWL; private set
    var startX = surface.startX; private set
    var startY = surface.startY; private set
    var lr = surface.defaultLr
    var noise = 0.0
    /** Lockstep steps since the start. */
    var steps = 0; private set

    fun setSurface(s: LossSurface) { surface = s; startX = s.startX; startY = s.startY; lr = s.defaultLr; reset() }
    fun setStart(x: Double, y: Double) {
        startX = x.coerceIn(surface.x0, surface.x1); startY = y.coerceIn(surface.y0, surface.y1); reset()
    }
    fun reset() { steps = 0; for (r in racers) r.reset(surface, startX, startY) }

    fun step() {
        if (allDone) return
        for ((i, r) in racers.withIndex()) if (active[i]) r.step(surface, lr, noise)
        steps++
    }

    /** Every racing ball has diverged, settled or run out of steps (or none is racing). */
    val allDone get() = racers.indices.none { active[it] && !racers[it].done }

    /** Distance of the start from the nearest minimum (in landscape units). */
    val startToMin get() = surface.nearestMin(startX, startY).let { hypot(it[0] - startX, it[1] - startY) }
}
