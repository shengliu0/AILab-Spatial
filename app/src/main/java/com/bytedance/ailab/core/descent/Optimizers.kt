// The three optimisers the demo races, exactly as torch.optim implements them (DescentCoreTest compares
// float64 trajectories with tools/train_descent.py to 1e-9), plus a seeded RNG for the gradient noise.

package com.bytedance.ailab.core.descent

import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

enum class Kind { SGD, MOMENTUM, ADAM }

/**
 * One optimiser's state for a 2-parameter problem.
 *
 *  SGD       θ ← θ − η·g
 *  MOMENTUM  v ← β·v + g ;  θ ← θ − η·v                        (heavy ball, Polyak 1964; torch.optim.SGD
 *            momentum=β, dampening 0 — identical to the textbook v ← βv − ηg, θ ← θ + v for a fixed η)
 *  ADAM      m ← β₁m + (1−β₁)g ;  v ← β₂v + (1−β₂)g² ;  m̂ = m/(1−β₁ᵗ) ;  v̂ = v/(1−β₂ᵗ) ;
 *            θ ← θ − η·m̂/(√v̂ + ε)                               (Kingma & Ba 2015, Algorithm 1)
 *
 * Arithmetic is ordered like torch's single-tensor kernels so the float64 trajectories agree to round-off.
 */
class Optimizer(val kind: Kind, val beta: Double = 0.9, val beta1: Double = 0.9, val beta2: Double = 0.999, val eps: Double = 1e-8) {
    /** Steps taken (Adam's t). */
    var t = 0; private set
    /** Momentum buffer v (MOMENTUM) or first moment m (ADAM). */
    val m = DoubleArray(2)
    /** Second moment (ADAM). */
    val v = DoubleArray(2)

    fun reset() { t = 0; m.fill(0.0); v.fill(0.0) }

    /** One update of [theta] (in place) with gradient [g] and learning rate [lr]. */
    fun step(theta: DoubleArray, g: DoubleArray, lr: Double) {
        t++
        when (kind) {
            Kind.SGD -> for (i in 0..1) theta[i] -= lr * g[i]
            Kind.MOMENTUM -> for (i in 0..1) {
                m[i] = if (t == 1) g[i] else m[i] * beta + g[i]        // torch: buf = clone(g) on the first step
                theta[i] -= lr * m[i]
            }
            Kind.ADAM -> {
                val bc1 = 1 - beta1.pow(t); val bc2 = 1 - beta2.pow(t)
                val stepSize = lr / bc1; val bc2Sqrt = sqrt(bc2)
                for (i in 0..1) {
                    m[i] += (g[i] - m[i]) * (1 - beta1)               // = β₁m + (1−β₁)g   (torch: lerp)
                    v[i] = v[i] * beta2 + (1 - beta2) * g[i] * g[i]
                    val denom = sqrt(v[i]) / bc2Sqrt + eps              // = √v̂ + ε
                    theta[i] -= stepSize * (m[i] / denom)               // = η·m̂/(√v̂ + ε)   (torch: addcdiv)
                }
            }
        }
    }
}

/**
 * SplitMix64 (Steele, Lea & Flood 2014; Vigna's reference splitmix64.c) + Box–Muller: a tiny, fully specified
 * generator, so a seeded noisy run is reproducible on any JVM / device.
 */
class Rng(seed: Long) {
    private var s = seed
    private var spare = Double.NaN

    fun nextLong(): Long {
        s += -0x61c8864680b583ebL                 // 0x9E3779B97F4A7C15
        var z = s
        z = (z xor (z ushr 30)) * -0x40a7b892e31b1a47L    // 0xBF58476D1CE4E5B9
        z = (z xor (z ushr 27)) * -0x6b2fb644ecceee15L    // 0x94D049BB133111EB
        return z xor (z ushr 31)
    }

    /** Uniform in [0, 1). */
    fun nextDouble(): Double = (nextLong() ushr 11) * (1.0 / (1L shl 53))

    /** Uniform integer in [0, n). */
    fun nextInt(n: Int): Int = ((nextLong() ushr 1) % n).toInt()

    /** Standard normal (Box–Muller; the second value of each pair is kept for the next call). */
    fun gaussian(): Double {
        if (!spare.isNaN()) { val r = spare; spare = Double.NaN; return r }
        val u1 = 1.0 - nextDouble()                 // (0, 1]
        val u2 = nextDouble()
        val r = sqrt(-2.0 * ln(u1)); val a = 2.0 * Math.PI * u2
        spare = r * sin(a)
        return r * cos(a)
    }
}
