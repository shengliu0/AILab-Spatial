package com.bytedance.ailab.core.descent

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * The gradient-descent core, checked against REAL numbers: analytic gradients vs finite differences, the known
 * minima, the closed-form stability limit of GD on a quadratic, the textbook update rules worked by hand, and
 * float64 trajectories of torch.optim (tools/train_descent.py) on the very same landscapes.
 */
class DescentCoreTest {

    // ------------------------------------------------------------------ landscapes

    @Test
    fun analyticGradientsMatchFiniteDifferences() {
        val rng = Rng(7)
        val g = DoubleArray(2); val gp = DoubleArray(2)
        for (s in Surfaces.all) {
            var worst = 0.0
            val h = 1e-6 * (s.x1 - s.x0)
            repeat(300) {
                val x = s.x0 + (s.x1 - s.x0) * rng.nextDouble(); val y = s.y0 + (s.y1 - s.y0) * rng.nextDouble()
                s.grad(x, y, g)
                val fx = (s.f(x + h, y) - s.f(x - h, y)) / (2 * h)
                val fy = (s.f(x, y + h) - s.f(x, y - h)) / (2 * h)
                val err = maxOf(abs(fx - g[0]), abs(fy - g[1])) / (1.0 + hypot(g[0], g[1]))
                worst = maxOf(worst, err)
            }
            println("GRAD ${s.key}: worst |FD − analytic|/(1+|∇f|) over 300 points = %.2e".format(worst))
            assertTrue("${s.key}: analytic gradient disagrees with central differences ($worst)", worst < 1e-6)
            // the minibatch gradient of the real model with the full batch IS the exact gradient
            if (s === Surfaces.LOGISTIC) { Logistic.batchGrad(0.3, -0.7, null, LogisticData.N, gp); s.grad(0.3, -0.7, g); assertEquals(g[0], gp[0], 0.0) }
        }
    }

    @Test
    fun knownMinima() {
        val g = DoubleArray(2)
        // Rosenbrock (1960): global minimum (1, 1), f = 0 exactly
        assertEquals(0.0, Surfaces.ROSENBROCK.f(1.0, 1.0), 0.0)
        Surfaces.ROSENBROCK.grad(1.0, 1.0, g); assertEquals(0.0, hypot(g[0], g[1]), 0.0)
        // Himmelblau (1972): f(3, 2) = 0 exactly; the other three minima are irrational. Refine the rounded literature
        // values (6 decimals) by Newton on the analytic gradient and check they land on the constants we draw.
        assertEquals(0.0, Surfaces.HIMMELBLAU.f(3.0, 2.0), 0.0)
        val lit = listOf(doubleArrayOf(3.0, 2.0), doubleArrayOf(-2.805118, 3.131312), doubleArrayOf(-3.779310, -3.283186), doubleArrayOf(3.584428, -1.848126))
        for ((k, p0) in lit.withIndex()) {
            val p = newton(Surfaces.HIMMELBLAU, p0[0], p0[1])
            val m = Surfaces.HIMMELBLAU.minima[k]
            println("HIMMELBLAU min $k: Newton (%.12f, %.12f) f = %.1e, λ = %s".format(p[0], p[1], Surfaces.HIMMELBLAU.f(p[0], p[1]),
                eig(Surfaces.HIMMELBLAU, p[0], p[1]).joinToString { "%.3f".format(it) }))
            assertEquals(m[0], p[0], 1e-12); assertEquals(m[1], p[1], 1e-12)
            assertTrue(Surfaces.HIMMELBLAU.f(m[0], m[1]) < 1e-20)
            assertTrue("minimum $k must be a minimum", eig(Surfaces.HIMMELBLAU, p[0], p[1]).all { it > 0 })
        }
        // ... and its local MAXIMUM (−0.270845, −0.923039), f = 181.617 (both Hessian eigenvalues negative)
        val mx = newton(Surfaces.HIMMELBLAU, -0.27, -0.92)
        assertEquals(181.6165215225827, Surfaces.HIMMELBLAU.f(mx[0], mx[1]), 1e-9)
        assertTrue(eig(Surfaces.HIMMELBLAU, mx[0], mx[1]).all { it < 0 })
        // bowl: minimum at the origin, Hessian eigenvalues exactly (1, 10) whatever the rotation
        assertEquals(0.0, Surfaces.BOWL.f(0.0, 0.0), 0.0)
        val eb = eig(Surfaces.BOWL, 0.3, -0.4)
        assertEquals(1.0, eb[0], 1e-6); assertEquals(10.0, eb[1], 1e-6)
        // saddle: (0, ±1) minima f = 0, (0, 0) a saddle — ∇f = 0 and Hessian eigenvalues (−1, 2): indefinite
        for (sy in listOf(1.0, -1.0)) { assertEquals(0.0, Surfaces.SADDLE.f(0.0, sy), 0.0); Surfaces.SADDLE.grad(0.0, sy, g); assertEquals(0.0, hypot(g[0], g[1]), 0.0) }
        Surfaces.SADDLE.grad(0.0, 0.0, g); assertEquals(0.0, hypot(g[0], g[1]), 0.0)
        val es = eig(Surfaces.SADDLE, 0.0, 0.0); assertEquals(-1.0, es[0], 1e-6); assertEquals(2.0, es[1], 1e-6)
        // logistic regression: the minimum from tools/train_descent.py (Newton; torch LBFGS agrees to 3e-9)
        val w = Surfaces.LOGISTIC.minima[0]
        Surfaces.LOGISTIC.grad(w[0], w[1], g)
        println("LOGISTIC w* = (%.9f, %.9f): |∇L| = %.1e, L* = %.12f, λ(H) = %s, accuracy %.3f".format(w[0], w[1], hypot(g[0], g[1]),
            Surfaces.LOGISTIC.f(w[0], w[1]), eig(Surfaces.LOGISTIC, w[0], w[1]).joinToString { "%.4f".format(it) }, accuracy(w)))
        assertTrue(hypot(g[0], g[1]) < 1e-11)
        assertEquals(0.355557735856, Surfaces.LOGISTIC.f(w[0], w[1]), 1e-11)
        assertEquals(0.875, accuracy(w), 1e-12)           // 35 of 40 training points (the blobs overlap)
        val p = newton(Surfaces.LOGISTIC, 0.0, 0.0); assertEquals(w[0], p[0], 1e-10); assertEquals(w[1], p[1], 1e-10)
    }

    // ------------------------------------------------------------------ GD on a quadratic: closed form

    /**
     * On ½(λ₁u² + λ₂v²) plain GD multiplies the error along each eigenvector by (1 − ηλ) per step (exactly), so it
     * converges iff |1 − ηλmax| < 1, i.e. η < 2/λmax = 0.2 — and above that it diverges geometrically.
     */
    @Test
    fun gdOnTheBowlConvergesIffLrBelowTwoOverLambdaMax() {
        val b = Surfaces.BOWL
        val x0 = Bowl.startX; val y0 = Bowl.startY
        val u0 = Bowl.C * x0 + Bowl.S * y0; val v0 = -Bowl.S * x0 + Bowl.C * y0
        for (lr in listOf(0.05, 0.1, 0.15, 0.199, 0.201, 0.25, 0.3)) {
            val th = doubleArrayOf(x0, y0); val g = DoubleArray(2); val o = Optimizer(Kind.SGD)
            var worst = 0.0
            for (t in 1..60) {
                b.grad(th[0], th[1], g); o.step(th, g, lr)
                val u = u0 * (1 - lr * Bowl.L1).pow(t); val v = v0 * (1 - lr * Bowl.L2).pow(t)
                val cx = Bowl.C * u - Bowl.S * v; val cy = Bowl.S * u + Bowl.C * v           // back to (x, y)
                worst = maxOf(worst, hypot(th[0] - cx, th[1] - cy) / (1 + hypot(cx, cy)))
            }
            val factor = maxOf(abs(1 - lr * Bowl.L1), abs(1 - lr * Bowl.L2))
            println("BOWL η=%.3f: |1−ηλ|max = %.3f → %s; sim vs closed form after 60 steps: %.1e".format(lr, factor, if (factor < 1) "converges" else "diverges", worst))
            assertTrue("η=$lr: simulation leaves the closed form ($worst)", worst < 1e-9)
            assertEquals(lr < 2 / Bowl.L2, factor < 1)
        }
        assertEquals(0.2, 2 / b.lambdaMax(0.4, 0.1), 1e-7)
        // the race runner agrees: η = 0.15 reaches the bottom, η = 0.3 leaves the landscape (发散)
        val race = Race(); race.setSurface(b)
        race.lr = 0.15; race.reset(); repeat(300) { race.step() }
        assertTrue(race.racers[0].firstHit in 1..300 && !race.racers[0].diverged)
        race.lr = 0.3; race.reset(); repeat(30) { race.step() }
        assertTrue("SGD at η = 0.3 must diverge", race.racers[0].diverged)
        println("BOWL race η=0.3: SGD diverged at step ${race.racers[0].steps}; momentum ${if (race.racers[1].diverged) "diverged" else "survives"} " +
            "(heavy-ball limit 2(1+β)/λmax = %.2f)".format(2 * 1.9 / Bowl.L2))
        // heavy ball tolerates up to 2(1+β)/λmax = 0.38: at η = 0.3 momentum survives where SGD explodes
        assertFalse(race.racers[1].diverged)
    }

    // ------------------------------------------------------------------ update rules, worked by hand

    @Test
    fun momentumMatchesTheTextbookRule() {
        // gradients fed by hand: g1 = (1, −2), g2 = (3, 0.5); η = 0.1, β = 0.9, θ0 = 0
        //  v1 = g1 = (1, −2)                θ1 = (−0.1, 0.2)
        //  v2 = 0.9·v1 + g2 = (3.9, −1.3)   θ2 = θ1 − 0.1·v2 = (−0.49, 0.33)
        val o = Optimizer(Kind.MOMENTUM); val th = DoubleArray(2)
        o.step(th, doubleArrayOf(1.0, -2.0), 0.1)
        assertEquals(-0.1, th[0], 1e-15); assertEquals(0.2, th[1], 1e-15)
        o.step(th, doubleArrayOf(3.0, 0.5), 0.1)
        assertEquals(-0.49, th[0], 1e-15); assertEquals(0.33, th[1], 1e-15)
        assertEquals(3.9, o.m[0], 1e-15); assertEquals(-1.3, o.m[1], 1e-15)
        // == Sutskever et al. 2013 form u ← βu − ηg, θ ← θ + u, step by step on a real landscape
        val a = doubleArrayOf(-1.0, 0.7); val b = doubleArrayOf(-1.0, 0.7); val u = DoubleArray(2); val g = DoubleArray(2)
        val om = Optimizer(Kind.MOMENTUM)
        repeat(40) {
            Surfaces.ROSENBROCK.grad(a[0], a[1], g); om.step(a, g, 0.003)
            Surfaces.ROSENBROCK.grad(b[0], b[1], g); for (i in 0..1) { u[i] = 0.9 * u[i] - 0.003 * g[i]; b[i] += u[i] }
            assertEquals(b[0], a[0], 1e-12); assertEquals(b[1], a[1], 1e-12)
        }
    }

    @Test
    fun adamMatchesKingmaBa2015IncludingBiasCorrection() {
        // Algorithm 1 of Kingma & Ba (2015), η = 0.1, β1 = 0.9, β2 = 0.999, ε = 1e-8, gradients g1 = (1, −2), g2 = (3, 0.5).
        //  t=1: m = (0.1, −0.2), v = (0.001, 0.004); m̂ = (1, −2), v̂ = (1, 4)
        //       θ1 = −0.1·m̂/(√v̂ + ε) = (−0.0999999990000000100, 0.0999999995000000025)   (40-digit decimal arithmetic)
        //  t=2: m = (0.39, −0.13), v = (0.009999, 0.004246); m̂ = m/0.19, v̂ = v/0.001999
        //       θ2 = (−0.1917781104876683721, 0.1469468162986654971)
        val o = Optimizer(Kind.ADAM); val th = DoubleArray(2)
        o.step(th, doubleArrayOf(1.0, -2.0), 0.1)
        assertEquals(0.1, o.m[0], 1e-16); assertEquals(-0.2, o.m[1], 1e-16)
        assertEquals(0.001, o.v[0], 1e-17); assertEquals(0.004, o.v[1], 1e-17)      // 1 − 0.999 is not exact in binary
        assertEquals(-0.0999999990000000100, th[0], 1e-15); assertEquals(0.0999999995000000025, th[1], 1e-15)
        o.step(th, doubleArrayOf(3.0, 0.5), 0.1)
        assertEquals(0.39, o.m[0], 1e-15); assertEquals(-0.13, o.m[1], 1e-15)
        assertEquals(0.009999, o.v[0], 1e-16); assertEquals(0.004246, o.v[1], 1e-16)
        assertEquals(-0.1917781104876683721, th[0], 1e-15); assertEquals(0.1469468162986654971, th[1], 1e-15)
        // without the bias correction the first step would be 0.1·0.1/√0.001 ≈ 0.316 instead of η: the correction
        // is what makes Adam's first step η·g/(|g| + ε) ≈ η·sign(g), whatever the gradient's scale (until |g| ~ ε)
        for (scale in listOf(1e-3, 1.0, 1e6)) {
            val oa = Optimizer(Kind.ADAM); val t2 = DoubleArray(2)
            oa.step(t2, doubleArrayOf(scale, -3 * scale), 0.05)
            assertEquals(-0.05 * scale / (scale + 1e-8), t2[0], 1e-15); assertEquals(0.05, t2[1], 1e-6)
        }
    }

    // ------------------------------------------------------------------ torch.optim reference trajectories

    private class Ref(val s: LossSurface, val start: DoubleArray, val kind: Kind, val lr: Double, val at: Map<Int, DoubleArray>)

    /** Printed by tools/train_descent.py (torch 2.11, float64): positions after steps 1, 2, 3, 10, 50. */
    private val refs = listOf(
        Ref(Surfaces.ROSENBROCK, doubleArrayOf(-1.0, 0.7), Kind.SGD, 0.002, mapOf(1 to doubleArrayOf(-0.752, 0.82), 2 to doubleArrayOf(-0.8980967936, 0.7182016), 3 to doubleArrayOf(-0.827008064536601, 0.75355210026984), 10 to doubleArrayOf(-0.840688810239023, 0.713985292012136), 50 to doubleArrayOf(-0.759451443100225, 0.584844689667473))),
        Ref(Surfaces.ROSENBROCK, doubleArrayOf(-1.0, 0.7), Kind.MOMENTUM, 0.003, mapOf(1 to doubleArrayOf(-0.628, 0.88), 2 to doubleArrayOf(-0.6493922176, 0.7506304), 3 to doubleArrayOf(-0.915066681104284, 0.436845671367667), 10 to doubleArrayOf(0.0647355440278121, -0.176134650726606), 50 to doubleArrayOf(0.825764939745515, 0.714524148104763))),
        Ref(Surfaces.ROSENBROCK, doubleArrayOf(-1.0, 0.7), Kind.ADAM, 0.05, mapOf(1 to doubleArrayOf(-0.950000000004032, 0.749999999991667), 2 to doubleArrayOf(-0.903408750443133, 0.796729491067143), 3 to doubleArrayOf(-0.865013585907503, 0.834632978768808), 10 to doubleArrayOf(-0.880424354992072, 0.77919411088515), 50 to doubleArrayOf(-0.791094313375607, 0.622360446954576))),
        Ref(Surfaces.HIMMELBLAU, doubleArrayOf(0.0, 0.0), Kind.SGD, 0.01, mapOf(1 to doubleArrayOf(0.14, 0.22), 2 to doubleArrayOf(0.33649024, 0.49515008), 3 to doubleArrayOf(0.604724168524076, 0.830104151257185), 10 to doubleArrayOf(2.89414972825764, 2.17238514692945), 50 to doubleArrayOf(2.99999954521065, 2.00000109795738))),
        Ref(Surfaces.HIMMELBLAU, doubleArrayOf(-0.5, -1.5), Kind.ADAM, 0.1, mapOf(1 to doubleArrayOf(-0.599999999928571, -1.59999999985714), 2 to doubleArrayOf(-0.69957756026752, -1.7001357839552), 3 to doubleArrayOf(-0.799026708030664, -1.80047713816917), 10 to doubleArrayOf(-1.50914809595638, -2.48592561472136), 50 to doubleArrayOf(-3.85631723244603, -3.29546355654021))),
        Ref(Surfaces.BOWL, doubleArrayOf(-1.2, -1.7), Kind.MOMENTUM, 0.02, mapOf(1 to doubleArrayOf(-1.08361370442519, -1.69962586164335), 2 to doubleArrayOf(-0.883284727859512, -1.69219730920418), 3 to doubleArrayOf(-0.642825258428798, -1.66713556218078), 10 to doubleArrayOf(-0.213170550653532, -0.578654800286859), 50 to doubleArrayOf(-0.0351225888020338, -0.145203684576187))),
        Ref(Surfaces.SADDLE, doubleArrayOf(-1.8, 0.02), Kind.ADAM, 0.05, mapOf(1 to doubleArrayOf(-1.75000000013889, 0.0699999749900085), 2 to doubleArrayOf(-1.700041667765, 0.115002514664603), 3 to doubleArrayOf(-1.65015439413427, 0.160642338320708), 10 to doubleArrayOf(-1.30563789492568, 0.50161421153799), 50 to doubleArrayOf(-0.0129071090789499, 0.998856155730955))),
        Ref(Surfaces.LOGISTIC, doubleArrayOf(-3.0, 3.0), Kind.MOMENTUM, 0.1, mapOf(1 to doubleArrayOf(-2.91024949802295, 2.93725655406905), 2 to doubleArrayOf(-2.74038970837468, 2.81826111633423), 3 to doubleArrayOf(-2.49971836954813, 2.64905605526921), 10 to doubleArrayOf(0.420272342671449, 0.508932769189657), 50 to doubleArrayOf(1.54350754817692, 0.0928400772449761))),
        Ref(Surfaces.LOGISTIC, doubleArrayOf(-3.0, 3.0), Kind.ADAM, 0.1, mapOf(1 to doubleArrayOf(-2.9000000011142, 2.90000000159379), 2 to doubleArrayOf(-2.80001462255271, 2.80003126316394), 3 to doubleArrayOf(-2.70005385764613, 2.70011600060857), 10 to doubleArrayOf(-2.00190597588889, 2.00434786897352), 50 to doubleArrayOf(1.19764223632028, 0.239358648902873))),
    )

    @Test
    fun trajectoriesMatchTorchOptim() {
        var worst = 0.0
        for (r in refs) {
            val o = Optimizer(r.kind); val th = r.start.copyOf(); val g = DoubleArray(2)
            var w = 0.0
            for (t in 1..50) {
                r.s.grad(th[0], th[1], g); o.step(th, g, r.lr)
                r.at[t]?.let { p -> w = maxOf(w, abs(p[0] - th[0]), abs(p[1] - th[1])) }
            }
            println("TORCH ${r.s.key} ${r.kind} η=${r.lr}: max |Kotlin − torch| over steps 1,2,3,10,50 = %.1e".format(w))
            worst = maxOf(worst, w)
        }
        // torch prints 15 significant digits: agreement to ~1e-13 means the same arithmetic
        assertTrue("Kotlin trajectories leave torch.optim's (worst $worst)", worst < 1e-9)
    }

    // ------------------------------------------------------------------ the experiences the lesson promises

    @Test
    fun saddleEscapeRatesMatchTheLinearisation() {
        // near (0, 0): y ← y(1 + η) for GD; heavy ball's growth factor is the larger root of r² − (1+β+η)r + β = 0
        val lr = 0.05
        val rMom = ((1.9 + lr) + sqrt((1.9 + lr).pow(2) - 4 * 0.9)) / 2         // = 1.2 exactly for η = 0.05
        for ((kind, want) in listOf(Kind.SGD to 1 + lr, Kind.MOMENTUM to rMom)) {
            val o = Optimizer(kind); val th = doubleArrayOf(0.0, 1e-9); val g = DoubleArray(2)
            var last = th[1]; var ratio = 0.0
            repeat(60) { Surfaces.SADDLE.grad(th[0], th[1], g); o.step(th, g, lr); ratio = th[1] / last; last = th[1] }
            println("SADDLE $kind η=$lr: growth per step %.6f (linearised %.6f)".format(ratio, want))
            assertEquals(want, ratio, 1e-6)
        }
        assertEquals(1.2, rMom, 1e-12)
        // from the default start (−1.8, 0.01): SGD dawdles by the pass, momentum and Adam do not
        val race = Race(); race.setSurface(Surfaces.SADDLE)
        val near = IntArray(3)
        repeat(300) { race.step(); for (i in 0..2) { val r = race.racers[i]; if (hypot(r.theta[0], r.theta[1]) < 0.15) near[i]++ } }
        println("SADDLE default start, η=0.05: steps spent within 0.15 of the saddle  SGD ${near[0]}, momentum ${near[1]}, Adam ${near[2]}")
        assertTrue(near[0] > 3 * maxOf(near[1], near[2], 1))
        assertTrue(race.racers.all { it.firstHit > 0 })              // all three end in a valley (0, ±1)
    }

    @Test
    fun lessonFactsHold() {
        val race = Race()
        // race on the ravine at η = 0.02: momentum reaches the bottom long before SGD (both stable)
        race.setSurface(Surfaces.BOWL); race.lr = 0.02; race.reset(); repeat(400) { race.step() }
        val (sgd, mom, adam) = race.racers
        println("BOWL η=0.02: first step with loss < 1e-3  SGD ${sgd.firstHit}, momentum ${mom.firstHit}, Adam ${adam.firstHit}")
        assertTrue(mom.firstHit in 1 until sgd.firstHit)
        // Rosenbrock at η = 0.05 (its default): SGD and momentum fly off at once, Adam walks into the valley
        race.setSurface(Surfaces.ROSENBROCK); repeat(200) { race.step() }
        println("ROSEN η=0.05: SGD diverged at ${race.racers[0].steps}, momentum at ${race.racers[1].steps}; Adam after 200 steps: loss %.3f".format(race.racers[2].loss))
        assertTrue(race.racers[0].diverged && race.racers[1].diverged && !race.racers[2].diverged)
        assertTrue(race.racers[2].loss < 0.5 * Surfaces.ROSENBROCK.f(-1.0, 0.7))
        // Himmelblau from (0, 0), η = 0.01: SGD settles in (3, 2), momentum overshoots into (3.584, −1.848)
        race.setSurface(Surfaces.HIMMELBLAU); repeat(1500) { race.step() }
        val endS = Surfaces.HIMMELBLAU.nearestMin(race.racers[0].theta[0], race.racers[0].theta[1])
        val endM = Surfaces.HIMMELBLAU.nearestMin(race.racers[1].theta[0], race.racers[1].theta[1])
        println("HIMMELBLAU (0,0) η=0.01: SGD → (%.3f, %.3f), momentum → (%.3f, %.3f)".format(endS[0], endS[1], endM[0], endM[1]))
        assertEquals(3.0, endS[0], 1e-9); assertEquals(3.584428340330492, endM[0], 1e-9)
    }

    /** The challenge: Rosenbrock from the default start, loss < 1e-3 within 200 steps. Possible — but not with SGD. */
    @Test
    fun challengeIsAchievable() {
        val race = Race(); race.setSurface(Surfaces.ROSENBROCK)
        val solved = ArrayList<String>()
        for (lr in LR_STEPS) {
            race.lr = lr; race.reset(); repeat(200) { race.step() }
            for (r in race.racers) if (r.firstHit in 1..200) solved += "${r.kind} η=$lr @${r.firstHit}"
        }
        println("CHALLENGE solutions over all 19 slider positions: $solved")
        assertTrue(solved.any { it.startsWith("MOMENTUM η=0.003") })
        assertTrue(solved.none { it.startsWith("SGD") })
        // matches torch.optim (tools/train_descent.py: momentum η=0.003 solves at step 163)
        race.lr = 0.003; race.reset(); repeat(200) { race.step() }
        assertEquals(163, race.racers[1].firstHit)
        assertTrue(race.startToMin >= 1.5)
    }

    // ------------------------------------------------------------------ noise: deterministic, unbiased

    @Test
    fun noiseIsDeterministicWithAFixedSeed() {
        // SplitMix64 reference vectors (seed 0): 0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4
        val r0 = Rng(0)
        assertEquals(0xE220A8397B1DCDAFuL.toLong(), r0.nextLong()); assertEquals(0x6E789E6AA1B965F4L, r0.nextLong())
        // Gaussian moments over 2e5 draws
        val rg = Rng(1847); var s1 = 0.0; var s2 = 0.0; val n = 200_000
        repeat(n) { val z = rg.gaussian(); s1 += z; s2 += z * z }
        println("RNG gaussian: mean %.4f, var %.4f (n = $n)".format(s1 / n, s2 / n))
        assertEquals(0.0, s1 / n, 0.01); assertEquals(1.0, s2 / n, 0.01)
        // same seed → identical noisy races; a different seed → a different race; zero noise → seed irrelevant
        fun run(seed: Long, noise: Double): DoubleArray {
            val race = Race(seed); race.setSurface(Surfaces.BOWL); race.noise = noise; race.reset(); repeat(120) { race.step() }
            return race.racers.flatMap { listOf(it.theta[0], it.theta[1]) }.toDoubleArray()
        }
        val a = run(5, 0.6); val b = run(5, 0.6); val c = run(6, 0.6)
        assertTrue(a.contentEquals(b))
        assertFalse(a.contentEquals(c))
        assertTrue(run(5, 0.0).contentEquals(run(99, 0.0)))
        // the logistic landscape's noise is a REAL minibatch: noise 0 → all 40 points, noise 1 → 4; the minibatch
        // gradient is unbiased (its mean over many draws is the full gradient)
        assertEquals(40, Logistic.batchSize(0.0)); assertEquals(4, Logistic.batchSize(1.0)); assertEquals(22, Logistic.batchSize(0.5))
        val rng = Rng(3); val g = DoubleArray(2); val full = DoubleArray(2); val m = 200_000
        var mx = 0.0; var my = 0.0; var qx = 0.0; var qy = 0.0
        Surfaces.LOGISTIC.grad(-1.0, 2.0, full)
        repeat(m) { Surfaces.LOGISTIC.noisyGrad(-1.0, 2.0, 1.0, rng, g); mx += g[0]; my += g[1]; qx += g[0] * g[0]; qy += g[1] * g[1] }
        mx /= m; my /= m
        val seX = sqrt((qx / m - mx * mx) / m); val seY = sqrt((qy / m - my * my) / m)       // standard errors of the means
        println("LOGISTIC minibatch(4) mean gradient (%.5f, %.5f) vs full (%.5f, %.5f): %.1f and %.1f standard errors".format(
            mx, my, full[0], full[1], abs(mx - full[0]) / seX, abs(my - full[1]) / seY))
        assertTrue(abs(mx - full[0]) < 4 * seX && abs(my - full[1]) < 4 * seY)
    }

    // ------------------------------------------------------------------ helpers

    private fun newton(s: LossSurface, x0: Double, y0: Double): DoubleArray {
        var x = x0; var y = y0; val g = DoubleArray(2)
        repeat(50) {
            s.grad(x, y, g); val h = s.hessian(x, y)
            val det = h[0] * h[3] - h[1] * h[2]
            x -= (h[3] * g[0] - h[1] * g[1]) / det; y -= (-h[2] * g[0] + h[0] * g[1]) / det
        }
        return doubleArrayOf(x, y)
    }

    private fun eig(s: LossSurface, x: Double, y: Double): DoubleArray {
        val h = s.hessian(x, y); val p = (h[0] + h[3]) / 2; val q = (h[1] + h[2]) / 2
        val r = sqrt(((h[0] - h[3]) / 2).pow(2) + q * q)
        return doubleArrayOf(p - r, p + r)
    }

    private fun accuracy(w: DoubleArray): Double =
        (0 until LogisticData.N).count { i -> (w[0] * LogisticData.x1(i) + w[1] * LogisticData.x2(i)) * LogisticData.label(i) > 0 }.toDouble() / LogisticData.N

    @Suppress("unused") private fun rot(a: Double) = doubleArrayOf(cos(a), sin(a))
}
