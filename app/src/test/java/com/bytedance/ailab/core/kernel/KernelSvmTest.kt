package com.bytedance.ailab.core.kernel

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.hypot

/**
 * The kernel-trick core, checked against real numbers:
 *  - SMO solutions satisfy the KKT conditions and equal independent references (a hand-solved case and scipy's
 *    trust-constr QP, tools/ref_kernel.py, numbers pasted below);
 *  - rings: 2-D linear baseline ≤ 70 %, lifted by x²+y² = 100 % separable; XOR needs x·y; moons need the RBF lift;
 *  - the plane ∩ paraboloid, projected to the floor, is the analytic circle;
 *  - moving a non-support vector (staying outside the margin) leaves (w, b) unchanged; moving a support vector does not;
 *  - determinism.
 */
class KernelSvmTest {

    private fun arr(vararg p: Double) = Array(p.size / 2) { doubleArrayOf(p[it * 2], p[it * 2 + 1]) }

    @Test fun handCaseFourPoints() {
        // + at (1,0),(2,1); − at (−1,0),(−2,−1). Closest hull points (1,0),(−1,0): plane x = 0, w = (1,0), b = 0,
        // margin 2, α = ½ on the two SVs (Σαᵢyᵢxᵢ = w), 0 on the others.
        val m = LinearSvm.train(arr(1.0, 0.0, 2.0, 1.0, -1.0, 0.0, -2.0, -1.0), intArrayOf(1, 1, -1, -1), 1e3)
        println("KERNEL hand: w=${m.w.toList()} b=${m.b} alpha=${m.sol.alpha.toList()} iters=${m.sol.iterations}")
        assertArrayEquals(doubleArrayOf(1.0, 0.0), m.w, 1e-8)
        assertEquals(0.0, m.b, 1e-8)
        assertEquals(2.0, m.margin, 1e-8)
        assertArrayEquals(doubleArrayOf(0.5, 0.0, 0.5, 0.0), m.sol.alpha, 1e-8)
        assertTrue(m.kktViolation() < 1e-8)
    }

    @Test fun matchesScipySoftMargin() {
        // tools/ref_kernel.py case B (scipy trust-constr): w = [1.27804877, 0.90243904], b = −0.06780489, acc 0.9
        val X = arr(0.2, 0.9, 0.5, 0.4, 0.9, 0.7, -0.1, 0.3, 0.6, -0.2, -0.5, -0.4, -0.8, 0.1, 0.1, -0.7, -0.3, 0.5, 0.3, 0.1)
        val y = intArrayOf(1, 1, 1, 1, 1, -1, -1, -1, -1, -1)
        val m = LinearSvm.train(X, y, 1.0)
        println("KERNEL scipy-B: w=${m.w.toList()} b=${m.b} kkt=${m.kktViolation()} acc=${m.accuracy()}")
        assertArrayEquals(doubleArrayOf(1.27804877, 0.90243904), m.w, 1e-5)
        assertEquals(-0.06780489, m.b, 1e-5)
        // (accuracy is not compared: one point lies exactly on the boundary, f ≈ 0, so its sign is a tie)
        assertTrue(m.kktViolation() < 1e-7)
    }

    @Test fun matchesScipyOnLiftedNoisyRings() {
        // case C: the demo's own RINGS, noise level 1, lifted (x, y, x²+y²), C = 1. Python re-implements the generator:
        // first points (0.01552126235915537, 0.2520357065595648), (0.4585773679113764, −0.7117225991341388).
        val p = KernelData.make(DataSet.RINGS, 1)
        assertEquals(0.01552126235915537, p.x[0], 1e-12); assertEquals(0.2520357065595648, p.y[0], 1e-12)
        assertEquals(0.4585773679113764, p.x[1], 1e-12); assertEquals(-0.7117225991341388, p.y[1], 1e-12)
        val lab = KernelLab().apply { configure(DataSet.RINGS, 1, Lift.RADIAL, 1.0) }
        val m = lab.svm3
        println("KERNEL scipy-C: w=${m.w.toList()} b=${m.b} margin=${m.margin} #SV=${m.sol.supportVectors().size} kkt=${m.kktViolation()}")
        assertArrayEquals(doubleArrayOf(-0.31705880, 0.03143142, -3.61339998), m.w, 1e-5)
        assertEquals(1.36834931, m.b, 1e-5)
        assertEquals(19, m.sol.supportVectors().size)
    }

    @Test fun kktHoldsEverywhere() {
        var worst = 0.0
        for (s in DataSet.values()) for (lift in Lift.values()) for (nz in 0..2) for (C in doubleArrayOf(0.1, 1.0, 1000.0)) {
            val lab = KernelLab().apply { configure(s, nz, lift, C) }
            for (m in listOf(lab.svm2, lab.svm3)) {
                val v = m.kktViolation() / maxOf(1.0, m.wNorm)
                worst = maxOf(worst, v)
                assertTrue("$s $lift noise $nz C $C: KKT violation $v", v < 1e-6)
            }
        }
        println("KERNEL worst scaled KKT violation over 4 sets × 3 lifts × 3 noises × 3 C × 2 models = $worst")
    }

    @Test fun liftingSeparatesTheRings() {
        val lab = KernelLab()
        val a2 = lab.accuracy(false); val a3 = lab.accuracy(true)
        println("KERNEL rings: 2-D line acc=$a2 (w=${lab.svm2.w.toList()}), 3-D plane acc=$a3 margin=${lab.svm3.margin} #SV=${lab.svm3.sol.supportVectors().size}")
        assertTrue("2-D baseline should fail, got $a2", a2 <= 0.70)
        assertEquals(1.0, a3, 0.0)
        // every point outside or on the margin: hard-margin separation
        for (i in 0 until lab.n) assertTrue(lab.marginOf(i, true) >= 1 - 1e-6)
    }

    @Test fun xorNeedsTheProductLiftMoonsTheRbf() {
        val xr = KernelLab().apply { configure(DataSet.XOR, 0, Lift.RADIAL, 1000.0) }
        val xp = KernelLab().apply { configure(DataSet.XOR, 0, Lift.PRODUCT, 1000.0) }
        val mr = KernelLab().apply { configure(DataSet.MOONS, 0, Lift.RADIAL, 1000.0) }
        val mb = KernelLab().apply { configure(DataSet.MOONS, 0, Lift.RBF, 1000.0) }
        val li = KernelLab().apply { configure(DataSet.LINEAR, 0, Lift.RADIAL, 1000.0) }
        println("KERNEL xor: 2-D ${xr.accuracy(false)}, x²+y² ${xr.accuracy(true)}, x·y ${xp.accuracy(true)}; moons: 2-D ${mr.accuracy(false)}, x²+y² ${mr.accuracy(true)}, RBF ${mb.accuracy(true)}; linear: 2-D ${li.accuracy(false)}")
        assertTrue(xr.accuracy(true) <= 0.75)
        assertEquals(1.0, xp.accuracy(true), 0.0)
        assertTrue(mr.accuracy(true) < 1.0)
        assertEquals(1.0, mb.accuracy(true), 0.0)
        assertEquals(1.0, li.accuracy(false), 0.0)
    }

    @Test fun projectedCutIsTheAnalyticCircle() {
        val lab = KernelLab()
        val c = lab.circle()!!
        val w = lab.svm3.w; val b = lab.svm3.b
        // radius must sit between the inner disc (r ≤ 0.42) and the outer ring (r ≥ 0.70)
        println("KERNEL circle: centre (${c[0]}, ${c[1]}) r=${c[2]}  plane w=${w.toList()} b=$b")
        assertTrue(c[2] in 0.42..0.70)
        // analytic: every point of the circle lies on the plane at z = x²+y²
        for (k in 0 until 36) {
            val t = 2 * Math.PI * k / 36; val x = c[0] + c[2] * Math.cos(t); val y = c[1] + c[2] * Math.sin(t)
            assertEquals(0.0, w[0] * x + w[1] * y + w[2] * (x * x + y * y) + b, 1e-9)
        }
        // the floor contour the demo draws lies on that circle (marching squares, 48 cells: O(h²) error)
        val segs = lab.contour(true, 48)
        var maxErr = 0.0
        for (s in segs) for (e in 0..1) maxErr = maxOf(maxErr, abs(hypot(s[e * 2] - c[0], s[e * 2 + 1] - c[1]) - c[2]))
        println("KERNEL contour: ${segs.size} segments, max |dist − r| = $maxErr")
        assertTrue(segs.size > 30)
        assertTrue(maxErr < 2e-3)
    }

    @Test fun onlySupportVectorsMatter() {
        for (s in listOf(DataSet.RINGS, DataSet.XOR)) {
            val lift = if (s == DataSet.XOR) Lift.PRODUCT else Lift.RADIAL
            val lab = KernelLab().apply { configure(s, 0, lift, 1000.0) }
            val w0 = lab.svm3.w.copyOf(); val b0 = lab.svm3.b
            val far = lab.farPoint(true)
            val step = lab.safeStepTowardBoundary(far, true, 0.25)
            val d = lab.towardBoundary(far, true)
            lab.move(far, lab.pts.x[far] + d[0] * step, lab.pts.y[far] + d[1] * step)
            val dFar = KernelLab.planeDelta(w0, b0, lab.svm3.w, lab.svm3.b)
            val sv = lab.freeSupport(true)
            val d2 = lab.towardBoundary(sv, true)
            lab.move(sv, lab.pts.x[sv] + d2[0] * 0.1, lab.pts.y[sv] + d2[1] * 0.1)
            val dSv = KernelLab.planeDelta(w0, b0, lab.svm3.w, lab.svm3.b)
            println("KERNEL $s: moved non-SV #$far by $step -> plane change $dFar; moved SV #$sv by 0.1 -> change $dSv")
            assertTrue(step >= 0.1)
            assertTrue(dFar < 1e-6)
            assertTrue(dSv > 1e-2)
        }
    }

    @Test fun deterministic() {
        val a = KernelLab().apply { configure(DataSet.MOONS, 2, Lift.RBF, 3.0) }
        val b = KernelLab().apply { configure(DataSet.MOONS, 2, Lift.RBF, 3.0) }
        assertArrayEquals(a.svm3.w, b.svm3.w, 0.0); assertEquals(a.svm3.b, b.svm3.b, 0.0)
        assertArrayEquals(a.svm3.sol.alpha, b.svm3.sol.alpha, 0.0)
    }

    @Test fun planeBoxPolygon() {
        // the plane z = 0.4 cuts the box [-1,1]²×[0,1] in the square at that height
        val poly = KernelGeom.planeBox(doubleArrayOf(0.0, 0.0, 1.0), -0.4, doubleArrayOf(-1.0, -1.0, 0.0), doubleArrayOf(1.0, 1.0, 1.0))
        assertEquals(4, poly.size); poly.forEach { assertEquals(0.4, it[2], 1e-12) }
        // a diagonal plane through the centre gives a hexagon
        val hex = KernelGeom.planeBox(doubleArrayOf(1.0, 1.0, 1.0), -1.5, doubleArrayOf(0.0, 0.0, 0.0), doubleArrayOf(1.0, 1.0, 1.0))
        assertEquals(6, hex.size)
    }
}
