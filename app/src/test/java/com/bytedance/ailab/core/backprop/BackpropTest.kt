package com.bytedance.ailab.core.backprop

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin

/**
 * The backpropagation core, checked against independent references:
 *   1. finite differences for EVERY weight and bias of random nets (the definition of a derivative);
 *   2. the textbook 2-2-2 example (M. Mazur, "A Step by Step Backpropagation Example", 2015) — published numbers;
 *   3. a hand-computed 2-2-1 tanh/BCE step (worked out below);
 *   4. PyTorch autograd on a 2-4-4-1 net (tools/ref_backprop.py, values pasted here);
 * then the training schedule the demo runs: XOR, spirals, determinism.
 */
class BackpropTest {

    private fun randomNet(sizes: IntArray, seed: Long, hidden: Act = Act.TANH, out: Act = Act.SIGMOID, loss: Loss = Loss.BCE): Net {
        val n = Net(sizes, hidden, out, loss); val r = SplitMix(seed)
        for (l in 0 until n.nL) { for (k in n.w[l].indices) n.w[l][k] = 3.0 * r.nextDouble() - 1.5; for (k in n.b[l].indices) n.b[l][k] = r.nextDouble() - 0.5 }
        return n
    }

    private fun lossAt(n: Net, x: DoubleArray, y: DoubleArray): Double { val t = n.newTrace(); n.forward(x, t); return n.lossOf(t.a[n.nL], y) }

    @Test
    fun analyticGradientsEqualFiniteDifferences() {
        val nets = listOf(
            Triple(intArrayOf(2, 4, 4, 1), Act.TANH, Loss.BCE),
            Triple(intArrayOf(2, 8, 8, 1), Act.TANH, Loss.BCE),
            Triple(intArrayOf(2, 3, 5, 1), Act.TANH, Loss.BCE),
            Triple(intArrayOf(3, 4, 2), Act.SIGMOID, Loss.HALF_SSE),
        )
        var worst = 0.0; var checked = 0
        for ((k, cfg) in nets.withIndex()) for (s in 0 until 5) {
            val (sizes, hid, loss) = cfg
            val net = randomNet(sizes, 100L * k + s, hid, Act.SIGMOID, loss)
            val r = SplitMix(9000L + 10 * k + s)
            val x = DoubleArray(sizes[0]) { 2 * r.nextDouble() - 1 }
            val y = DoubleArray(sizes.last()) { if (loss == Loss.BCE) (s % 2).toDouble() else r.nextDouble() }
            val t = net.newTrace(); net.forward(x, t); net.backward(t, y)
            // 5-point central difference, h = 1e-3: truncation O(h^4) ~ 1e-13, rounding ~ 1e-16 / h = 1e-13 —
            // the plain 2-point rule (rounding ~ 1e-10) cannot resolve 1e-6 relative on gradients as small as 1e-5
            val h = 1e-3
            fun check(arr: DoubleArray, k2: Int, analytic: Double) {
                val o = arr[k2]
                fun at(d: Double): Double { arr[k2] = o + d; return lossAt(net, x, y) }
                val num = (-at(2 * h) + 8 * at(h) - 8 * at(-h) + at(-2 * h)) / (12 * h)
                arr[k2] = o
                val rel = abs(analytic - num) / max(max(abs(analytic), abs(num)), 1e-7)
                worst = max(worst, rel); checked++
                assertTrue("rel err $rel (analytic $analytic, numeric $num) sizes ${sizes.toList()}", rel < 1e-6)
            }
            for (l in 0 until net.nL) {
                for (i in net.w[l].indices) check(net.w[l], i, t.gw[l][i])
                for (i in net.b[l].indices) check(net.b[l], i, t.gb[l][i])
            }
        }
        println("BACKPROP finite differences: $checked parameters, worst relative error %.2e (limit 1e-6)".format(worst))
    }

    /** Mazur 2015: published values after one step at lr 0.5 (w1..w8; biases not updated in his text). */
    @Test
    fun textbookTwoTwoTwoExample() {
        val n = Net(intArrayOf(2, 2, 2), Act.SIGMOID, Act.SIGMOID, Loss.HALF_SSE)
        doubleArrayOf(0.15, 0.20, 0.25, 0.30).copyInto(n.w[0]); n.b[0].fill(0.35)
        doubleArrayOf(0.40, 0.45, 0.50, 0.55).copyInto(n.w[1]); n.b[1].fill(0.60)
        val t = n.newTrace(); n.forward(doubleArrayOf(0.05, 0.10), t); n.backward(t, doubleArrayOf(0.01, 0.99))
        assertEquals(0.298371109, t.lossValue, 1e-9)                       // E_total
        assertEquals(0.75136507, t.a[2][0], 1e-8); assertEquals(0.772928465, t.a[2][1], 1e-9)
        n.sgd(t, 0.5)
        val published = doubleArrayOf(0.149780716, 0.19956143, 0.24975114, 0.29950229, 0.35891648, 0.408666186, 0.511301270, 0.561370121)
        val got = n.w[0] + n.w[1]
        for (k in published.indices) assertEquals("w${k + 1}", published[k], got[k], 1e-8)
        assertArrayEquals(MAZ_W1, n.w[0], 1e-12); assertArrayEquals(MAZ_W2, n.w[1], 1e-12)
        println("BACKPROP Mazur 2-2-2: E=%.9f, w5+=%.8f, w1+=%.9f (published 0.298371109, 0.35891648, 0.149780716)".format(t.lossValue, n.w[1][0], n.w[0][0]))
    }

    /**
     * HAND CALCULATION. 2-2-1 net, tanh hidden, sigmoid out, BCE, x = (1, 0.5), y = 1, lr = 0.5,
     * W1 = [[0.5, -0.5], [0.25, 1.0]], b1 = 0, W2 = [1, -1], b2 = 0.
     *   forward   z1 = (0.5*1 - 0.5*0.5, 0.25*1 + 1*0.5) = (0.25, 0.75)
     *             a1 = (tanh 0.25, tanh 0.75)          = (0.2449187, 0.6351490)
     *             z2 = 0.2449187 - 0.6351490           = -0.3902303
     *             p  = sigmoid(-0.3902303)             = 0.4036619,   L = -ln p = 0.9071777
     *   backward  delta2 = p - y                       = -0.5963381
     *             dL/dW2 = delta2 * a1                 = (-0.1460543, -0.3787635)
     *             delta1 = (1 - a1^2) * W2 * delta2    = (0.9400148 * -0.5963381, 0.5965858 * +0.5963381)
     *                                                  = (-0.5605667, +0.3557669)
     *             dL/dW1 = delta1 (x) x                = [[-0.5605667, -0.2802834], [0.3557669, 0.1778834]]
     *   update    W2' = W2 - 0.5 dL/dW2                = (1.0730272, -0.8106182)
     *             W1' = [[0.7802834, -0.3598583], [0.0721166, 0.9110583]]
     * (PyTorch agrees to 1e-15: HAND_* below.)
     */
    @Test
    fun handComputedStep() {
        val n = Net(intArrayOf(2, 2, 1))
        doubleArrayOf(0.5, -0.5, 0.25, 1.0).copyInto(n.w[0]); doubleArrayOf(1.0, -1.0).copyInto(n.w[1])
        val t = n.newTrace()
        val p = n.forward(doubleArrayOf(1.0, 0.5), t); n.backward(t, doubleArrayOf(1.0))
        assertEquals(0.4036619, p, 1e-7); assertEquals(0.9071777, t.lossValue, 1e-7)
        assertEquals(-0.5963381, t.delta[2][0], 1e-7)
        assertEquals(-0.5605667, t.delta[1][0], 1e-7); assertEquals(0.3557669, t.delta[1][1], 1e-7)
        // the chain-rule product the demo shows for a picked neuron: delta = f'(z) * sum w * delta_next
        assertEquals(t.delta[1][0], n.slope(t, 1, 0) * n.blameSum(t, 1, 0), 1e-15)
        assertArrayEquals(HAND_GW1, t.gw[0], 1e-12); assertArrayEquals(HAND_GW2, t.gw[1], 1e-12); assertArrayEquals(HAND_GB1, t.gb[0], 1e-12)
        n.sgd(t, 0.5)
        assertArrayEquals(doubleArrayOf(0.7802834, -0.3598583, 0.0721166, 0.9110583), n.w[0], 1e-7)
        assertArrayEquals(doubleArrayOf(1.0730272, -0.8106182), n.w[1], 1e-7)
        assertArrayEquals(HAND_NW1, n.w[0], 1e-12); assertArrayEquals(HAND_NW2, n.w[1], 1e-12)
        println("BACKPROP hand step: p=%.7f L=%.7f delta1=(%.7f, %.7f) W2'=(%.7f, %.7f)".format(p, t.lossValue, t.delta[1][0], t.delta[1][1], n.w[1][0], n.w[1][1]))
    }

    /** PyTorch autograd (tools/ref_backprop.py case 1): every gradient and every updated weight of a 2-4-4-1 net. */
    @Test
    fun matchesPytorchAutograd() {
        val sizes = intArrayOf(2, 4, 4, 1)
        val n = Net(sizes)
        for (l in 0 until n.nL) {
            val nIn = sizes[l]
            for (j in 0 until sizes[l + 1]) { for (i in 0 until nIn) n.w[l][j * nIn + i] = 0.6 * sin(1.3 * (j * nIn + i) + 0.7 * l + 0.1); n.b[l][j] = 0.1 * cos(j + 2.0 * l) }
        }
        val t = n.newTrace()
        val p = n.forward(doubleArrayOf(0.3, -0.7), t); n.backward(t, doubleArrayOf(1.0))
        assertEquals(P1, p, 1e-12); assertEquals(LOSS1, t.lossValue, 1e-12)
        val gw = listOf(GW1_0, GW1_1, GW1_2); val gb = listOf(GB1_0, GB1_1, GB1_2); val nw = listOf(NW1_0, NW1_1, NW1_2)
        var worst = 0.0
        for (l in 0 until 3) {
            for (k in gw[l].indices) worst = max(worst, abs(gw[l][k] - t.gw[l][k]))
            for (k in gb[l].indices) worst = max(worst, abs(gb[l][k] - t.gb[l][k]))
        }
        n.sgd(t, 0.1)
        for (l in 0 until 3) for (k in nw[l].indices) worst = max(worst, abs(nw[l][k] - n.w[l][k]))
        println("BACKPROP vs PyTorch: 37 gradients + 28 updated weights, max abs diff %.2e (limit 1e-9)".format(worst))
        assertTrue(worst < 1e-9)
    }

    // ---------------------------------------------------------------- the training the demo runs

    private fun epochsTo(kind: DataKind, width: Int, lrIdx: Int, target: Double, maxEpochs: Int): Int {
        val tr = Trainer(); tr.reset(kind, width, Trainer.SEED); tr.lr = Trainer.LRS[lrIdx]
        while (tr.epoch < maxEpochs) { tr.trainEpochs(1); if (tr.evaluate() >= target) return tr.epoch }
        return -1
    }

    /** Default demo settings (2-4-4-1, lr 0.03, seed 1): XOR reaches 100 % training accuracy within N = 60 epochs. */
    @Test
    fun xorReachesFullAccuracy() {
        val e = epochsTo(DataKind.XOR, Trainer.NARROW, 1, 1.0, 60)
        println("BACKPROP XOR 2-4-4-1 lr 0.03 seed 1: 100 %% after %d epochs (limit 60)".format(e))
        assertTrue(e in 1..60)
        val c = epochsTo(DataKind.CIRCLES, Trainer.NARROW, 1, 1.0, 300)
        println("BACKPROP circles 2-4-4-1 lr 0.03 seed 1: 100 %% after %d epochs".format(c))
        assertTrue(c in 1..300)
    }

    /** No hidden layer = one straight line: logistic regression cannot pass 75 % on XOR (the lesson's prediction). */
    @Test
    fun xorIsNotLinearlySeparable() {
        val d = Datasets.make(DataKind.XOR)
        val n = Net(intArrayOf(2, 1)).init(1); val t = n.newTrace(); val x = DoubleArray(2); val y = DoubleArray(1)
        var best = 0.0
        repeat(2000) {
            for (i in 0 until d.n) { x[0] = d.x1(i); x[1] = d.x2(i); y[0] = d.y[i]; n.forward(x, t); n.backward(t, y); n.sgd(t, 0.03) }
            var ok = 0
            for (i in 0 until d.n) { x[0] = d.x1(i); x[1] = d.x2(i); if ((n.forward(x, t) > 0.5) == (d.y[i] > 0.5)) ok++ }
            best = max(best, ok.toDouble() / d.n)
        }
        println("BACKPROP XOR with no hidden layer: best accuracy %.0f %% over 2000 epochs".format(best * 100))
        assertTrue(best <= 0.80)
    }

    /**
     * The challenge: spirals ≥ 95 % with the demo's own schedule. Budget = what 90 s at the fast speed gives
     * (6000 steps/s x 90 s / 100 points = 5400 epochs). 2-8-8-1 at lr 0.03 makes it; lr 0.3 overshoots.
     */
    @Test
    fun spiralsReachChallengeTarget() {
        val budget = Trainer.STEPS_PER_S[2] * 90 / (2 * Datasets.PER_CLASS)
        val wide = epochsTo(DataKind.SPIRALS, Trainer.WIDE, 1, 0.95, budget)
        val big = epochsTo(DataKind.SPIRALS, Trainer.WIDE, 2, 0.95, 2000)
        println("BACKPROP spirals 2-8-8-1 lr 0.03 seed 1: >= 95 %% after %d epochs (budget %d); lr 0.3: %s within 2000".format(
            wide, budget, if (big < 0) "never" else "$big epochs"))
        assertTrue(wide in 1..budget)
        assertEquals(-1, big)
    }

    @Test
    fun trainingIsDeterministic() {
        fun run(seed: Long): Trainer = Trainer().also { it.reset(DataKind.SPIRALS, 8, seed); it.lr = 0.03; it.trainEpochs(40) }
        val a = run(1); val b = run(1); val c = run(2)
        for (l in 0 until a.net.nL) { assertArrayEquals(a.net.w[l], b.net.w[l], 0.0); assertArrayEquals(a.net.b[l], b.net.b[l], 0.0) }
        assertTrue((0 until a.net.nL).any { l -> !a.net.w[l].contentEquals(c.net.w[l]) })
        assertEquals(Datasets.make(DataKind.XOR).x.toList(), Datasets.make(DataKind.XOR).x.toList())
        println("BACKPROP determinism: seed 1 twice -> identical weights (loss %.6f); seed 2 differs".format(a.evaluate().let { a.loss }))
    }

    @Test
    fun predictEqualsForward() {
        val n = randomNet(intArrayOf(2, 8, 8, 1), 5); val t = n.newTrace(); val buf = n.predictBuf()
        for (k in 0 until 20) { val x = doubleArrayOf(sin(k * 1.0), cos(k * 0.7)); assertEquals(n.forward(x, t), n.predict(x[0], x[1], buf), 1e-15) }
    }

    companion object {
        // ---- pasted from `python3 tools/ref_backprop.py` (PyTorch 2.11, float64) ----
        val P1 = 5.274724970003469e-01; val LOSS1 = 6.396585533791364e-01
        val GW1_0 = doubleArrayOf(-8.862168698330561e-02, 2.067839362943797e-01, -7.908309149007937e-02, 1.845272134768519e-01, 4.080491093867934e-02, -9.521145885691847e-02, 1.202696888972112e-01, -2.806292740934927e-01)
        val GB1_0 = doubleArrayOf(-2.954056232776854e-01, -2.636103049669313e-01, 1.360163697955978e-01, 4.008989629907039e-01)
        val NW1_0 = doubleArrayOf(6.876221868642746e-02, 5.705914443636381e-01, 2.643362372893058e-01, -4.725342185324422e-01, -5.034409564282087e-01, 1.964459639937186e-01, 5.873378362141420e-01, 1.617968758694978e-01)
        val GW1_1 = doubleArrayOf(8.038510611312756e-02, -1.176607441356817e-01, 8.708955448502953e-02, 3.578774235242626e-03, 2.728725301119225e-02, -3.994071352219285e-02, 2.956312211021541e-02, 1.214838453899282e-03, -6.613005463043405e-02, 9.679543654019589e-02, -7.164557309571845e-02, -2.944134145366705e-03, -6.198925118215319e-02, 9.073448770772607e-02, -6.715940961397415e-02, -2.759784065974432e-03)
        val GB1_1 = doubleArrayOf(-2.795290077444208e-01, -9.488796030890455e-02, 2.299588748059234e-01, 2.155597561740006e-01)
        val NW1_1 = doubleArrayOf(4.223751439284008e-01, 5.296916944028923e-01, -1.620336166646017e-01, -6.003118319619847e-01, -1.703780242204747e-01, 5.142560437293580e-01, 4.376819465134465e-01, -2.746430201105818e-01, -5.808936320277470e-01, -4.947268206474000e-02, 5.733819589760344e-01, 3.430125352105298e-01, -3.766650842905548e-01, -5.566229186454824e-01, 9.664226675916882e-02, 5.959358219081416e-01)
        val GW1_2 = doubleArrayOf(-5.086738683478630e-02, -1.427753398303602e-02, 4.426870181208379e-02, 5.999402089739884e-02)
        val GB1_2 = doubleArrayOf(-4.725275029996531e-01)
        val NW1_2 = doubleArrayOf(6.035837306459113e-01, 2.024206434918464e-01, -4.953931368198545e-01, -4.696580946233321e-01)
        val MAZ_W1 = doubleArrayOf(1.497807161327628e-01, 1.995614322655257e-01, 2.497511436323696e-01, 2.995022872647392e-01)
        val MAZ_W2 = doubleArrayOf(3.589164797178847e-01, 4.086661860762334e-01, 5.113012702387375e-01, 5.613701211079891e-01)
        val HAND_GW1 = doubleArrayOf(-5.605667024605712e-01, -2.802833512302856e-01, 3.557668686911613e-01, 1.778834343455806e-01)
        val HAND_GW2 = doubleArrayOf(-1.460543385341569e-01, -3.787635421537526e-01)
        val HAND_GB1 = doubleArrayOf(-5.605667024605712e-01, 3.557668686911613e-01)
        val HAND_NW1 = doubleArrayOf(7.802833512302856e-01, -3.598583243848572e-01, 7.211656565441937e-02, 9.110582828272097e-01)
        val HAND_NW2 = doubleArrayOf(1.073027169267079e+00, -8.106182289231236e-01)
    }
}
