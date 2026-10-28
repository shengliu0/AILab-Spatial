// Trainer — plain stochastic gradient descent, one sample at a time, exactly what the demo animates:
//   pick a sample (shuffled order each epoch) -> forward -> backward -> w <- w - lr * dL/dw.
// No momentum, no Adam, no batches: "Train" is nothing but the single step repeated. The demo and the tests run
// the SAME schedule (seeded weights, seeded shuffles), so "XOR reaches 100 % in N epochs" is a property of what
// the user sees, not of a different script.

package com.bytedance.ailab.core.backprop

import kotlin.math.abs

class Trainer {
    var kind = DataKind.XOR; private set
    var width = 4; private set
    var seed = 1L; private set
    var lr = 0.1

    lateinit var data: Data; private set
    lateinit var net: Net; private set
    lateinit var trace: Net.Trace; private set
    /** Second trace for read-only probes (the readout's chain rule for a picked neuron). */
    lateinit var probeTrace: Net.Trace; private set
    private lateinit var buf: DoubleArray

    /** Completed epochs and SGD steps so far. */
    var epoch = 0; private set
    var steps = 0L; private set
    private var cursor = 0
    private var order = IntArray(0)
    private var rng = SplitMix(0)
    private val xi = DoubleArray(2)
    private val yi = DoubleArray(1)

    /** Running mean of |dL/dw| per weight (exponential, ~1 epoch) — what the gradient threads show while training. */
    lateinit var gAvg: Array<DoubleArray>; private set

    /** Last evaluation over the whole set. */
    var loss = 0.0; private set
    var accuracy = 0.0; private set

    val sizes: IntArray get() = intArrayOf(2, width, width, 1)

    /** Fresh data + fresh seeded weights (Xavier uniform). */
    fun reset(kind: DataKind, width: Int, seed: Long) {
        this.kind = kind; this.width = width; this.seed = seed
        data = Datasets.make(kind)
        net = Net(sizes).init(seed)
        trace = net.newTrace(); probeTrace = net.newTrace(); buf = net.predictBuf()
        gAvg = Array(net.nL) { DoubleArray(net.w[it].size) }
        epoch = 0; steps = 0; cursor = 0
        rng = SplitMix(seed * 7919L + 17L)
        order = IntArray(data.n) { it }; shuffle()
        evaluate()
    }

    private fun shuffle() { for (i in order.size - 1 downTo 1) { val j = rng.nextInt(i + 1); val t = order[i]; order[i] = order[j]; order[j] = t } }

    private fun load(i: Int) { xi[0] = data.x1(i); xi[1] = data.x2(i); yi[0] = data.y[i] }

    /** Forward + backward on sample [i] WITHOUT changing the weights (fills [t]). */
    fun probe(i: Int, t: Net.Trace = trace): Net.Trace { load(i); net.forward(xi, t); net.backward(t, yi); return t }

    /** One SGD step on sample [i]; [trace] keeps the pre-update forward values, deltas and gradients. */
    fun stepOn(i: Int): Net.Trace { probe(i); applyStep(trace); return trace }

    /** The update half of a step: w <- w - lr * dL/dw with the gradients a [probe] left in [t]. */
    fun applyStep(t: Net.Trace = trace) {
        net.sgd(t, lr); steps++
        for (l in 0 until net.nL) { val g = t.gw[l]; val m = gAvg[l]; for (k in m.indices) m[k] += EMA * (abs(g[k]) - m[k]) }
    }

    /** [n] SGD steps continuing the shuffled epoch order. */
    fun trainSteps(n: Int) {
        repeat(n) {
            stepOn(order[cursor])
            if (++cursor >= order.size) { cursor = 0; epoch++; shuffle() }
        }
    }

    fun trainEpochs(e: Int) = trainSteps(e * data.n)

    /** Fraction of the current epoch done (0..1). */
    val epochFrac get() = if (order.isEmpty()) 0.0 else cursor.toDouble() / order.size

    fun predict(x1: Double, x2: Double) = net.predict(x1, x2, buf)

    /** Mean BCE loss and accuracy (p > 0.5 <=> y = 1) over the whole set. */
    fun evaluate(): Double {
        var l = 0.0; var ok = 0
        for (i in 0 until data.n) {
            val p = predict(data.x1(i), data.x2(i)); val y = data.y[i]
            val pc = p.coerceIn(1e-15, 1 - 1e-15)
            l -= y * kotlin.math.ln(pc) + (1 - y) * kotlin.math.ln(1 - pc)
            if ((p > 0.5) == (y > 0.5)) ok++
        }
        loss = l / data.n; accuracy = ok.toDouble() / data.n
        return accuracy
    }

    /** Sample with the largest loss under the current weights (the point the net gets most wrong). */
    fun worstSample(): Int {
        var best = 0; var bl = -1.0
        for (i in 0 until data.n) { val e = abs(predict(data.x1(i), data.x2(i)) - data.y[i]); if (e > bl) { bl = e; best = i } }
        return best
    }

    companion object {
        /** The three learning rates on the panel: too small (slow), right, too large (spirals oscillate). */
        val LRS = doubleArrayOf(0.003, 0.03, 0.3)
        /** SGD steps per second at the three speeds (slow / medium / fast); fast = 60 epochs/s on 100 points. */
        val STEPS_PER_S = intArrayOf(50, 300, 6000)
        const val SEED = 1L
        const val NARROW = 4
        const val WIDE = 8
        /** Gradient-display smoothing per step (time constant ~ 1/EMA = 50 steps, half an epoch). */
        const val EMA = 0.02
    }
}
