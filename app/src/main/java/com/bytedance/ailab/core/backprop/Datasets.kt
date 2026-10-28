// Datasets — three toy 2-D classification problems on the input plane [-1, 1]^2, generated deterministically
// (SplitMix64, fixed seeds), the classic trio of the TensorFlow Playground (Smilkov & Carter 2016):
//   XOR      four quadrant clusters, label 1 where x1*x2 > 0. No straight line separates them.
//   CIRCLES  a disc (label 1) inside a ring (label 0): needs a closed boundary.
//   SPIRALS  two interleaved spirals (Lang & Witbrock 1988, here [SPIRAL_TURNS] turns): needs many bends.
// Every set keeps a gap between the classes so 100 % training accuracy is reachable.

package com.bytedance.ailab.core.backprop

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

enum class DataKind { XOR, CIRCLES, SPIRALS }

/** n points: x = (x1, x2) interleaved, y in {0, 1}. */
class Data(val kind: DataKind, val x: DoubleArray, val y: DoubleArray) {
    val n get() = y.size
    fun x1(i: Int) = x[2 * i]
    fun x2(i: Int) = x[2 * i + 1]
}

object Datasets {
    const val SPIRAL_TURNS = 1.0
    const val PER_CLASS = 50

    fun make(kind: DataKind): Data = when (kind) {
        DataKind.XOR -> xor()
        DataKind.CIRCLES -> circles()
        DataKind.SPIRALS -> spirals()
    }

    private fun xor(): Data {
        val r = SplitMix(1001); val xs = ArrayList<Double>(); val ys = ArrayList<Double>()
        val per = PER_CLASS / 2
        for (q in 0 until 4) {
            val sx = if (q == 0 || q == 3) 1.0 else -1.0; val sy = if (q < 2) 1.0 else -1.0
            for (k in 0 until per) {
                val a = sx * (0.16 + 0.74 * r.nextDouble()); val b = sy * (0.16 + 0.74 * r.nextDouble())
                xs += a; xs += b; ys += if (a * b > 0) 1.0 else 0.0
            }
        }
        return Data(DataKind.XOR, xs.toDoubleArray(), ys.toDoubleArray())
    }

    private fun circles(): Data {
        val r = SplitMix(2002); val xs = ArrayList<Double>(); val ys = ArrayList<Double>()
        for (k in 0 until PER_CLASS) {
            val a = 2 * PI * r.nextDouble(); val rad = 0.42 * sqrt(r.nextDouble())
            xs += rad * cos(a); xs += rad * sin(a); ys += 1.0
        }
        for (k in 0 until PER_CLASS) {
            val a = 2 * PI * r.nextDouble(); val rad = 0.64 + 0.28 * r.nextDouble()
            xs += rad * cos(a); xs += rad * sin(a); ys += 0.0
        }
        return Data(DataKind.CIRCLES, xs.toDoubleArray(), ys.toDoubleArray())
    }

    fun spirals(turns: Double = SPIRAL_TURNS): Data {
        val xs = ArrayList<Double>(); val ys = ArrayList<Double>()
        for (c in 0..1) for (k in 0 until PER_CLASS) {
            val t = (k + 0.5) / PER_CLASS
            val rad = 0.12 + 0.80 * t
            val a = turns * 2 * PI * t + c * PI
            xs += rad * cos(a); xs += rad * sin(a); ys += if (c == 0) 1.0 else 0.0
        }
        return Data(DataKind.SPIRALS, xs.toDoubleArray(), ys.toDoubleArray())
    }
}
