// CnnCore — a small convolutional network (LeNet-style) re-implemented in pure Kotlin, loading the weights
// that tools/train_cnn.py trained with PyTorch. Semantics are torch's exactly (checked against exported
// reference activations to 1e-4 in CnnCoreTest):
//   * Conv2d(C, K, 3): VALID cross-correlation (no kernel flip, no padding, stride 1)
//       y[k, i, j] = b[k] + Σ_c Σ_u Σ_v w[k, c, u, v] · x[c, i + u, j + v]
//   * max_pool2d(2): 2×2 windows, stride 2, floor (an odd last row/column is dropped)
//   * flatten in (c, h, w) order, Linear: z = W·x + b; softmax (digits) or per-label sigmoid (scenes)
// Network: C×28×28 → conv 8 → ReLU → pool → 8×13×13 → conv 16 → ReLU → pool → 16×5×5 → dense N.
// No Android imports: the same code runs in JVM tests and on the headset.

package com.bytedance.ailab.core.cnn

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp

/** A C×H×W tensor, row-major (channel, row, column), like a torch tensor without the batch dim. */
class Tensor(val c: Int, val h: Int, val w: Int, val d: FloatArray = FloatArray(c * h * w)) {
    init { require(d.size == c * h * w) { "tensor data ${d.size} != $c×$h×$w" } }
    operator fun get(ch: Int, y: Int, x: Int): Float = d[(ch * h + y) * w + x]
    operator fun set(ch: Int, y: Int, x: Int, v: Float) { d[(ch * h + y) * w + x] = v }
    /** "8×26×26" */
    val shape: String get() = "$c×$h×$w"
    fun maxAbs(): Float { var m = 0f; for (v in d) m = maxOf(m, kotlin.math.abs(v)); return m }
}

object CnnOps {
    /** Output side of a valid k×k convolution or a k×k / stride-s pooling: floor((n − k) / s) + 1. */
    fun outSize(n: Int, k: Int, s: Int = 1): Int = (n - k) / s + 1

    /** torch.nn.Conv2d(x.c, nOut, k) on [x]: [w] is [nOut][x.c][k][k], [b] is [nOut]. */
    fun conv(x: Tensor, w: FloatArray, b: FloatArray, nOut: Int, k: Int = 3): Tensor {
        val oh = outSize(x.h, k); val ow = outSize(x.w, k)
        require(w.size == nOut * x.c * k * k && b.size == nOut)
        val y = Tensor(nOut, oh, ow)
        for (o in 0 until nOut) {
            val base = o * oh * ow
            for (i in 0 until oh) for (j in 0 until ow) y.d[base + i * ow + j] = b[o]
            for (c in 0 until x.c) for (u in 0 until k) for (v in 0 until k) {
                val wt = w[((o * x.c + c) * k + u) * k + v]
                if (wt == 0f) continue
                for (i in 0 until oh) {
                    val xr = (c * x.h + i + u) * x.w + v
                    val yr = base + i * ow
                    for (j in 0 until ow) y.d[yr + j] += wt * x.d[xr + j]
                }
            }
        }
        return y
    }

    fun relu(x: Tensor): Tensor = Tensor(x.c, x.h, x.w, FloatArray(x.d.size) { maxOf(0f, x.d[it]) })

    /** max_pool2d(kernel 2, stride 2), floor mode. */
    fun maxPool2(x: Tensor): Tensor {
        val oh = x.h / 2; val ow = x.w / 2
        val y = Tensor(x.c, oh, ow)
        for (c in 0 until x.c) for (i in 0 until oh) for (j in 0 until ow) {
            y[c, i, j] = maxOf(maxOf(x[c, 2 * i, 2 * j], x[c, 2 * i, 2 * j + 1]), maxOf(x[c, 2 * i + 1, 2 * j], x[c, 2 * i + 1, 2 * j + 1]))
        }
        return y
    }

    /** z = W x + b, W is [n][x.size]. */
    fun dense(x: FloatArray, w: FloatArray, b: FloatArray, n: Int): FloatArray {
        require(w.size == n * x.size)
        return FloatArray(n) { o -> var s = b[o]; val r = o * x.size; for (i in x.indices) s += w[r + i] * x[i]; s }
    }

    fun softmax(z: FloatArray): FloatArray {
        val m = z.max(); val e = FloatArray(z.size) { exp((z[it] - m).toDouble()).toFloat() }; val s = e.sum()
        return FloatArray(z.size) { e[it] / s }
    }

    fun sigmoid(z: FloatArray): FloatArray = FloatArray(z.size) { (1.0 / (1.0 + exp(-z[it].toDouble()))).toFloat() }

    /**
     * What the sliding kernel computes at output (row [i], column [j]): one partial sum PER INPUT CHANNEL,
     * Σ_u Σ_v w[k, c, u, v] · x[c, i + u, j + v]. Their total + bias is conv(x)[k, i, j] — the channels are
     * mixed inside the kernel, which is why a 3×3×3 kernel makes ONE feature map, not three.
     */
    fun partials(x: Tensor, w: FloatArray, kIdx: Int, i: Int, j: Int, k: Int = 3): FloatArray =
        FloatArray(x.c) { c ->
            var s = 0f
            for (u in 0 until k) for (v in 0 until k) s += w[((kIdx * x.c + c) * k + u) * k + v] * x[c, i + u, j + v]
            s
        }
}

/** The trained network (weights as torch stores them). [sigmoidHead]: multi-label (scenes) instead of softmax. */
class CnnModel(
    val name: String, val cIn: Int, val k1: Int, val k2: Int, val nOut: Int, val sigmoidHead: Boolean,
    val w1: FloatArray, val b1: FloatArray, val w2: FloatArray, val b2: FloatArray, val wf: FloatArray, val bf: FloatArray,
) {
    /** Every intermediate the demo draws. conv1/conv2 are PRE-activation (ReLU shown as its own step). */
    class Acts(val input: Tensor, val conv1: Tensor, val pool1: Tensor, val conv2: Tensor, val pool2: Tensor,
               val logits: FloatArray, val probs: FloatArray) {
        fun top(): Int = probs.indices.maxBy { probs[it] }
    }

    fun forward(x: Tensor): Acts {
        require(x.c == cIn) { "$name expects $cIn channels, got ${x.c}" }
        val a1 = CnnOps.conv(x, w1, b1, k1)
        val p1 = CnnOps.maxPool2(CnnOps.relu(a1))
        val a2 = CnnOps.conv(p1, w2, b2, k2)
        val p2 = CnnOps.maxPool2(CnnOps.relu(a2))
        val z = CnnOps.dense(p2.d, wf, bf, nOut)
        return Acts(x, a1, p1, a2, p2, z, if (sigmoidHead) CnnOps.sigmoid(z) else CnnOps.softmax(z))
    }

    /** First-layer kernel [k] as [c][u][v] (cIn·9 floats). */
    fun kernel(k: Int): FloatArray = w1.copyOfRange(k * cIn * 9, (k + 1) * cIn * 9)

    /** Parameter count (weights + biases). */
    val params: Int get() = w1.size + b1.size + w2.size + b2.size + wf.size + bf.size

    companion object {
        const val MAGIC = 0x434E4E31   // 'CNN1'

        /** Parse the layout written by tools/train_cnn.py (little-endian int32 header + float32 tensors). */
        fun parse(name: String, bytes: ByteArray): CnnModel {
            val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            require(bb.int == MAGIC) { "$name: not a CNN1 file" }
            val c = bb.int; val k1 = bb.int; val k2 = bb.int; val n = bb.int; val head = bb.int
            fun f(len: Int) = FloatArray(len).also { bb.asFloatBuffer().get(it); bb.position(bb.position() + 4 * len) }
            val side = CnnOps.outSize(CnnOps.outSize(CnnOps.outSize(CnnOps.outSize(28, 3), 2, 2), 3), 2, 2)
            val m = CnnModel(name, c, k1, k2, n, head == 1, f(k1 * c * 9), f(k1), f(k2 * k1 * 9), f(k2), f(n * k2 * side * side), f(n))
            require(bb.remaining() == 0) { "$name: ${bb.remaining()} trailing bytes" }
            return m
        }
    }
}
