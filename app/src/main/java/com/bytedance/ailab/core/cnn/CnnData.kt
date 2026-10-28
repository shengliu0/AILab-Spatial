// CnnData — the shipped synthetic data (tools/train_cnn.py): held-out evaluation sets (shown by the demo and
// scored by the JVM test), pre-rendered label glyphs, and the training metrics. Pure Kotlin.

package com.bytedance.ailab.core.cnn

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * A uint8 image set: [labels] (digit 0-9, or a presence bitmask for the 5 scene classes) and pixels
 * [count][c][h][w], value = byte / 255. Layout: int32 'CNND', count, C, H, W; u8 labels; u8 pixels.
 */
class CnnSet(val count: Int, val c: Int, val h: Int, val w: Int, val labels: IntArray, private val px: ByteArray) {
    /** Sample [i] as a tensor with its first [channels] channels (an RGB-only sensor = drop NIR). */
    fun sample(i: Int, channels: Int = c): Tensor {
        val t = Tensor(channels, h, w)
        val base = i * c * h * w
        for (k in 0 until channels * h * w) t.d[k] = (px[base + k].toInt() and 0xFF) / 255f
        return t
    }

    companion object {
        const val MAGIC = 0x434E4E44   // 'CNND'
        fun parse(bytes: ByteArray): CnnSet {
            val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            require(bb.int == MAGIC) { "not a CNND set" }
            val n = bb.int; val c = bb.int; val h = bb.int; val w = bb.int
            val lab = IntArray(n) { bb.get().toInt() and 0xFF }
            val px = ByteArray(n * c * h * w); bb.get(px)
            require(bb.remaining() == 0) { "${bb.remaining()} trailing bytes" }
            return CnnSet(n, c, h, w, lab, px)
        }
    }
}

/** A pre-rendered text label (alpha mask, row 0 = top). */
class Glyph(val w: Int, val h: Int, val alpha: ByteArray) {
    fun a(x: Int, y: Int): Float = (alpha[y * w + x].toInt() and 0xFF) / 255f
}

object Glyphs {
    const val MAGIC = 0x434E4E47   // 'CNNG'
    fun parse(bytes: ByteArray): Map<String, Glyph> {
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        require(bb.int == MAGIC) { "not a CNNG glyph file" }
        val n = bb.int
        val out = LinkedHashMap<String, Glyph>()
        repeat(n) {
            val kb = ByteArray(bb.int); bb.get(kb)
            val w = bb.int; val h = bb.int
            val a = ByteArray(w * h); bb.get(a)
            out[String(kb, Charsets.US_ASCII)] = Glyph(w, h, a)
        }
        return out
    }
}

/** key=value lines written by the training script (held-out accuracies etc.). */
object Metrics {
    fun parse(text: String): Map<String, String> = text.lines().map { it.trim() }
        .filter { it.isNotEmpty() && !it.startsWith("#") && '=' in it }
        .associate { it.substringBefore('=') to it.substringAfter('=') }
}

/** Scene classes (bit c of a scene label), in the training script's order. */
object SceneClass {
    const val VEG = 0; const val WATER = 1; const val TURF = 2; const val ROOF = 3; const val SOIL = 4
    const val N = 5
    fun has(label: Int, c: Int) = (label shr c) and 1 == 1
}
