// GlyphAtlas — the character bitmaps the scene draws (token tiles, fan labels, digits).
//
// The 3-D scene cannot use the device's text labels for 30+ characters, so tools/train_gpt.py rasterises
// every vocabulary character plus printable ASCII into 48 x 48 alpha cells (system fonts; only the small
// bitmaps ship). File glyphs.bin (little-endian):
//   "GLY1" | int32 cell | int32 n | int32 codepoint[n] | int32 zlen | zlib( alpha[n][cell][cell], row 0 = top )
// Pure Kotlin + java.util.zip, so the JVM tests read exactly the shipped bytes.

package com.bytedance.ailab.core.gpt

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.Inflater

class GlyphAtlas(bytes: ByteArray) {
    val cell: Int
    val chars: IntArray
    /** alpha[glyph * cell * cell + y * cell + x], 0..255 */
    val alpha: ByteArray
    private val index = HashMap<Int, Int>()

    init {
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val magic = ByteArray(4).also { b.get(it) }
        require(String(magic, Charsets.US_ASCII) == "GLY1") { "not a GLY1 glyph file" }
        cell = b.int
        val n = b.int
        chars = IntArray(n) { b.int }
        val z = ByteArray(b.int).also { b.get(it) }
        alpha = ByteArray(n * cell * cell)
        val inf = Inflater()
        inf.setInput(z)
        var off = 0
        while (off < alpha.size && !inf.finished()) off += inf.inflate(alpha, off, alpha.size - off)
        inf.end()
        require(off == alpha.size) { "glyph data truncated ($off of ${alpha.size} bytes)" }
        chars.forEachIndexed { i, c -> index[c] = i }
    }

    val size: Int get() = chars.size
    /** Glyph slot of [c], or -1. The newline token is drawn as the return arrow U+21B5. */
    fun indexOf(c: Char): Int = index[if (c == '\n') 0x21B5 else c.code] ?: -1
    fun has(c: Char) = indexOf(c) >= 0
}
