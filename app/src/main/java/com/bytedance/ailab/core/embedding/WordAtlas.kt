// WordAtlas — pre-rendered word labels for the embedding demo's 3-D scene (one strip per word and language).
//
// tools/train_embedding.py rasterises every shown word in Chinese (Heiti SC Medium) and English (Helvetica
// Bold) from the macOS system fonts into one alpha atlas; only the bitmaps ship. File words.bin (LE):
//   "WRD1" | int32 W, H, n | n x int32 (x, y, w, h) | int32 zlen | zlib(alpha[H][W], row 0 = top)
// entry 2i = word i in Chinese, 2i+1 = in English; w = 0 for hidden (never drawn) words.

package com.bytedance.ailab.core.embedding

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.Inflater

class WordAtlas(bytes: ByteArray) {
    val w: Int
    val h: Int
    /** rects[e] = (x, y, w, h) of entry e in pixels. */
    val rects: Array<IntArray>
    /** alpha[y * w + x], 0..255 */
    val alpha: ByteArray

    init {
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val magic = ByteArray(4).also { b.get(it) }
        require(String(magic, Charsets.US_ASCII) == "WRD1") { "not a WRD1 atlas" }
        w = b.int; h = b.int; val n = b.int
        rects = Array(n) { IntArray(4) { b.int } }
        val z = ByteArray(b.int).also { b.get(it) }
        alpha = ByteArray(w * h)
        val inf = Inflater(); inf.setInput(z)
        var off = 0
        while (off < alpha.size && !inf.finished()) off += inf.inflate(alpha, off, alpha.size - off)
        inf.end()
        require(off == alpha.size) { "atlas truncated ($off of ${alpha.size})" }
    }

    fun rect(word: Int, english: Boolean): IntArray = rects[word * 2 + if (english) 1 else 0]
}
