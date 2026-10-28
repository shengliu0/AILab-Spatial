// StrokeRaster — pen strokes on the drawing pad -> the 28x28 image the network reads, the way MNIST was made.
//
// MNIST (LeCun, Bottou, Bengio & Haffner 1998; yann.lecun.com/exdb/mnist): "the images were size normalized to
// fit in a 20x20 pixel box while preserving their aspect ratio ... centered in a 28x28 image by computing the
// center of mass of the pixels, and translating the image so as to position this point at the center".
//
// Here the strokes are VECTORS, so both steps are exact:
//   1. scale: the strokes' bounding box (longer side, at least MIN_SIDE pad units so a dot is not blown up)
//      maps to 20 - 2 PEN_R px, so the INK (centre line + pen radius) fits the 20x20 box;
//   2. draw every segment as a round-capped line of radius PEN_R px with a EDGE-px anti-aliasing ramp
//      (coverage = clamp((PEN_R + EDGE/2 - d) / EDGE, 0, 1), d = distance from the pixel centre to the segment,
//      union = max), pixel centres at (col + 0.5, row + 0.5);
//   3. centre: move the strokes so the centre of mass of the drawn pixels lands on (14, 14) and draw again
//      (COM_PASSES times: re-sampling after a sub-pixel shift moves the discrete centre by ~0.01 px).
// Stroke width is fixed in OUTPUT pixels, so a digit written large or small looks the same to the network.
// tools/train_mlp.py has the identical rasteriser (raster_strokes) and draws its pen training digits with it;
// StrokeRasterTest checks Kotlin == Python to 1e-4 on the shipped sample digits.

package com.bytedance.ailab.core.mlp

import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** Pen strokes in pad units (x right, y DOWN, 0..1), grown point by point while the user writes. */
class Ink {
    private val xs = ArrayList<FloatArray>()
    private val ns = ArrayList<Int>()
    val strokeCount get() = xs.size
    var pointCount = 0; private set

    fun clear() { xs.clear(); ns.clear(); pointCount = 0 }
    fun begin(x: Float, y: Float) { xs += FloatArray(64); ns += 0; add(x, y) }
    fun add(x: Float, y: Float) {
        val k = xs.size - 1; if (k < 0) { begin(x, y); return }
        var a = xs[k]; val n = ns[k]
        if (n * 2 + 2 > a.size) { a = a.copyOf(a.size * 2); xs[k] = a }
        a[n * 2] = x; a[n * 2 + 1] = y; ns[k] = n + 1; pointCount++
    }
    fun points(stroke: Int): Int = ns[stroke]
    fun x(stroke: Int, i: Int) = xs[stroke][i * 2]
    fun y(stroke: Int, i: Int) = xs[stroke][i * 2 + 1]
    /** Last point of the last stroke (null when empty). */
    fun last(): Pair<Float, Float>? { val k = xs.size - 1; if (k < 0 || ns[k] == 0) return null; return x(k, ns[k] - 1) to y(k, ns[k] - 1) }
    fun addStroke(p: FloatArray) { begin(p[0], p[1]); for (i in 1 until p.size / 2) add(p[i * 2], p[i * 2 + 1]) }
}

object StrokeRaster {
    const val GRID = 28
    const val BOX = 20.0
    const val CENTRE = 14.0
    const val PEN_R = 1.3
    const val EDGE = 1.5
    const val MIN_SIDE = 0.04
    const val COM_PASSES = 2

    /** What the normalisation did (for the UI and the tests). */
    class Info(var empty: Boolean = true, var scalePxPerUnit: Double = 0.0, var comX: Double = 0.0, var comY: Double = 0.0,
               var shiftX: Double = 0.0, var shiftY: Double = 0.0)

    /** Rasterise [ink] into [out] (784, row-major, row 0 = top). Returns what was done. */
    fun rasterise(ink: Ink, out: FloatArray = FloatArray(GRID * GRID), penR: Double = PEN_R, info: Info = Info()): FloatArray {
        out.fill(0f); info.empty = true
        if (ink.pointCount == 0) return out
        var lx = Double.MAX_VALUE; var ly = Double.MAX_VALUE; var hx = -Double.MAX_VALUE; var hy = -Double.MAX_VALUE
        for (s in 0 until ink.strokeCount) for (i in 0 until ink.points(s)) {
            val x = ink.x(s, i).toDouble(); val y = ink.y(s, i).toDouble()
            lx = min(lx, x); ly = min(ly, y); hx = max(hx, x); hy = max(hy, y)
        }
        val side = max(max(hx - lx, hy - ly), MIN_SIDE)
        val sc = (BOX - 2 * penR) / side
        val cx = (lx + hx) / 2; val cy = (ly + hy) / 2
        var ox = CENTRE; var oy = CENTRE
        val img = DoubleArray(GRID * GRID)
        render(ink, sc, cx, cy, ox, oy, penR, img)
        repeat(COM_PASSES) {
            var m = 0.0; var mx = 0.0; var my = 0.0
            for (r in 0 until GRID) for (c in 0 until GRID) { val v = img[r * GRID + c]; m += v; mx += v * (c + 0.5); my += v * (r + 0.5) }
            if (m <= 1e-9) return@repeat
            ox += CENTRE - mx / m; oy += CENTRE - my / m
            render(ink, sc, cx, cy, ox, oy, penR, img)
        }
        for (k in img.indices) out[k] = img[k].toFloat()
        val com = centreOfMass(out)
        info.empty = false; info.scalePxPerUnit = sc; info.comX = com[0]; info.comY = com[1]
        info.shiftX = ox - CENTRE; info.shiftY = oy - CENTRE
        return out
    }

    /** Convenience for strokes given as (x0, y0, x1, y1, ...) arrays. */
    fun rasterise(strokes: List<FloatArray>, penR: Double = PEN_R): FloatArray =
        rasterise(Ink().also { k -> strokes.forEach { k.addStroke(it) } }, penR = penR)

    private fun render(ink: Ink, sc: Double, cx: Double, cy: Double, ox: Double, oy: Double, penR: Double, img: DoubleArray) {
        img.fill(0.0)
        val reach = penR + EDGE / 2
        for (s in 0 until ink.strokeCount) {
            val n = ink.points(s)
            if (n == 0) continue
            val segs = if (n == 1) 1 else n - 1
            for (k in 0 until segs) {
                val ax = (ink.x(s, k) - cx) * sc + ox; val ay = (ink.y(s, k) - cy) * sc + oy
                val b = if (n == 1) k else k + 1
                val bx = (ink.x(s, b) - cx) * sc + ox; val by = (ink.y(s, b) - cy) * sc + oy
                segment(ax, ay, bx, by, penR, reach, img)
            }
        }
    }

    private fun segment(ax: Double, ay: Double, bx: Double, by: Double, penR: Double, reach: Double, img: DoubleArray) {
        val c0 = max(0, floor(min(ax, bx) - reach - 0.5).toInt()); val c1 = min(GRID - 1, ceil(max(ax, bx) + reach).toInt())
        val r0 = max(0, floor(min(ay, by) - reach - 0.5).toInt()); val r1 = min(GRID - 1, ceil(max(ay, by) + reach).toInt())
        if (c0 > c1 || r0 > r1) return
        val abx = bx - ax; val aby = by - ay
        val l2 = abx * abx + aby * aby
        for (r in r0..r1) for (c in c0..c1) {
            val px = c + 0.5; val py = r + 0.5
            var t = if (l2 > 1e-12) ((px - ax) * abx + (py - ay) * aby) / max(l2, 1e-12) else 0.0
            t = t.coerceIn(0.0, 1.0)
            val qx = ax + t * abx; val qy = ay + t * aby
            val dx = px - qx; val dy = py - qy
            val d = sqrt(dx * dx + dy * dy)
            val cov = ((penR + EDGE / 2 - d) / EDGE).coerceIn(0.0, 1.0)
            val i = r * GRID + c
            if (cov > img[i]) img[i] = cov
        }
    }

    /** (x, y) centre of mass in pixel units (pixel centres at i + 0.5); (14, 14) = MNIST centre. */
    fun centreOfMass(img: FloatArray): DoubleArray {
        var m = 0.0; var mx = 0.0; var my = 0.0
        for (r in 0 until GRID) for (c in 0 until GRID) { val v = img[r * GRID + c].toDouble(); m += v; mx += v * (c + 0.5); my += v * (r + 0.5) }
        return if (m <= 0.0) doubleArrayOf(CENTRE, CENTRE) else doubleArrayOf(mx / m, my / m)
    }

    /** Bounding box of pixels above [thr]: (col0, row0, col1, row1), or null when blank. */
    fun inkBox(img: FloatArray, thr: Float = 0.05f): IntArray? {
        var c0 = GRID; var r0 = GRID; var c1 = -1; var r1 = -1
        for (r in 0 until GRID) for (c in 0 until GRID) if (img[r * GRID + c] > thr) { c0 = min(c0, c); c1 = max(c1, c); r0 = min(r0, r); r1 = max(r1, r) }
        return if (c1 < 0) null else intArrayOf(c0, r0, c1, r1)
    }
}
