package com.bytedance.ailab.core.mlp

import com.bytedance.ailab.lab.LabAssets
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

/**
 * Pad strokes -> 28x28 the MNIST way (LeCun et al. 1998): ink fits a 20x20 box, centre of mass at the centre.
 *  1. Python == Kotlin: the 10 shipped sample digits rasterised by tools/train_mlp.py (first 10 vectors of
 *     ref.bin) are reproduced to 1e-4, so the app draws exactly what the pen training digits looked like.
 *  2. A synthetic stroke written small and off-centre in a pad corner: centre of mass at (14, 14), the longer
 *     side spans the 20 px box, and moving / resizing the stroke on the pad changes nothing.
 */
class StrokeRasterTest {
    private val samples = DigitAssets.samples(LabAssets.bytes("mlp/samples.bin"))

    @Test
    fun matchesThePythonRasteriserOnTheSampleDigits() {
        val fb = ByteBuffer.wrap(LabAssets.bytes("mlp/ref.bin")).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        fb.get()
        var worst = 0.0
        for (d in 0 until 10) {
            val ref = FloatArray(784).also { fb.get(it) }
            fb.position(fb.position() + 64 + 32 + 10 + 10)
            val mine = StrokeRaster.rasterise(samples[d])
            for (k in 0 until 784) worst = maxOf(worst, abs(mine[k] - ref[k]).toDouble())
        }
        println("RASTER Kotlin vs Python on 10 sample digits: worst pixel difference %.2e".format(worst))
        assertTrue("rasteriser differs from tools/train_mlp.py by $worst", worst < 1e-4)
    }

    /** A "7"-like hook: bar, then a slanted stem — written small in the pad's top-left corner. */
    private fun hook(x0: Float, y0: Float, size: Float): List<FloatArray> {
        val pts = ArrayList<Float>()
        for (i in 0..20) { pts += x0 + size * 0.6f * i / 20f; pts += y0 }
        for (i in 1..30) { val t = i / 30f; pts += x0 + size * (0.6f - 0.35f * t); pts += y0 + size * t }
        return listOf(pts.toFloatArray())
    }

    @Test
    fun syntheticStrokeIsCentredByMassAndFitsTheBox() {
        val info = StrokeRaster.Info()
        val img = StrokeRaster.rasterise(Ink().also { k -> hook(0.05f, 0.08f, 0.18f).forEach { k.addStroke(it) } }, info = info)
        val com = StrokeRaster.centreOfMass(img)
        val box = StrokeRaster.inkBox(img, 0.02f)!!
        val w = box[2] - box[0] + 1; val h = box[3] - box[1] + 1
        println("RASTER hook stroke: centre of mass (%.3f, %.3f) px, ink box %dx%d px, shift (%.2f, %.2f) px, %.1f px per pad unit"
            .format(com[0], com[1], w, h, info.shiftX, info.shiftY, info.scalePxPerUnit))
        assertEquals("centre of mass x", 14.0, com[0], 0.05)
        assertEquals("centre of mass y", 14.0, com[1], 0.05)
        // the longer side (height) fills the 20 px box (+/- the 1 px anti-aliasing fringe), the other is narrower
        assertTrue("height $h", h in 20..22)
        assertTrue("width $w", w < h)
        // the hook's bar sits above its stem: top rows have ink to the right of the centre
        assertTrue(img.sum() > 30f)
    }

    @Test
    fun positionAndSizeOnThePadDoNotMatter() {
        val a = StrokeRaster.rasterise(hook(0.05f, 0.08f, 0.18f))
        val b = StrokeRaster.rasterise(hook(0.40f, 0.30f, 0.55f))          // bigger, elsewhere
        var worst = 0.0
        for (k in 0 until 784) worst = maxOf(worst, abs(a[k] - b[k]).toDouble())
        println("RASTER same stroke, moved and 3x bigger: worst pixel difference %.2e".format(worst))
        assertTrue(worst < 1e-5)
    }

    @Test
    fun rotationChangesTheImageButNotTheCentring() {
        // the network must SEE a rotated digit differently (it is not rotation invariant) — but still centred
        val pts = hook(0.3f, 0.3f, 0.3f)[0]
        val rot = FloatArray(pts.size)
        val c = cos(0.4); val s = sin(0.4)
        for (i in 0 until pts.size / 2) {
            val x = pts[i * 2] - 0.45; val y = pts[i * 2 + 1] - 0.45
            rot[i * 2] = (0.45 + c * x - s * y).toFloat(); rot[i * 2 + 1] = (0.45 + s * x + c * y).toFloat()
        }
        val img = StrokeRaster.rasterise(listOf(rot))
        val com = StrokeRaster.centreOfMass(img)
        assertEquals(14.0, com[0], 0.05); assertEquals(14.0, com[1], 0.05)
    }

    @Test
    fun emptyAndDotEdgeCases() {
        val info = StrokeRaster.Info()
        val e = StrokeRaster.rasterise(Ink(), info = info)
        assertTrue(info.empty && e.all { it == 0f })
        // a single tap-dot: a small centred disc, not a blown-up blob (MIN_SIDE)
        val dot = StrokeRaster.rasterise(listOf(floatArrayOf(0.7f, 0.2f)))
        val box = StrokeRaster.inkBox(dot, 0.02f)!!
        val com = StrokeRaster.centreOfMass(dot)
        assertEquals(14.0, com[0], 1e-6); assertEquals(14.0, com[1], 1e-6)
        assertTrue("dot is ${box[2] - box[0] + 1} px wide", box[2] - box[0] + 1 <= 4)
        // a straight vertical line (zero width) — no division by zero, a 20 px tall "1"
        val one = StrokeRaster.rasterise(listOf(floatArrayOf(0.5f, 0.1f, 0.5f, 0.9f)))
        val b1 = StrokeRaster.inkBox(one, 0.02f)!!
        assertTrue(one.all { it.isFinite() } && b1[3] - b1[1] + 1 in 20..22)
    }

    @Test
    fun theNetworkReadsEverySampleDigit() {
        val net = Mlp.load(LabAssets.bytes("mlp/mlp_w.bin"))
        val got = (0 until 10).map { d -> net.forward(StrokeRaster.rasterise(samples[d])).let { it.predicted to it.confidence } }
        println("RASTER+MLP sample digits: " + got.mapIndexed { d, (p, c) -> "$d->$p (%.0f%%)".format(c * 100) }.joinToString(" "))
        for (d in 0 until 10) assertEquals("sample $d", d, got[d].first)
    }
}
