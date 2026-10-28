// Spectrum — what colour IS a wavelength, pure Kotlin (shared by the prism and the double slit).
//
// The Unity app painted its six "rainbow" rays with hand-picked colours and no wavelengths, so
// nothing on screen could be checked. Here every colour comes from the CIE 1931 2-degree standard
// observer: wavelength -> (x̄, ȳ, z̄) -> linear sRGB, using the CIE table at 10 nm (linear
// interpolation). An analytic Gaussian fit was tried first and rejected: it is within 0.01 of the
// table in absolute terms but its relative tails are wrong, which turned 700 nm orange and made the
// violet end wobble in hue (PrismCoreTest checks the spectral locus and the hue order).
//
// Two uses, two normalisations:
//   * [weights]   — physically additive. White light is an equal-energy mix; each wavelength gets
//                   its share of each channel, white-balanced so the sum over the visible band is
//                   exactly (1,1,1). Sum these for fringes or a spectrum and white comes out white.
//                   Individual weights can be negative (spectral colours lie outside the sRGB
//                   gamut); clamp only the final sum.
//   * [display]   — the brightest in-gamut colour of ONE wavelength (for a monochromatic beam):
//                   negative channels clipped, max channel = 1. (Desaturating toward white
//                   instead was tried: it turns 650 nm into hot pink, sRGB (1, 0, 0.35).)

package com.bytedance.ailab.core.optics

import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

object Spectrum {
    /** The band every white-light sum runs over (nm) and its sampling step. */
    const val L_MIN = 380.0
    const val L_MAX = 730.0
    const val STEP = 2.5
    val GRID: DoubleArray = DoubleArray(((L_MAX - L_MIN) / STEP).toInt() + 1) { L_MIN + it * STEP }

    /** CIE 1931 2-degree colour matching functions, 380..780 nm every 10 nm: x̄, ȳ, z̄. */
    private val CIE = doubleArrayOf(
        0.001368, 0.000039, 0.006450, 0.004243, 0.000120, 0.020050, 0.014310, 0.000396, 0.067850,
        0.043510, 0.001210, 0.207400, 0.134380, 0.004000, 0.645600, 0.283900, 0.011600, 1.385600,
        0.348280, 0.023000, 1.747060, 0.336200, 0.038000, 1.772110, 0.290800, 0.060000, 1.669200,
        0.195360, 0.090980, 1.287640, 0.095640, 0.139020, 0.812950, 0.032010, 0.208020, 0.465180,
        0.004900, 0.323000, 0.272000, 0.009300, 0.503000, 0.158200, 0.063270, 0.710000, 0.078250,
        0.165500, 0.862000, 0.042160, 0.290400, 0.954000, 0.020300, 0.433450, 0.994950, 0.008750,
        0.594500, 0.995000, 0.003900, 0.762100, 0.952000, 0.002100, 0.916300, 0.870000, 0.001650,
        1.026300, 0.757000, 0.001100, 1.062200, 0.631000, 0.000800, 1.002600, 0.503000, 0.000340,
        0.854450, 0.381000, 0.000190, 0.642400, 0.265000, 0.000050, 0.447900, 0.175000, 0.000020,
        0.283500, 0.107000, 0.000000, 0.164900, 0.061000, 0.000000, 0.087400, 0.032000, 0.000000,
        0.046770, 0.017000, 0.000000, 0.022700, 0.008210, 0.000000, 0.011359, 0.004102, 0.000000,
        0.005790, 0.002091, 0.000000, 0.002899, 0.001047, 0.000000, 0.001440, 0.000520, 0.000000,
        0.000690, 0.000249, 0.000000, 0.000332, 0.000120, 0.000000, 0.000166, 0.000060, 0.000000,
        0.000083, 0.000030, 0.000000, 0.000042, 0.000015, 0.000000,
    )

    /** CIE 1931 2-degree colour matching functions (x̄, ȳ, z̄) at [lnm] nanometres (0 outside 380-780). */
    fun cmf(lnm: Double): DoubleArray {
        if (lnm < 380.0 || lnm > 780.0) return DoubleArray(3)
        val f = (lnm - 380.0) / 10.0
        val i = minOf(f.toInt(), 39); val t = f - i
        return DoubleArray(3) { CIE[i * 3 + it] * (1 - t) + CIE[(i + 1) * 3 + it] * t }
    }

    /** CIE XYZ -> linear sRGB (D65), IEC 61966-2-1. */
    fun xyzToRgb(x: Double, y: Double, z: Double) = doubleArrayOf(
        3.2406 * x - 1.5372 * y - 0.4986 * z,
        -0.9689 * x + 1.8758 * y + 0.0415 * z,
        0.0557 * x - 0.2040 * y + 1.0570 * z,
    )

    /** Unnormalised linear sRGB of unit power at [lnm] (may be negative: outside the gamut). */
    fun rawRgb(lnm: Double): DoubleArray { val c = cmf(lnm); return xyzToRgb(c[0], c[1], c[2]) }

    /** Per-channel white balance: sum of rawRgb over [GRID] (equal-energy white). */
    private val WHITE: DoubleArray by lazy {
        val w = DoubleArray(3)
        for (l in GRID) { val c = rawRgb(l); for (k in 0..2) w[k] += c[k] }
        w
    }

    /**
     * Additive, white-balanced share of one [GRID] sample: summing weights(l) over GRID gives
     * exactly (1, 1, 1). Multiply by [STEP]/dλ when integrating with a different step.
     */
    fun weights(lnm: Double): DoubleArray { val c = rawRgb(lnm); return DoubleArray(3) { c[it] / WHITE[it] } }

    /** Brightest displayable linear-sRGB colour of monochromatic light (max channel = 1). */
    fun display(lnm: Double): DoubleArray {
        val c = weights(lnm)
        for (k in 0..2) c[k] = c[k].coerceAtLeast(0.0)  // spectral colours lie outside sRGB: clip
        val hi = max(c[0], max(c[1], c[2])).coerceAtLeast(1e-12)
        return DoubleArray(3) { c[it] / hi }
    }

    /** sRGB transfer function (linear 0..1 -> encoded 0..1). */
    fun encode(l: Double): Double {
        val v = l.coerceIn(0.0, 1.0)
        return if (v <= 0.0031308) 12.92 * v else 1.055 * v.pow(1 / 2.4) - 0.055
    }

    /** HSV hue (deg, 0 = red, 120 = green, 240 = blue) of a linear colour — for order tests. */
    fun hueDeg(c: DoubleArray): Double {
        val r = c[0]; val g = c[1]; val b = c[2]
        val mx = max(r, max(g, b)); val mn = min(r, min(g, b)); val d = mx - mn
        if (d < 1e-12) return 0.0
        val h = when (mx) {
            r -> 60 * (((g - b) / d) % 6)
            g -> 60 * ((b - r) / d + 2)
            else -> 60 * ((r - g) / d + 4)
        }
        return (h + 360) % 360
    }

    /** Short Chinese colour name of a wavelength (Newton's seven-ish bands). */
    fun name(lnm: Double): String = when {
        lnm >= 620 -> "红"
        lnm >= 590 -> "橙"
        lnm >= 565 -> "黄"
        lnm >= 495 -> "绿"
        lnm >= 475 -> "青"
        lnm >= 440 -> "蓝"
        else -> "紫"
    }
}
