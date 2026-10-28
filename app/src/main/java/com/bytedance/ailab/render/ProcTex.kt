// ProcTex — pure Kotlin port of the Unity app's NewtonLab.ProcTex, plus the bakes that REPLACE the
// Unity custom shaders. ZERO Android references: every bake returns a packed ARGB IntArray, so the
// same code runs in a JVM unit test (which dumps PNGs for review) and on the headset (which wraps
// the array in a Bitmap). The noise / colour maths is line-for-line the C#, with the same seeds, so
// the two stacks paint the same sun and the same planets — any visual difference is the renderer.
//
// WHY THE EXTRA BAKES. Unity drew the sun with three hand-written shaders (SunSurface, Corona,
// PlanetLit). SpatialSDK offers Unlit / PBR / ShaderGraph (editor-authored) materials, no runtime
// HLSL. Every view- or sun-dependent term in those shaders turns out to be a function of ONE angle,
// so it can be baked into a 1-D lookup and carried by geometry that is oriented every frame:
//   * limb darkening    I = 1 - u(1 - mu)          mu = cos(angle to VIEWER)  -> viewer-facing shell
//   * corona            sum_i I_i (1 - mu_i)^p_i   projected fresnel shells   -> viewer-facing disc
//   * planet day/night  Lambert(cos angle to SUN)                             -> sun-facing shell
// A black shell with alpha a over an unlit albedo gives  albedo * (1 - a),  i.e. exactly a
// multiplicative lighting term — no shader, no light units, no 1/r^2 falloff to fight.

package com.bytedance.ailab.render

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/** sRGB-encoded colour (same convention as Unity's gamma-space Color). */
data class Rgb(val r: Float, val g: Float, val b: Float, val a: Float = 1f)

/** A baked image: [w] x [h] packed 0xAARRGGBB, row 0 = top (Android Bitmap order). */
class Image(val w: Int, val h: Int, val px: IntArray = IntArray(w * h))

object ProcTex {
    const val T_PHOTO = 5772f
    const val T_UMBRA = 3800f
    const val T_PENUMBRA = 5450f
    fun stefanRatio(t: Float): Float = (t / T_PHOTO).pow(4f)

    // =========================================================== sun

    /**
     * The photosphere, fully shaded except for limb darkening: Unity's BakeSun channels
     * (R,G = two granulation phases, B = sunspot intensity), then FixGranulationPolarity, then the
     * SunSurface fragment maths at a fixed boil phase. Phase 0 and 1 are the two textures whose
     * cross-fade is the boiling convection; pass the same [raw] channels to both (the Worley pass
     * is ~95% of the cost and both phases come out of it).
     */
    fun bakeSunPhotosphere(w: Int, h: Int, phase: Int, seed: Int = 20260731,
                           raw: IntArray = bakeSunChannels(w, h, seed)): Image {
        // SunSurface + TameSunExposure: lane/core endpoints soft-clipped at exposure 1.7.
        val lane = softClip(Rgb(1.00f, 0.72f, 0.36f), 1.7f, 0.45f)
        val core = softClip(Rgb(1.00f, 0.97f, 0.93f), 1.7f, 0.45f)
        val out = Image(w, h)
        for (i in 0 until w * h) {
            val c = raw[i]
            // FixGranulationPolarity: invert the two granulation channels
            val gA = 1f - ((c ushr 16) and 0xFF) / 255f
            val gB = 1f - ((c ushr 8) and 0xFF) / 255f
            val spot = (c and 0xFF) / 255f
            val g = if (phase == 0) gA else gB
            var r = lerp(lane.r, core.r, g) * spot
            var gg = lerp(lane.g, core.g, g) * spot
            var b = lerp(lane.b, core.b, g) * spot
            // the shader's Reinhard-style rolloff (intensity already 1 after TameSunExposure)
            r = r / (1f + r * 0.72f) * 1.55f
            gg = gg / (1f + gg * 0.72f) * 1.55f
            b = b / (1f + b * 0.72f) * 1.55f
            out.px[i] = argb(1f, r, gg, b)
        }
        return out
    }

    /** Unity ProcTex.BakeSun verbatim: R,G granulation phases, B spot mask, A penumbra. */
    fun bakeSunChannels(w: Int, h: Int, seed: Int): IntArray {
        val px = IntArray(w * h)
        val cellFreq = 90f
        val rs = CsRandom(seed)
        val nSpots = 7
        val spotDir = Array(nSpots) { V3(0f, 0f, 0f) }
        val spotR = FloatArray(nSpots)
        for (i in 0 until nSpots) {
            var lat = ((rs.nextDouble() * 0.55 - 0.275) * PI).toFloat()
            if (rs.nextDouble() < 0.5) lat = -lat
            val lon = (rs.nextDouble() * 2.0 * PI).toFloat()
            spotDir[i] = sphereDir(lat, lon)
            spotR[i] = 0.055f + rs.nextDouble().toFloat() * 0.075f
        }
        parallelRows(h) { y ->
            // Same row->latitude rule as the C#. Which way up the image lands on the sphere does
            // not matter for the sun or the planets (both look right either way up).
            val lat = (PI * (0.5 - (y + 0.5) / h)).toFloat()
            for (x in 0 until w) {
                val lon = (2.0 * PI * ((x + 0.5) / w)).toFloat()
                val d = sphereDir(lat, lon)
                val a = granuleField(d, cellFreq, 0f, seed)
                val b = granuleField(d, cellFreq, 7.3f, seed)
                var spot = 1f
                var pen = 0f
                for (i in 0 until nSpots) {
                    val ang = angleRad(d, spotDir[i])
                    val r = spotR[i]
                    if (ang < r * 1.9f) {
                        val wob = 1f + 0.28f * (fbm(d * 9f, 3, seed + 77) - 0.5f)
                        val rr = r * wob
                        val umbra = 1f - smooth01((ang - rr * 0.55f) / (rr * 0.30f))
                        val penum = 1f - smooth01((ang - rr) / (rr * 0.65f))
                        var sI = lerp(1f, stefanRatio(T_UMBRA), umbra)
                        sI = min(sI, lerp(1f, stefanRatio(T_PENUMBRA), penum))
                        spot = min(spot, sI)
                        pen = max(pen, penum)
                    }
                }
                px[y * w + x] = (byte01(pen) shl 24) or (byte01(a) shl 16) or (byte01(b) shl 8) or byte01(spot)
            }
        }
        return px
    }

    private fun granuleField(d: V3, freq: Float, phase: Float, seed: Int): Float {
        val p = d * freq + V3(phase, phase, phase)
        val (f1, f2) = worley(p, seed)
        var cell = clamp01((f2 - f1) * 5.2f)
        cell = cell * cell * (3f - 2f * cell)
        val rough = fbm(d * (freq * 0.6f) + V3(phase, phase, phase), 3, seed + 11)
        return clamp01(cell * 0.86f + rough * 0.14f)
    }

    /**
     * Viewer-facing LIMB DARKENING shell, as a strip along U (u = 0 at the +Y pole = the point
     * facing the viewer). Shell strips vary along U only: texture V conventions differ between
     * engines (top-left vs bottom-left origin) but U always runs left to right, so a gradient on U
     * cannot be silently inverted into "night side lit". Black, alpha = u(1 - mu) with u = 0.62 (Unity _LimbDark). The shell sits
     * at 1.003 R so it never z-fights the photosphere; its last sliver beyond the photosphere's
     * silhouette fades to zero so it cannot draw a dark ring around the sun.
     */
    fun bakeLimbShell(n: Int, u: Float = 0.62f, shellScale: Float = 1.003f): Image {
        val h = 4
        val img = Image(n, h)
        for (x in 0 until n) {
            val theta = PI.toFloat() * (x + 0.5f) / n            // angle from the viewer axis
            val mu = max(0f, cos(theta))
            var a = u * (1f - mu)
            val projected = shellScale * sin(theta)               // in photosphere radii
            if (theta < PI / 2) a *= 1f - smooth01((projected - 0.993f) / (1.0f - 0.993f))
            else a = 0f                                           // back half: hidden anyway
            for (y in 0 until h) img.px[y * n + x] = argb(a, 0f, 0f, 0f)
        }
        return img
    }

    /**
     * The CORONA as a viewer-facing additive disc — SMOOTH, physically shaped.
     *
     * WHY NOT A LITERAL PORT. Projecting Unity's three Cull-Front fresnel shells (radius k_i R, rim
     * (1-mu)^p_i) gives a contribution that is BRIGHTEST at each shell's silhouette (mu -> 0) and
     * zero just outside it: three hard bright rings at 1.18, 1.55 and 2.30 R. [bakeCoronaShellsUnity]
     * keeps that exact projection for the A/B record; this bake replaces it.
     *
     * The radial profile is the Baumbach K-corona law (Allen, Astrophysical Quantities)
     *     I(rho) ~ 0.0532 rho^-2.5 + 1.425 rho^-7 + 2.565 rho^-17      (rho in solar radii)
     * which falls ~600x by 2.3 R — compressed with a 0.45 power so a display can show it — plus a
     * thin chromospheric rim right at the limb and gentle streamer structure that is brighter near
     * the solar equator (helmet streamers), as in every eclipse photograph.
     */
    fun bakeCoronaDisc(n: Int, seed: Int = 20260731): Image {
        val half = 2.30f
        val f1 = baumbach(1f)
        val img = Image(n, n)
        for (y in 0 until n) for (x in 0 until n) {
            val px = ((x + 0.5f) / n * 2f - 1f) * half
            val py = ((y + 0.5f) / n * 2f - 1f) * half
            val rho = sqrt(px * px + py * py)
            if (rho < 0.98f || rho >= half) { img.px[y * n + x] = argb(1f, 0f, 0f, 0f); continue }
            val r1 = max(rho, 1f)
            var b = 0.80f * (baumbach(r1) / f1).pow(0.45f)
            // streamers: angular structure, stronger toward the equator (|py| small)
            val ang = kotlin.math.atan2(py, px)
            val dir = V3(cos(ang), sin(ang), 0f)
            val streak = fbm(dir * 3.2f + V3(0f, 0f, r1 * 0.35f), 4, seed + 501)
            val equator = 0.55f + 0.45f * cos(ang).let { it * it }
            b *= lerp(0.70f, 1.35f, streak) * equator
            // window: reach exactly 0 at the disc edge, no square corners
            b *= 1f - smooth01((rho - 1.75f) / (half - 1.75f))
            // colour: pearly-gold near the limb, warmer and dimmer outward
            val t = clamp01((r1 - 1f) / 1.2f)
            var cr = lerp(1.00f, 1.00f, t) * b
            var cg = lerp(0.80f, 0.58f, t) * b
            var cb = lerp(0.52f, 0.30f, t) * b
            // chromosphere: a thin orange-red rim hugging the limb
            val chrom = kotlin.math.exp(-((rho - 1.0f) / 0.018f).let { it * it }) * 0.55f
            cr += chrom * 1.0f; cg += chrom * 0.42f; cb += chrom * 0.20f
            img.px[y * n + x] = argb(1f, min(1f, cr), min(1f, cg), min(1f, cb))
        }
        return img
    }

    private fun baumbach(rho: Float): Float =
        0.0532f * rho.pow(-2.5f) + 1.425f * rho.pow(-7f) + 2.565f * rho.pow(-17f)

    /** The literal projection of Unity's three corona shells (kept for the A/B record). */
    fun bakeCoronaShellsUnity(n: Int): Image {
        val k = floatArrayOf(1.18f, 1.55f, 2.30f)
        val inten = floatArrayOf(0.64f, 0.34f, 0.17f)
        val pw = floatArrayOf(2.2f, 3.2f, 4.5f)
        val col = arrayOf(Rgb(1.00f, 0.72f, 0.34f), Rgb(1.00f, 0.58f, 0.26f), Rgb(1.00f, 0.46f, 0.22f))
        val half = 2.30f
        val img = Image(n, n)
        for (y in 0 until n) for (x in 0 until n) {
            val px = ((x + 0.5f) / n * 2f - 1f) * half
            val py = ((y + 0.5f) / n * 2f - 1f) * half
            val rho = sqrt(px * px + py * py)
            var r = 0f; var g = 0f; var b = 0f
            for (i in 0 until 3) {
                if (rho >= k[i]) continue
                val q = rho / k[i]
                val mu = sqrt(max(0f, 1f - q * q))
                val a = (1f - mu).pow(pw[i]) * inten[i]
                r += col[i].r * a; g += col[i].g * a; b += col[i].b * a
            }
            img.px[y * n + x] = argb(1f, min(1f, r), min(1f, g), min(1f, b))
        }
        return img
    }

    // =========================================================== planets

    /** Unity ProcTex.BakePlanet verbatim (equirectangular albedo, OKLab stops). */
    fun bakePlanet(w: Int, h: Int, deep: Rgb, mid: Rgb, high: Rgb, bandiness: Float, seed: Int,
                   polarCaps: Boolean = false): Image {
        val img = Image(w, h)
        parallelRows(h) { y ->
            val v = (y + 0.5f) / h
            val lat = PI.toFloat() * (0.5f - v)
            for (x in 0 until w) {
                val lon = 2f * PI.toFloat() * ((x + 0.5f) / w)
                val d = sphereDir(lat, lon)
                val warp = (fbm(d * 2.6f, 4, seed) - 0.5f) * 0.55f
                var band = sin((sin(lat) * 7.5f + warp * 3f) * PI.toFloat())
                band = band * 0.5f + 0.5f
                val blotch = fbm(d * 3.4f, 5, seed + 31)
                val t = lerp(blotch, band, bandiness)
                var c = if (t < 0.5f) okLerp(deep, mid, t * 2f) else okLerp(mid, high, (t - 0.5f) * 2f)
                if (polarCaps) {
                    var capT = smooth01((abs(sin(lat)) - 0.80f) / 0.16f)
                    capT *= 0.55f + 0.45f * fbm(d * 8f, 3, seed + 5)
                    c = okLerp(c, Rgb(0.96f, 0.97f, 1f), clamp01(capT))
                }
                img.px[y * w + x] = argb(1f, c.r, c.g, c.b)
            }
        }
        return img
    }

    /**
     * Sun-facing LIGHTING shell, as a strip along U (u = 0 at the +Y pole = the sub-solar point).
     * Black, alpha = 1 - L(theta) with a slightly wrapped Lambert so the terminator has the soft
     * edge every real planet photo has, and an ambient floor so the night side is very dark but
     * not a hole. The same strip lights the Moon — so its phases are produced by geometry, not by
     * a phase-index lookup.
     */
    fun bakeLightShell(n: Int, ambient: Float = 0.006f, wrap: Float = 0.06f): Image {
        val h = 4
        val img = Image(n, h)
        for (x in 0 until n) {
            val theta = PI.toFloat() * (x + 0.5f) / n
            val lambert = clamp01((cos(theta) + wrap) / (1f + wrap))
            val l = ambient + (1f - ambient) * lambert
            for (y in 0 until h) img.px[y * n + x] = argb(1f - l, 0f, 0f, 0f)
        }
        return img
    }

    /**
     * EARTH that reads as Earth. The Unity palette (green/blue latitude bands, max albedo ~0.4) made
     * a dark striped ball whose day side barely beat its night side — the close-up review showed no
     * readable terminator. Real Earth from space is mostly bright: blue ocean, green/brown land,
     * ~67% cloud cover, white polar ice. Continents = thresholded low-frequency FBM; a desert band
     * near +-25 deg (Hadley descent); clouds = higher-frequency FBM through a soft threshold.
     */
    fun bakeEarth(w: Int, h: Int, seed: Int = 314): Image {
        val img = Image(w, h)
        val deep = Rgb(0.02f, 0.08f, 0.26f); val shallow = Rgb(0.06f, 0.26f, 0.46f)
        val forest = Rgb(0.13f, 0.30f, 0.11f); val grass = Rgb(0.33f, 0.45f, 0.20f); val desert = Rgb(0.72f, 0.60f, 0.40f)
        val ice = Rgb(0.93f, 0.95f, 0.98f)
        for (y in 0 until h) {
            val lat = PI.toFloat() * (0.5f - (y + 0.5f) / h)
            for (x in 0 until w) {
                val lon = 2f * PI.toFloat() * ((x + 0.5f) / w)
                val d = sphereDir(lat, lon)
                val cont = fbm(d * 1.7f, 5, seed)
                var c: Rgb
                if (cont > 0.53f) {
                    val alat = abs(lat) * 180f / PI.toFloat()
                    val dry = (1f - abs(alat - 24f) / 14f).coerceIn(0f, 1f) * smooth01((fbm(d * 4f, 3, seed + 9) - 0.35f) / 0.3f)
                    val lush = smooth01((fbm(d * 6f, 3, seed + 21) - 0.4f) / 0.3f)
                    c = okLerp(okLerp(forest, grass, lush), desert, dry)
                } else {
                    c = okLerp(deep, shallow, smooth01((cont - 0.40f) / 0.13f))
                }
                // polar ice
                val cap = smooth01((abs(sin(lat)) - 0.90f) / 0.06f)
                c = okLerp(c, ice, cap)
                // clouds: swirly, ~half cover, never fully opaque
                val cl = fbm(d * 3.3f + V3(0f, fbm(d * 1.1f, 2, seed + 4) * 1.5f, 0f), 5, seed + 13)
                val cover = smooth01((cl - 0.50f) / 0.20f) * 0.85f
                c = okLerp(c, Rgb(0.97f, 0.97f, 0.99f), cover)
                img.px[y * w + x] = argb(1f, c.r, c.g, c.b)
            }
        }
        return img
    }

    /** Unity BakeRingProfile: Saturn's C / B / Cassini / A / Encke opacity, radial in u. */
    fun bakeRingProfile(w: Int): Image {
        val h = 4
        val img = Image(w, h)
        val rng = CsRandom(20260731)
        val row = IntArray(w)
        for (i in 0 until w) {
            val f = (i + 0.5f) / w
            val rr = lerp(1.28f, 2.27f, f)
            var a = when {
                rr < 1.53f -> 0.16f
                rr < 1.95f -> 0.66f
                rr < 2.03f -> 0.05f
                else -> 0.44f
            }
            if (abs(rr - 2.214f) < 0.008f) a = 0.08f
            a *= 0.86f + 0.28f * rng.nextDouble().toFloat()
            a *= smoothStep(clamp01(f / 0.04f)) * smoothStep(clamp01((1f - f) / 0.05f))
            val l = clamp01(0.86f + 0.14f * rng.nextDouble().toFloat())
            row[i] = argb(clamp01(a), l, l, l)
        }
        for (y in 0 until h) System.arraycopy(row, 0, img.px, y * w, w)
        return img
    }

    /** Rows are independent (each writes only its own slice), so spread them over the cores.
     *  Deterministic: same pixels as the serial loop, in any thread order. */
    private inline fun parallelRows(h: Int, crossinline row: (Int) -> Unit) {
        java.util.stream.IntStream.range(0, h).parallel().forEach { row(it) }
    }

    // =========================================================== noise (verbatim C#)

    fun sphereDir(lat: Float, lon: Float) = V3(cos(lat) * cos(lon), sin(lat), cos(lat) * sin(lon))

    fun smooth01(t0: Float): Float { val t = clamp01(t0); return t * t * (3f - 2f * t) }
    private fun smoothStep(t: Float): Float = t * t * (3f - 2f * t)

    /** C# int arithmetic wraps exactly like Kotlin Int, so the hash is bit-identical. */
    fun hash(x: Int, y: Int, z: Int, seed: Int): Float {
        var h = seed
        h = h * 73856093 xor x * 19349663 xor y * 83492791 xor z * 2971215073u.toInt()
        h = h xor (h shr 13); h *= 1274126177; h = h xor (h shr 16)
        return (h and 0x7fffffff) / 0x7fffffff.toFloat()
    }

    private fun cellPoint(x: Int, y: Int, z: Int, seed: Int) =
        V3(x + hash(x, y, z, seed), y + hash(x, y, z, seed + 1), z + hash(x, y, z, seed + 2))

    fun worley(p: V3, seed: Int): Pair<Float, Float> {
        val xi = floor(p.x).toInt(); val yi = floor(p.y).toInt(); val zi = floor(p.z).toInt()
        var f1 = 1e9f; var f2 = 1e9f
        for (dz in -1..1) for (dy in -1..1) for (dx in -1..1) {
            val c = cellPoint(xi + dx, yi + dy, zi + dz, seed)
            val ex = c.x - p.x; val ey = c.y - p.y; val ez = c.z - p.z
            val d = ex * ex + ey * ey + ez * ez
            if (d < f1) { f2 = f1; f1 = d } else if (d < f2) f2 = d
        }
        return sqrt(f1) to sqrt(f2)
    }

    private fun valueNoise(p: V3, seed: Int): Float {
        val xi = floor(p.x).toInt(); val yi = floor(p.y).toInt(); val zi = floor(p.z).toInt()
        var fx = p.x - xi; var fy = p.y - yi; var fz = p.z - zi
        fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy); fz = fz * fz * (3 - 2 * fz)
        val c000 = hash(xi, yi, zi, seed); val c100 = hash(xi + 1, yi, zi, seed)
        val c010 = hash(xi, yi + 1, zi, seed); val c110 = hash(xi + 1, yi + 1, zi, seed)
        val c001 = hash(xi, yi, zi + 1, seed); val c101 = hash(xi + 1, yi, zi + 1, seed)
        val c011 = hash(xi, yi + 1, zi + 1, seed); val c111 = hash(xi + 1, yi + 1, zi + 1, seed)
        val x00 = lerp(c000, c100, fx); val x10 = lerp(c010, c110, fx)
        val x01 = lerp(c001, c101, fx); val x11 = lerp(c011, c111, fx)
        return lerp(lerp(x00, x10, fy), lerp(x01, x11, fy), fz)
    }

    fun fbm(p0: V3, octaves: Int, seed: Int): Float {
        var p = p0
        var sum = 0f; var amp = 0.5f; var norm = 0f
        for (i in 0 until octaves) {
            sum += valueNoise(p, seed + i * 101) * amp
            norm += amp; amp *= 0.5f; p = p * 2.02f
        }
        return if (norm > 0) sum / norm else 0f
    }

    // =========================================================== colour

    fun okLerp(a: Rgb, b: Rgb, t: Float): Rgb {
        val (l1, a1, b1) = okFrom(a)
        val (l2, a2, b2) = okFrom(b)
        return okTo(lerp(l1, l2, t), lerp(a1, a2, t), lerp(b1, b2, t))
    }

    private fun okFrom(c: Rgb): Triple<Float, Float, Float> {
        var l = 0.4122214708f * c.r + 0.5363325363f * c.g + 0.0514459929f * c.b
        var m = 0.2119034982f * c.r + 0.6806995451f * c.g + 0.1073969566f * c.b
        var s = 0.0883024619f * c.r + 0.2817188376f * c.g + 0.6299787005f * c.b
        l = cbrt(l); m = cbrt(m); s = cbrt(s)
        return Triple(
            0.2104542553f * l + 0.7936177850f * m - 0.0040720468f * s,
            1.9779984951f * l - 2.4285922050f * m + 0.4505937099f * s,
            0.0259040371f * l + 0.7827717662f * m - 0.8086757660f * s)
    }

    private fun okTo(lL: Float, aA: Float, bB: Float): Rgb {
        var l = lL + 0.3963377774f * aA + 0.2158037573f * bB
        var m = lL - 0.1055613458f * aA - 0.0638541728f * bB
        var s = lL - 0.0894841775f * aA - 1.2914855480f * bB
        l = l * l * l; m = m * m * m; s = s * s * s
        return Rgb(
            clamp01(4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s),
            clamp01(-1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s),
            clamp01(-0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s))
    }

    private fun cbrt(v: Float): Float = if (v <= 0f) 0f else v.pow(1f / 3f)

    fun softClip(c: Rgb, exposure: Float, shoulder: Float): Rgb {
        val r = c.r * exposure; val g = c.g * exposure; val b = c.b * exposure
        return Rgb(r / (1f + shoulder * r), g / (1f + shoulder * g), b / (1f + shoulder * b), c.a)
    }

    /** sRGB -> linear, for Color4 constants (textures are decoded by the GPU). */
    fun toLinear(c: Float): Float = if (c <= 0.04045f) c / 12.92f else ((c + 0.055f) / 1.055f).pow(2.4f)

    // =========================================================== small helpers

    fun lerp(a: Float, b: Float, t: Float) = a + (b - a) * t
    fun clamp01(v: Float) = if (v < 0f) 0f else if (v > 1f) 1f else v
    private fun byte01(v: Float) = (clamp01(v) * 255f).toInt()
    fun argb(a: Float, r: Float, g: Float, b: Float): Int =
        (byte01(a) shl 24) or (byte01(r) shl 16) or (byte01(g) shl 8) or byte01(b)

    private fun angleRad(a: V3, b: V3): Float {
        // Unity Vector3.Angle: acos(clamp(dot/(|a||b|)))
        val den = sqrt((a.x * a.x + a.y * a.y + a.z * a.z) * (b.x * b.x + b.y * b.y + b.z * b.z))
        if (den < 1e-15f) return 0f
        val d = ((a.x * b.x + a.y * b.y + a.z * b.z) / den).coerceIn(-1f, 1f)
        return acos(d)
    }
}

/** Minimal float vector for the bakes (kept separate from the SDK's Vector3 on purpose). */
data class V3(val x: Float, val y: Float, val z: Float) {
    operator fun times(s: Float) = V3(x * s, y * s, z * s)
    operator fun plus(o: V3) = V3(x + o.x, y + o.y, z + o.z)
}

/**
 * Port of .NET Framework / Mono System.Random (Knuth subtractive generator), so seeded bakes that
 * draw random numbers (sunspot placement, ring grain) land on the SAME values as the Unity build.
 */
class CsRandom(seed: Int) {
    private val seedArray = IntArray(56)
    private var inext = 0
    private var inextp = 21

    init {
        val subtraction = if (seed == Int.MIN_VALUE) Int.MAX_VALUE else abs(seed)
        var mj = 161803398 - subtraction
        seedArray[55] = mj
        var mk = 1
        for (i in 1 until 55) {
            val ii = (21 * i) % 55
            seedArray[ii] = mk
            mk = mj - mk
            if (mk < 0) mk += Int.MAX_VALUE
            mj = seedArray[ii]
        }
        for (k in 1 until 5) for (i in 1 until 56) {
            seedArray[i] -= seedArray[1 + (i + 30) % 55]
            if (seedArray[i] < 0) seedArray[i] += Int.MAX_VALUE
        }
    }

    private fun internalSample(): Int {
        var locINext = inext; var locINextp = inextp
        if (++locINext >= 56) locINext = 1
        if (++locINextp >= 56) locINextp = 1
        var retVal = seedArray[locINext] - seedArray[locINextp]
        if (retVal == Int.MAX_VALUE) retVal--
        if (retVal < 0) retVal += Int.MAX_VALUE
        seedArray[locINext] = retVal
        inext = locINext; inextp = locINextp
        return retVal
    }

    fun nextDouble(): Double = internalSample() * (1.0 / Int.MAX_VALUE)
}
