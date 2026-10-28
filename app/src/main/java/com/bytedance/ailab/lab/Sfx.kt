// Sfx — short sound effects for every lab, synthesised in code (no audio files, no licences): written once
// as 16-bit WAVs into the cache, then played through SoundPool (low latency, many voices). A demo calls
// `Sfx.play(Sfx.CLACK, gain, pitch, pan)`; on the JVM (tests) nothing is initialised and play() is a no-op.
//
// Rate limiting: a sound fired faster than [MIN_GAP_MS] is merged (the colliding blocks reach thousands of
// collisions per second — the ear hears a buzz either way, and SoundPool would drop voices).

package com.bytedance.ailab.lab

import android.content.Context
import android.media.AudioAttributes
import android.media.SoundPool
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.sin

object Sfx {
    // ---- the palette (names are what demos pass) ----
    /** Hard wooden clack — colliding blocks, pendulum bob hitting a stop. */
    const val CLACK = "clack"
    /** Soft UI confirmation click (every pinch / panel tap gets one from LabHost). */
    const val TAP = "tap"
    /** Low muted "nope" for a greyed-out action. */
    const val DENY = "deny"
    /** Tiny tick — a ball passing a peg, a counter step. */
    const val TICK = "tick"
    /** Round pop — a bubble bursting, a piece snapping into place. */
    const val POP = "pop"
    /** Bright bell — success, a lesson step done, a challenge solved. */
    const val CHIME = "chime"
    /** Soft rising sweep — launch, release, a comet thrown. */
    const val WHOOSH = "whoosh"
    /** Gentle fizzing noise (~1 s) — gas, foam, electrolysis bubbles. */
    const val FIZZ = "fizz"
    /** Short electric crackle — sparks, a discharge, an alpha hit. */
    const val ZAP = "zap"
    /** Deep soft thump — something heavy lands, a weight on a scale. */
    const val THUD = "thud"
    /** Glassy ding — glassware, a drop landing in a flask. */
    const val GLASS = "glass"

    private const val RATE = 44100
    private const val MIN_GAP_MS = 18L
    private var pool: SoundPool? = null
    private val ids = HashMap<String, Int>()
    private val last = HashMap<String, Long>()
    /** Master switch (top bar 🔊); remembered by LabSettings. */
    @Volatile var enabled = true

    fun init(context: Context) {
        if (pool != null) return
        val p = SoundPool.Builder().setMaxStreams(12).setAudioAttributes(
            AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_GAME).setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION).build()
        ).build()
        pool = p
        val dir = File(context.cacheDir, "sfx_v1").apply { mkdirs() }
        for ((name, gen) in synths) {
            val f = File(dir, "$name.wav")
            if (!f.isFile || f.length() < 100) writeWav(f, gen())
            ids[name] = p.load(f.path, 1)
        }
    }

    /**
     * Play [name]. [gain] 0..1, [pitch] 0.5..2 (playback rate), [pan] -1 (left) .. +1 (right) — use the
     * object's x in the volume for a sense of where the sound comes from.
     */
    fun play(name: String, gain: Float = 1f, pitch: Float = 1f, pan: Float = 0f) {
        val p = pool ?: return
        if (!enabled) return
        val id = ids[name] ?: return
        val now = System.currentTimeMillis()
        if (now - (last[name] ?: 0L) < MIN_GAP_MS) return
        last[name] = now
        val g = gain.coerceIn(0f, 1f); val pn = pan.coerceIn(-1f, 1f)
        p.play(id, g * minOf(1f, 1f - pn), g * minOf(1f, 1f + pn), 1, 0, pitch.coerceIn(0.5f, 2f))
    }

    // ---------------------------------------------------------------- synthesis (mono float -1..1)
    private fun buf(seconds: Float) = FloatArray((seconds * RATE).toInt())
    private val rnd = java.util.Random(7)
    private fun noise() = rnd.nextFloat() * 2f - 1f

    /** Damped sinusoid partials + optional noise transient. */
    private fun partials(seconds: Float, parts: List<Triple<Float, Float, Float>>, noiseAmp: Float = 0f, noiseDecay: Float = 400f): FloatArray {
        val b = buf(seconds)
        for (i in b.indices) {
            val t = i.toFloat() / RATE
            var v = 0f
            for ((f, a, d) in parts) v += a * exp(-d * t) * sin(2 * PI.toFloat() * f * t)
            if (noiseAmp > 0f) v += noiseAmp * exp(-noiseDecay * t) * noise()
            b[i] = v
        }
        return norm(b, 0.9f)
    }

    private fun norm(b: FloatArray, peak: Float): FloatArray {
        // 1 ms fade-in / 3 ms fade-out (no clicks at the ends) BEFORE normalising, so the peak is exact
        val fin = (0.001f * RATE).toInt(); val fout = (0.003f * RATE).toInt()
        for (i in 0 until minOf(fin, b.size)) b[i] *= i.toFloat() / fin
        for (i in 0 until minOf(fout, b.size)) b[b.size - 1 - i] *= i.toFloat() / fout
        var m = 1e-6f; for (v in b) m = maxOf(m, kotlin.math.abs(v))
        val k = peak / m; for (i in b.indices) b[i] *= k
        return b
    }

    /** Every sound rendered (tests check they are finite, non-empty and peak-normalised). */
    internal fun renderAll(): Map<String, FloatArray> = synths.associate { (n, g) -> n to g() }

    private val synths: List<Pair<String, () -> FloatArray>> = listOf(
        CLACK to { partials(0.09f, listOf(Triple(2200f, 1f, 90f), Triple(3450f, 0.5f, 120f), Triple(900f, 0.3f, 60f)), 0.6f, 900f) },
        TAP to { partials(0.05f, listOf(Triple(1800f, 0.6f, 160f), Triple(2700f, 0.3f, 220f)), 0.15f, 1500f) },
        DENY to { partials(0.16f, listOf(Triple(180f, 1f, 22f), Triple(240f, 0.6f, 30f))) },
        TICK to { partials(0.03f, listOf(Triple(4200f, 0.5f, 300f)), 0.4f, 2000f) },
        POP to {
            val b = buf(0.08f)
            for (i in b.indices) { val t = i.toFloat() / RATE; val f = 900f - 5000f * t; b[i] = exp(-60f * t) * sin(2 * PI.toFloat() * f * t) }
            norm(b, 0.9f)
        },
        CHIME to { partials(0.9f, listOf(Triple(1318f, 1f, 4f), Triple(1976f, 0.55f, 5f), Triple(2637f, 0.35f, 7f), Triple(3951f, 0.15f, 10f))) },
        WHOOSH to {
            val b = buf(0.45f); var lp = 0f
            for (i in b.indices) { val t = i.toFloat() / RATE; val env = sin(PI.toFloat() * t / 0.45f); val a = 0.02f + 0.25f * t / 0.45f; lp += a * (noise() - lp); b[i] = env * lp }
            norm(b, 0.8f)
        },
        FIZZ to {
            val b = buf(1.0f); var hp = 0f; var prev = 0f
            for (i in b.indices) {
                val t = i.toFloat() / RATE; val x = noise(); hp = 0.85f * (hp + x - prev); prev = x
                val crackle = if (rnd.nextFloat() < 0.004f) 3f else 1f
                b[i] = hp * crackle * (1f - exp(-30f * t)) * exp(-1.2f * t)
            }
            norm(b, 0.6f)
        },
        ZAP to {
            val b = buf(0.12f)
            for (i in b.indices) { val t = i.toFloat() / RATE; val sq = if (sin(2 * PI.toFloat() * 120f * t) > 0) 1f else -1f; b[i] = (0.6f * sq + noise()) * exp(-30f * t) }
            norm(b, 0.7f)
        },
        THUD to { partials(0.25f, listOf(Triple(70f, 1f, 18f), Triple(110f, 0.5f, 25f)), 0.3f, 200f) },
        GLASS to { partials(0.6f, listOf(Triple(2350f, 1f, 6f), Triple(5200f, 0.4f, 9f), Triple(7900f, 0.2f, 14f))) },
    )

    private fun writeWav(f: File, s: FloatArray) {
        val data = ByteBuffer.allocate(s.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (v in s) data.putShort((v.coerceIn(-1f, 1f) * 32767).toInt().toShort())
        val h = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        h.put("RIFF".toByteArray()); h.putInt(36 + s.size * 2); h.put("WAVE".toByteArray())
        h.put("fmt ".toByteArray()); h.putInt(16); h.putShort(1); h.putShort(1); h.putInt(RATE); h.putInt(RATE * 2); h.putShort(2); h.putShort(16)
        h.put("data".toByteArray()); h.putInt(s.size * 2)
        FileOutputStream(f).use { it.write(h.array()); it.write(data.array()) }
    }
}
