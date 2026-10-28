// 扩散模型 · Diffusion Model (Ho, Jain & Abbeel 2020 — DDPM; presented the way the AI image/video generation
// explainer on 3Blue1Brown's channel does it: 2-D point clouds, a learned vector field, time as an axis).
//
// LESSON: generation = learning to UNDO noise. A "time loaf" of eight glass slices runs from the data (t = 0,
// front left) to pure Gaussian noise (t = T, back right): each slice is the forward marginal q(x_t), so the
// spiral visibly melts into noise along the TIME axis. One bright LIVE slice rides that axis with 300 particles:
//   * forward (scrub the time handle): every particle random-walks (the real Markov chain, beta_t kicks);
//   * reverse (pinch the noise slice): 300 fresh noise points are denoised by the trained network, step by
//     step (DDPM or DDIM, 2 … 200 steps); the arrows on the live slice are the network's learned field
//     (-eps_hat ∝ score) at that noise level — at high noise they point at the middle, near t = 0 at the shape.
// Three marked particles can be followed: their paths through time are drawn as threads through the loaf
// (jittery for the forward walk and DDPM, smooth for DDIM; with few steps you see the big straight jumps).
// A live quality meter (min of precision and coverage, core/diffusion/Quality) scores whatever is on the slice.
//
// Honest scope (said in the readout): a mini model — 2-D points, one MLP denoiser (3 layers x 64), trained by
// tools/train_diffusion.py on synthetic clouds; inference is core/diffusion (== PyTorch to 1e-6, tested).
//
// Device budget: particles / arrows / thread segments are pooled nodes moved by transform; the network runs
// at most ~1.5 ms per frame (a 200-step run = 60 000 calls is spread over frames — the playback waits for it);
// the eight slice textures are baked once per dataset (128 px, cached).

package com.bytedance.ailab.demos.diffusion

import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.width
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.bytedance.ailab.core.P3
import com.bytedance.ailab.core.diffusion.DiffusionModel
import com.bytedance.ailab.core.diffusion.ForwardChain
import com.bytedance.ailab.core.diffusion.Quality
import com.bytedance.ailab.core.diffusion.ReverseRun
import com.bytedance.ailab.core.diffusion.Rng
import com.bytedance.ailab.lab.Bi
import com.bytedance.ailab.lab.Challenge
import com.bytedance.ailab.lab.LabAssets
import com.bytedance.ailab.lab.Lesson
import com.bytedance.ailab.lab.LessonStep
import com.bytedance.ailab.lab.Q
import com.bytedance.ailab.lab.SpecDemo
import com.bytedance.ailab.lab.TapTarget
import com.bytedance.ailab.lab.spec.Blend
import com.bytedance.ailab.lab.spec.LabelSpec
import com.bytedance.ailab.lab.spec.Mat
import com.bytedance.ailab.lab.spec.Node
import com.bytedance.ailab.lab.tr
import com.bytedance.ailab.lab.ui.Button
import com.bytedance.ailab.lab.ui.FieldRef
import com.bytedance.ailab.lab.ui.LabPanel
import com.bytedance.ailab.lab.ui.LabRow
import com.bytedance.ailab.lab.ui.LabSwitch
import com.bytedance.ailab.lab.ui.SegmentControl
import com.bytedance.ailab.lab.ui.SegmentItem
import com.bytedance.ailab.render.Image
import com.bytedance.ailab.render.MeshData
import com.bytedance.ailab.render.MeshGen
import com.bytedance.ailab.render.OptGeo
import com.bytedance.ailab.render.ProcTex
import com.bytedance.ailab.render.Rgb
import com.pico.spatial.ui.design.Text
import java.io.File
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.sqrt
import kotlin.math.tanh

object DiffusionDemo : SpecDemo() {
    override val id = "diffusion"
    override val title = "扩散模型"  // i18n-ok (zh half of DemoRegistry.titleOf; English in TITLE_EN)
    override val credit = "Ho, Jain & Abbeel · 2020"

    private val model by lazy { DiffusionModel(LabAssets.bytes("diffusion/model.bin")) }
    private val T get() = model.T

    // ---------------------------------------------------------------- layout (metres, volume space)
    private const val N = 300                       // live particles (= points generated per run)
    private const val NSLICE = 8                     // static slices t = k T / 7
    private const val HS = 0.20f                     // slice half size
    private const val RANGE = 3.2f                   // data units at the slice edge (clouds have std 1)
    private const val UNIT = HS / RANGE              // metres per data unit
    private val P0 = P3(-0.50f, 0.055f, 0.32f)       // centre of the t = 0 slice (front left)
    private val P1 = P3(0.46f, 0.175f, -0.46f)       // centre of the t = T slice (back right, a little higher)
    /** Every slice is turned a little towards the time axis, so the stack reads as one loaf. */
    private val YAW = Q.axisAngle(0f, 1f, 0f, -0.38f)
    private const val RAIL_DY = -HS - 0.045f         // time rail below the slices
    private const val G = 10                         // arrow grid G x G
    private const val ARROW_MAX = 0.032f
    private const val SEGS = 32                      // thread segments per followed particle
    private const val TEX = 128                      // slice texture size
    private const val DOT = 0.016f                   // particle glow size
    private const val DOT_TRACER = 0.026f
    private const val COMPUTE_NS = 1_500_000L        // network time allowed per frame

    val STEP_OPTIONS = intArrayOf(2, 5, 10, 50, 200)
    private val SWEEP_S = floatArrayOf(18f, 9f, 4.5f)  // seconds for t: T -> 0 (slow / medium / fast)
    private val SPEED_LABELS get() = listOf(tr("慢", "Slow"), tr("中", "Medium"), tr("快", "Fast"))
    private val DATA_ZH = listOf("螺旋", "双月", "光 字")   // i18n-ok (zh halves, English in DATA_EN)
    private val DATA_EN = listOf("spiral", "two moons", "glyph")
    private fun dataName(d: Int) = tr(DATA_ZH[d], DATA_EN[d])
    private val TRACER_NAMES = listOf("DotA", "DotB", "DotC")
    private val TILE_NAMES = listOf("Tile0", "Tile1", "Tile2")

    // ---------------------------------------------------------------- settings (panel binds these)
    var datasetIdx = 0
    var stepsIdx = 4
    var ddim = false
    var speedIdx = 1
    var showArrows = true
    var showSlices = true
    val follow = BooleanArray(3)

    // ---------------------------------------------------------------- state
    private enum class Mode { FORWARD, GENERATING, DONE }
    private var mode = Mode.FORWARD
    /** A reverse run is being computed / played back. */
    val generating: Boolean get() = mode == Mode.GENERATING
    private var tShow = 0f                          // displayed diffusion time (real, 0..T)
    private var tTarget = -1f                       // >= 0: animate tShow towards it (forward / replay)
    private var paused = false
    private var dragging = false
    private var chain: ForwardChain? = null
    private var chainHue = FloatArray(N)
    private var run: ReverseRun? = null
    private var runSeed = 0L
    /** Generations started since the demo was built (lesson baselines). */
    var genCount = 0; private set
    data class GenResult(val dataset: Int, val steps: Int, val ddim: Boolean, val q: Float)
    var lastDone: GenResult? = null; private set
    private var forwardPeak = 0f
    private var quality = Quality.Score(1f, 1f)
    private var qualityT = 0f
    private val qualities = arrayOfNulls<Quality>(3)
    private val cur = FloatArray(N * 2)              // displayed particle positions (data units)
    private var arrowT = -1
    private var arrowClock = 0f
    private var arrowDataset = -1
    private val arrowXY = FloatArray(G * G * 2)
    private var hueGrid = IntArray(0)
    private var hueGridFor = -1
    private var pathsDirty = true
    private var pathR = 0.0032f
    private var lastMsg: String? = null

    // ---------------------------------------------------------------- nodes
    private lateinit var live: Node
    private lateinit var liveBack: Node
    private lateinit var dots: Array<Node>
    private lateinit var arrows: Array<Node>
    private lateinit var slices: Array<Node>
    private lateinit var sliceFrames: Array<Node>
    private lateinit var ticks: Array<Node>
    private lateinit var handle: Node
    private lateinit var paths: Array<Array<Node>>
    private lateinit var tiles: Array<Node>
    private lateinit var tileFrames: Array<Node>
    private lateinit var gaugeFill: Node
    private val dotBand = IntArray(N) { -1 }
    private val arrowBand = IntArray(G * G) { -1 }
    private val arrowOn = BooleanArray(G * G)

    // ---------------------------------------------------------------- materials (sRGB)
    private val HUE_STOPS = arrayOf(Rgb(0.15f, 0.95f, 0.95f), Rgb(0.30f, 0.62f, 1.00f), Rgb(0.66f, 0.44f, 1.00f), Rgb(1.00f, 0.38f, 0.78f))
    private fun hueColor(h: Float): Rgb {
        val x = h.coerceIn(0f, 1f) * (HUE_STOPS.size - 1); val k = minOf(HUE_STOPS.size - 2, x.toInt())
        return ProcTex.okLerp(HUE_STOPS[k], HUE_STOPS[k + 1], x - k)
    }
    private val NOISE_C = Rgb(0.78f, 0.86f, 1.00f)
    private val TRACER_C = arrayOf(Rgb(1.00f, 0.80f, 0.22f), Rgb(0.55f, 1.00f, 0.35f), Rgb(1.00f, 0.50f, 0.28f))
    private const val BANDS = 16
    private const val LEVELS = 3

    private val glowTex by lazy {
        OptGeo.lumTex(32, 32) { u, v ->
            val r2 = ((u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f)) * 4f     // 0 centre .. 1 edge
            (exp(-r2 * 9f) + 0.22f * exp(-r2 * 2.6f)).coerceAtMost(1f) * (1f - r2).coerceIn(0f, 1f)
        }
    }
    /** Particle materials: [level][band]; level LEVELS = full data colour (forward particles). */
    private val dotMats by lazy {
        Array(LEVELS + 1) { lv -> Array(BANDS) { b ->
            val c = ProcTex.okLerp(NOISE_C, hueColor((b + 0.5f) / BANDS), (lv.toFloat() / LEVELS))
            Mat.Unlit(c, tex = glowTex, blend = Blend.ADD)
        } }
    }
    private val noiseDot by lazy { Mat.Unlit(dim(NOISE_C, 0.85f), tex = glowTex, blend = Blend.ADD) }
    private val tracerMats by lazy { Array(3) { Mat.Unlit(TRACER_C[it], tex = glowTex, blend = Blend.ADD) } }
    private val ARROW_C = arrayOf(Rgb(0.55f, 0.48f, 0.20f), Rgb(0.78f, 0.68f, 0.28f), Rgb(1.00f, 0.86f, 0.40f), Rgb(1.00f, 0.96f, 0.70f))
    private val arrowMats by lazy { Array(4) { Mat.Unlit(ARROW_C[it], blend = Blend.ADD) } }
    private val pathMats by lazy { Array(3) { Mat.Unlit(dim(TRACER_C[it], 0.9f), blend = Blend.ADD) } }
    private val frameLive = Mat.Unlit(Rgb(0.55f, 0.88f, 1.00f), blend = Blend.ADD)
    private val frameDim = Mat.Unlit(Rgb(0.22f, 0.34f, 0.46f), blend = Blend.ADD)
    private val frameNoise = Mat.Unlit(Rgb(0.45f, 0.55f, 0.75f), blend = Blend.ADD)
    private val axisMat = Mat.Unlit(Rgb(0.16f, 0.24f, 0.32f), blend = Blend.ADD)
    private val railMat = Mat.Unlit(Rgb(0.35f, 0.62f, 0.85f), blend = Blend.ADD)
    private val tickMat = Mat.Unlit(Rgb(0.30f, 0.48f, 0.66f), blend = Blend.ADD)
    private val handleMat = Mat.Pbr(Rgb(0.85f, 0.95f, 1.0f), metallic = 0.2f, roughness = 0.25f, emissive = Rgb(0.45f, 0.75f, 1.0f))
    private val backMat = Mat.Unlit(Rgb(0.02f, 0.025f, 0.035f), alpha = 0.55f, blend = Blend.TRANSPARENT)
    private val alu = Mat.Pbr(Rgb(0.80f, 0.82f, 0.85f), metallic = 1f, roughness = 0.32f)
    private val darkSteel = Mat.Pbr(Rgb(0.20f, 0.21f, 0.23f), metallic = 0.8f, roughness = 0.45f)
    private val slabTex by lazy {
        ProcTex.bakePlanet(256, 128, Rgb(0.07f, 0.075f, 0.085f), Rgb(0.12f, 0.125f, 0.14f), Rgb(0.20f, 0.205f, 0.22f), 0f, 77)
    }
    private val slab get() = Mat.Pbr(Rgb(1f, 1f, 1f), metallic = 0f, roughness = 0.55f, tex = slabTex)
    private val gaugeTrack = Mat.Pbr(Rgb(0.06f, 0.07f, 0.09f), metallic = 0f, roughness = 0.1f, alpha = 0.85f)

    // ---------------------------------------------------------------- meshes
    private val quad by lazy { OptGeo.quad(1f, 1f) }
    private val sphere by lazy { MeshGen.sphere(1f, 16, 32, MeshGen.SphereUv.EQUIRECT) }
    private val unitCyl by lazy { MeshGen.cylinder(1f, 1f, 6) }
    private val unitBox by lazy { MeshGen.box(1f, 1f, 1f) }
    /** Square outline, unit half-size, line width [w] (unit units), in the XY plane. */
    private fun frameMesh(w: Float): MeshData {
        val pos = ArrayList<Float>(); val idx = ArrayList<Int>()
        fun rect(x0: Float, y0: Float, x1: Float, y1: Float) {
            val b = pos.size / 3
            pos += listOf(x0, y1, 0f, x1, y1, 0f, x0, y0, 0f, x1, y0, 0f)
            idx += listOf(b, b + 2, b + 1, b + 1, b + 2, b + 3)
        }
        rect(-1f, 1f - w, 1f, 1f); rect(-1f, -1f, 1f, -1f + w); rect(-1f, -1f + w, -1f + w, 1f - w); rect(1f - w, -1f + w, 1f, 1f - w)
        val n = pos.size / 3
        return MeshData(pos.toFloatArray(), FloatArray(n * 3) { if (it % 3 == 2) 1f else 0f }, FloatArray(n * 2), idx.toIntArray())
    }
    /** Flat arrow along +X, length 1 (shaft to 0.68, head to 1), in the XY plane. */
    private val arrowMesh by lazy {
        val sw = 0.075f; val hw = 0.24f; val hx = 0.62f
        val p = floatArrayOf(0f, sw, 0f, hx, sw, 0f, 0f, -sw, 0f, hx, -sw, 0f, hx, hw, 0f, 1f, 0f, 0f, hx, -hw, 0f)
        MeshData(p, FloatArray(21) { if (it % 3 == 2) 1f else 0f }, FloatArray(14), intArrayOf(0, 2, 1, 1, 2, 3, 4, 6, 5))
    }

    // ---------------------------------------------------------------- slice textures (per dataset, cached)
    private val sliceTexCache = HashMap<Int, Array<Image>>()
    private val tileTex = arrayOfNulls<Image>(3)
    private fun sliceT(k: Int): Int = Math.round(k * T / (NSLICE - 1f))

    /** Forward marginal q(x_t) of dataset d drawn as glowing dust: every reference point, noised once. */
    private fun bakeCloud(d: Int, t: Int, size: Int, gain: Float, seed: Long): Image {
        val acc = FloatArray(size * size * 3)
        val ref = model.ref[d]; val hue = model.hue[d]
        val sa = sqrt(model.abar[t]); val s1 = model.sigma(t)
        val rng = Rng(seed)
        val k = floatArrayOf(0.30f, 1f, 0.30f)
        for (i in 0 until model.nRef) {
            val x = sa * ref[2 * i] + s1 * rng.gauss().toFloat(); val y = sa * ref[2 * i + 1] + s1 * rng.gauss().toFloat()
            val px = ((x / RANGE) * 0.5f + 0.5f) * size; val py = ((-y / RANGE) * 0.5f + 0.5f) * size
            val ix = floor(px).toInt(); val iy = floor(py).toInt()
            if (ix < 1 || iy < 1 || ix >= size - 1 || iy >= size - 1) continue
            val c = hueColor(hue[i]); val cr = ProcTex.toLinear(c.r) * gain; val cg = ProcTex.toLinear(c.g) * gain; val cb = ProcTex.toLinear(c.b) * gain
            for (dy in -1..1) for (dx in -1..1) {
                val w = k[dx + 1] * k[dy + 1]; val o = ((iy + dy) * size + ix + dx) * 3
                acc[o] += cr * w; acc[o + 1] += cg * w; acc[o + 2] += cb * w
            }
        }
        val img = Image(size, size)
        for (i in 0 until size * size) {
            // soft knee so dense ridges glow white-ish instead of clipping flat
            fun tone(v: Float) = OptGeo.enc(v / (1f + 0.55f * v))
            img.px[i] = ProcTex.argb(1f, tone(acc[i * 3]), tone(acc[i * 3 + 1]), tone(acc[i * 3 + 2]))
        }
        return img
    }

    private fun sliceTextures(d: Int): Array<Image> = sliceTexCache.getOrPut(d) {
        Array(NSLICE) { k -> bakeCloud(d, sliceT(k), TEX, if (k == 0) 0.10f else 0.32f, 4000L + 31 * d + k) }
    }

    // ---------------------------------------------------------------- build

    override fun buildSpec(cacheDir: File): Node {
        // every setting and cache starts fresh: re-entering the demo gives the same scene
        datasetIdx = 0; stepsIdx = 4; ddim = false; speedIdx = 1; showArrows = true; showSlices = true
        follow.fill(false); follow[0] = true
        mode = Mode.FORWARD; tShow = 0f; tTarget = -1f; paused = false; dragging = false
        run = null; runSeed = 0L; genCount = 0; lastDone = null; forwardPeak = 0f
        quality = Quality.Score(1f, 1f); qualityT = 0f; arrowT = -1; arrowClock = 0f; arrowDataset = -1
        hueGridFor = -1; pathsDirty = true; lastMsg = null
        placedT = -1f; placedRun = null; placedMode = null; placedStamp = -1
        dotBand.fill(-1); arrowBand.fill(-1)

        val root = Node("DiffusionRoot")

        // ---- bench: a dark slab with an aluminium edge, like every other lab bench
        root.add("Plinth", MeshGen.box(1.78f, 0.035f, 1.08f), slab, P3(0f, -0.262f, -0.04f))
        root.add("PlinthBand", MeshGen.box(1.79f, 0.008f, 1.09f), alu, P3(0f, -0.262f, -0.04f))

        // ---- the time loaf: eight static slices (forward marginals), faint frames
        val fDim = frameMesh(0.018f); val fLive = frameMesh(0.03f)
        val tex = sliceTextures(0)
        slices = Array(NSLICE) { k ->
            val name = if (k == NSLICE - 1) "NoiseSlice" else "Slice$k"
            root.add(name, quad, Mat.Unlit(Rgb(1f, 1f, 1f), tex = tex[k], blend = Blend.ADD), along(sliceT(k).toFloat())).also { it.uniformScale(2 * HS); it.rot = YAW }
        }
        slices[NSLICE - 1].tapRadius = 0.12f / (2 * HS)
        sliceFrames = Array(NSLICE) { k ->
            root.add("SliceFrame$k", fDim, if (k == NSLICE - 1) frameNoise else frameDim, along(sliceT(k).toFloat())).also { it.uniformScale(HS); it.rot = YAW }
        }

        // ---- time rail, ticks at the slices, the handle
        val r0 = along(0f) + P3(0f, RAIL_DY, 0f); val r1 = along(T.toFloat()) + P3(0f, RAIL_DY, 0f)
        root.add("Rail", MeshGen.tube(floatArrayOf(r0.x, r0.y, r0.z, r1.x, r1.y, r1.z), 0.0032f, 8, false), railMat)
        ticks = Array(NSLICE) { k ->
            root.add("Tick$k", unitBox, tickMat, along(sliceT(k).toFloat()) + P3(0f, RAIL_DY + 0.012f, 0f)).also { it.scale = P3(0.003f, 0.024f, 0.003f) }
        }
        handle = root.add("TimeHandle", sphere, handleMat, r0).also { it.uniformScale(0.016f); it.tapRadius = 0.052f / 0.016f }

        // ---- the live slice: dark glass back, bright frame, axes, particles, arrows
        live = root.add(Node("LiveSlice")).also { it.pos = along(0f); it.rot = YAW }
        liveBack = live.add("LiveBack", quad, backMat, P3(0f, 0f, -0.006f)).also { it.uniformScale(2 * HS) }
        live.add("LiveFrame", fLive, frameLive).also { it.uniformScale(HS * 1.01f) }
        live.add("AxisX", unitBox, axisMat, P3(0f, 0f, 0.001f)).also { it.scale = P3(2 * HS, 0.0016f, 0.0005f) }
        live.add("AxisY", unitBox, axisMat, P3(0f, 0f, 0.001f)).also { it.scale = P3(0.0016f, 2 * HS, 0.0005f) }
        arrows = Array(G * G) { i ->
            val gx = i % G; val gy = i / G
            val x = ((gx + 0.5f) / G * 2f - 1f) * 2.9f; val y = ((gy + 0.5f) / G * 2f - 1f) * 2.9f
            arrowXY[2 * i] = x; arrowXY[2 * i + 1] = y
            live.add("Arrow$i", arrowMesh, arrowMats[0], P3(x * UNIT, y * UNIT, 0.004f)).also { it.uniformScale(0.001f) }
        }
        dots = Array(N) { i ->
            val name = if (i < 3) TRACER_NAMES[i] else "Dot$i"
            live.add(name, quad, if (i < 3) tracerMats[i] else noiseDot, P3(0f, 0f, 0.003f + if (i < 3) 0.002f else 0f)).also {
                it.uniformScale(if (i < 3) DOT_TRACER else DOT)
                if (i < 3) it.tapRadius = 0.05f / DOT_TRACER
            }
        }

        // ---- threads: the followed particles' paths through time (pooled segments, loaf space)
        paths = Array(3) { j -> Array(SEGS) { s -> root.add("Path${j}_$s", unitCyl, pathMats[j]).also { it.visible = false } } }
        pathR = 0.0032f

        // ---- dataset shelf (front right): three tiles, the active one framed
        val shelfC = P3(0.60f, -0.135f, 0.30f)
        root.add("ShelfBase", MeshGen.box(0.40f, 0.014f, 0.11f), darkSteel, shelfC + P3(0f, -0.068f, 0f))
        root.add("ShelfLip", MeshGen.box(0.41f, 0.005f, 0.115f), alu, shelfC + P3(0f, -0.060f, 0f))
        tiles = Array(3) { k ->
            tileTex[k] = tileTex[k] ?: bakeCloud(k, 0, 96, 0.18f, 77L + k)
            root.add("Tile$k", quad, Mat.Unlit(Rgb(1f, 1f, 1f), tex = tileTex[k], blend = Blend.ADD), shelfC + P3((k - 1) * 0.13f, 0f, 0f)).also {
                it.uniformScale(0.105f); it.rot = Q.axisAngle(1f, 0f, 0f, -0.25f); it.tapRadius = 0.055f / 0.105f
            }
        }
        tileFrames = Array(3) { k ->
            root.add("TileFrame$k", frameMesh(0.05f), frameDim, shelfC + P3((k - 1) * 0.13f, 0f, -0.002f)).also {
                it.uniformScale(0.0545f); it.rot = Q.axisAngle(1f, 0f, 0f, -0.25f)
            }
        }
        for (k in 0 until 3) root.add("TileBack$k", quad, backMat, shelfC + P3((k - 1) * 0.13f, 0f, -0.004f)).also {
            it.uniformScale(0.11f); it.rot = Q.axisAngle(1f, 0f, 0f, -0.25f)
        }

        // ---- quality gauge (left of the data slice): track, fill, 80 % target mark
        val gx = P0.x - HS - 0.035f; val gz = P0.z - 0.04f
        root.add("GaugeTrack", unitBox, gaugeTrack, P3(gx, P0.y, gz - 0.006f)).also { it.scale = P3(0.026f, 2 * HS + 0.012f, 0.008f) }
        gaugeFill = root.add("GaugeFill", unitBox, Mat.Unlit(Rgb(0.3f, 1f, 0.5f), blend = Blend.ADD), P3(gx, P0.y, gz)).also { it.scale = P3(0.016f, 0.01f, 0.004f) }
        root.add("GaugeTarget", unitBox, Mat.Unlit(Rgb(1f, 1f, 1f), blend = Blend.ADD), P3(gx, P0.y - HS + 0.8f * 2 * HS, gz + 0.003f)).also { it.scale = P3(0.036f, 0.0025f, 0.002f) }

        applyDataset(0)
        return root
    }

    /** Volume position of the slice centre at diffusion time t. */
    private fun along(t: Float): P3 { val f = t / T; return P3(P0.x + (P1.x - P0.x) * f, P0.y + (P1.y - P0.y) * f, P0.z + (P1.z - P0.z) * f) }

    // ---------------------------------------------------------------- dataset / runs

    private fun applyDataset(d: Int) {
        datasetIdx = d
        val tex = sliceTextures(d)
        for (k in 0 until NSLICE) slices[k].setMat(Mat.Unlit(Rgb(1f, 1f, 1f), tex = tex[k], blend = Blend.ADD))
        for (k in 0 until 3) tileFrames[k].setMat(if (k == d) frameLive else frameDim)
        // the forward chain starts from the first N reference samples; three spread-out ones are the tracers
        val x0 = model.ref[d].copyOf(N * 2); val hue = model.hue[d].copyOf(N)
        for ((j, want) in floatArrayOf(0.12f, 0.50f, 0.88f).withIndex()) {
            var best = j; for (i in j until N) if (abs(hue[i] - want) < abs(hue[best] - want)) best = i
            swap(x0, hue, j, best)
        }
        chain = ForwardChain(model, x0, 31L + d)
        chainHue = hue
        run = null; mode = Mode.FORWARD; tShow = 0f; tTarget = -1f; paused = false
        dotBand.fill(-1); pathsDirty = true; arrowT = -1; forwardPeak = 0f
    }

    private fun swap(x: FloatArray, h: FloatArray, a: Int, b: Int) {
        if (a == b) return
        val tx = x[2 * a]; val ty = x[2 * a + 1]; x[2 * a] = x[2 * b]; x[2 * a + 1] = x[2 * b + 1]; x[2 * b] = tx; x[2 * b + 1] = ty
        val th = h[a]; h[a] = h[b]; h[b] = th
    }

    val steps get() = STEP_OPTIONS[stepsIdx]

    /** Pinch the noise slice: N fresh Gaussian points at t = T, denoised by the network on the chosen grid. */
    fun generate() {
        runSeed++
        run = ReverseRun(model, datasetIdx, steps, ddim, N, 9000L + runSeed)
        genCount++
        mode = Mode.GENERATING; tShow = T.toFloat(); tTarget = -1f; paused = false
        dotBand.fill(-1); pathsDirty = true; arrowT = -1
    }

    fun setSteps(s: Int) { stepsIdx = STEP_OPTIONS.indexOf(s).coerceAtLeast(0) }
    fun selectDataset(d: Int) { if (d != datasetIdx || mode != Mode.FORWARD) applyDataset(d) }
    fun resetToData() { applyDataset(datasetIdx) }

    // ---------------------------------------------------------------- per frame

    override fun update(dt: Float) {
        val rate = T / SWEEP_S[speedIdx]                 // diffusion-time units per second
        val frameStart = System.nanoTime()
        updateArrows(dt)                                 // network calls share one per-frame budget
        when (mode) {
            Mode.FORWARD -> {
                if (tTarget >= 0f && !dragging) tShow = approach(tShow, tTarget, rate * dt).also { if (it == tTarget) tTarget = -1f }
                forwardPeak = maxOf(forwardPeak, tShow)
            }
            Mode.GENERATING -> {
                val r = run!!
                while (!r.finished && System.nanoTime() - frameStart < COMPUTE_NS) r.advance(DiffusionModel.CHUNK)
                val front = r.tau[r.stepsDone].toFloat()     // lowest t whose positions are known
                if (!paused && !dragging) tShow = maxOf(front, tShow - rate * dt)
                tShow = maxOf(tShow, front)
                if (r.finished && tShow <= 0f) {
                    mode = Mode.DONE; tShow = 0f
                    val q = qualityOf(r.result())
                    lastDone = GenResult(r.dataset, r.steps, r.ddim, q.q)
                    pathsDirty = true
                }
            }
            Mode.DONE -> if (tTarget >= 0f && !dragging) tShow = approach(tShow, tTarget, rate * dt).also { if (it == tTarget) tTarget = -1f }
        }
        tShow = tShow.coerceIn(0f, T.toFloat())

        placeParticles()
        live.pos = along(tShow)
        handle.pos = along(tShow) + P3(0f, RAIL_DY, 0f)
        for (k in 0 until NSLICE) { slices[k].visible = showSlices; sliceFrames[k].visible = showSlices }
        slices[NSLICE - 1].visible = true; sliceFrames[NSLICE - 1].visible = true   // the noise slice stays: it is the "generate" button
        if (dt == 0f) { arrowT = -1; updateArrows(0f) }
        updatePaths()
        qualityT -= dt
        if (qualityT <= 0f || dt == 0f) { qualityT = 0.15f; quality = qualityOf(cur) }
        val gh = (2 * HS) * quality.q.coerceIn(0.01f, 1f)
        gaugeFill.scale = P3(0.016f, gh, 0.004f)
        gaugeFill.pos = P3(gaugeFill.pos.x, P0.y - HS + gh / 2, gaugeFill.pos.z)
        gaugeFill.setMat(Mat.Unlit(gaugeColour(quality.q), blend = Blend.ADD))
        updateReadout()
        updateLabels()
    }

    private fun approach(v: Float, target: Float, step: Float) = if (abs(target - v) <= step) target else v + Math.signum(target - v) * step

    private fun qualityOf(pts: FloatArray): Quality.Score {
        val q = qualities[datasetIdx] ?: Quality(model.ref[datasetIdx], model.nCov).also { qualities[datasetIdx] = it }
        return q.score(pts, N)
    }

    private fun gaugeColour(q: Float): Rgb = when {
        q >= 0.8f -> Rgb(0.35f, 1.00f, 0.55f)
        q >= 0.5f -> ProcTex.okLerp(Rgb(1.0f, 0.55f, 0.20f), Rgb(1.0f, 0.90f, 0.30f), ((q - 0.5f) / 0.3f).toFloat())
        else -> Rgb(1.0f, 0.35f, 0.25f)
    }

    private var placedT = -1f
    private var placedRun: Any? = null
    private var placedMode: Mode? = null

    /** Positions (data units) of every particle at the displayed time, and their glows. */
    private fun placeParticles() {
        val runRef: Any? = if (mode == Mode.FORWARD) chain else run
        val stamp = (run?.stepsDone ?: 0)
        if (tShow == placedT && placedRun === runRef && placedMode == mode && stamp == placedStamp) return
        placedT = tShow; placedRun = runRef; placedMode = mode; placedStamp = stamp
        pathsDirty = true
        val sig = model.sigma(tShow.toInt().coerceIn(0, T))
        when (mode) {
            Mode.FORWARD -> {
                val c = chain!!
                val t0 = floor(tShow).toInt().coerceIn(0, T); val t1 = minOf(T, t0 + 1); val f = tShow - t0
                for (i in 0 until N) {
                    cur[2 * i] = c.x(t0, i) + (c.x(t1, i) - c.x(t0, i)) * f
                    cur[2 * i + 1] = c.y(t0, i) + (c.y(t1, i) - c.y(t0, i)) * f
                }
            }
            else -> {
                val r = run!!
                val (k, f) = gridPos(r, tShow)
                val k1 = minOf(k + 1, r.stepsDone)
                for (i in 0 until N) {
                    cur[2 * i] = r.x(k, i) + (r.x(k1, i) - r.x(k, i)) * f
                    cur[2 * i + 1] = r.y(k, i) + (r.y(k1, i) - r.y(k, i)) * f
                }
            }
        }
        // colour: forward particles keep the colour of where they started; generated ones are noise-white
        // and take the colour of the part of the shape they are landing on as the noise level drops
        val level = if (mode == Mode.FORWARD) LEVELS else ((0.62f - sig) / 0.42f * LEVELS).toInt().coerceIn(0, LEVELS)
        for (i in 0 until N) {
            val x = cur[2 * i]; val y = cur[2 * i + 1]
            val n = dots[i]
            val ex = x.coerceIn(-RANGE * 1.06f, RANGE * 1.06f); val ey = y.coerceIn(-RANGE * 1.06f, RANGE * 1.06f)
            n.pos = P3(ex * UNIT, ey * UNIT, n.pos.z)
            if (i < 3) continue
            val band = if (level == 0) -2 else level * BANDS + bandOf(if (mode == Mode.FORWARD) chainHue[i] else hueAt(x, y))
            if (band != dotBand[i]) {
                dotBand[i] = band
                n.setMat(if (band == -2) noiseDot else dotMats[band / BANDS][band % BANDS])
            }
        }
    }

    private var placedStamp = -1

    private fun bandOf(h: Float) = (h * BANDS).toInt().coerceIn(0, BANDS - 1)

    /** Grid index k and fraction f with tau[k] >= t >= tau[k+1] (clamped to the computed part). */
    private fun gridPos(r: ReverseRun, t: Float): Pair<Int, Float> {
        val kr = r.steps * (T - t) / T
        var k = floor(kr).toInt().coerceIn(0, r.steps)
        var f = kr - k
        if (k >= r.stepsDone) { k = r.stepsDone; f = 0f }
        return k to f
    }

    /** Hue of the nearest reference point (memoised on a 64 x 64 grid over the slice). */
    private fun hueAt(x: Float, y: Float): Float {
        val d = datasetIdx
        if (hueGridFor != d) { hueGrid = IntArray(64 * 64) { -1 }; hueGridFor = d }
        val cx = (((x / RANGE) * 0.5f + 0.5f) * 64).toInt().coerceIn(0, 63)
        val cy = (((y / RANGE) * 0.5f + 0.5f) * 64).toInt().coerceIn(0, 63)
        val c = cy * 64 + cx
        if (hueGrid[c] < 0) {
            val px = ((cx + 0.5f) / 64 * 2 - 1) * RANGE; val py = ((cy + 0.5f) / 64 * 2 - 1) * RANGE
            val ref = model.ref[d]; var best = 0; var bd = Float.MAX_VALUE
            for (j in 0 until model.nRef) { val ex = ref[2 * j] - px; val ey = ref[2 * j + 1] - py; val dd = ex * ex + ey * ey; if (dd < bd) { bd = dd; best = j } }
            hueGrid[c] = best
        }
        return model.hue[d][hueGrid[c]]
    }

    /** The learned field at the displayed noise level: arrows along -eps_hat (the score direction). */
    private fun updateArrows(dt: Float) {
        for (i in arrows.indices) arrows[i].visible = showArrows && arrowOn[i]
        if (!showArrows) return
        arrowClock -= dt
        val te = Math.round(tShow).coerceIn(4, T)
        if ((te == arrowT && arrowDataset == datasetIdx) || (arrowClock > 0f && dt > 0f)) return
        arrowClock = 0.15f; arrowT = te; arrowDataset = datasetIdx
        val e = model.epsAll(datasetIdx, te, arrowXY)
        for (i in 0 until G * G) {
            val ex = -e[2 * i]; val ey = -e[2 * i + 1]
            val m = sqrt(ex * ex + ey * ey)
            val len = ARROW_MAX * tanh(m / 1.4f)
            val a = arrows[i]
            arrowOn[i] = len >= 0.004f
            a.visible = arrowOn[i]
            if (!arrowOn[i]) continue
            a.rot = Q.axisAngle(0f, 0f, 1f, atan2(ey, ex))
            a.scale = P3(len, len, 1f)
            val b = when { m < 0.5f -> 0; m < 1f -> 1; m < 2f -> 2; else -> 3 }
            if (b != arrowBand[i]) { arrowBand[i] = b; a.setMat(arrowMats[b]) }
        }
    }

    /** Threads: each followed particle's path from where it started to the displayed time. */
    private fun updatePaths() {
        if (!pathsDirty) return
        pathsDirty = false
        for (j in 0 until 3) {
            val segs = paths[j]
            if (!follow[j]) { for (s in segs) s.visible = false; continue }
            // vertices (t, x, y) from the start of the process to now
            val vt = ArrayList<Float>(SEGS + 2); val vx = ArrayList<Float>(SEGS + 2); val vy = ArrayList<Float>(SEGS + 2)
            when (mode) {
                Mode.FORWARD -> {
                    val c = chain!!; val tEnd = tShow
                    var s = 0
                    while (s <= SEGS) {
                        val t = T * s / SEGS.toFloat(); if (t > tEnd) break
                        vt += t; vx += c.x(t.toInt(), j); vy += c.y(t.toInt(), j); s++
                    }
                    vt += tShow; vx += cur[2 * j]; vy += cur[2 * j + 1]
                }
                else -> {
                    val r = run!!
                    val (k, _) = gridPos(r, tShow)
                    val stride = maxOf(1, (r.steps + SEGS - 1) / SEGS)
                    var kk = 0
                    while (kk <= k) { vt += r.tau[kk].toFloat(); vx += r.x(kk, j); vy += r.y(kk, j); kk += stride }
                    vt += tShow; vx += cur[2 * j]; vy += cur[2 * j + 1]
                }
            }
            var used = 0
            for (s in 0 until vt.size - 1) {
                if (used >= SEGS) break
                val a = world(vt[s], vx[s], vy[s]); val b = world(vt[s + 1], vx[s + 1], vy[s + 1])
                val d = b - a; val l = d.length
                if (l < 1e-4f) continue
                val n = segs[used++]
                n.visible = true
                n.pos = P3((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2)
                n.rot = Q.fromTo(0f, 1f, 0f, d.x, d.y, d.z)
                n.scale = P3(pathR, l, pathR)
            }
            for (s in used until SEGS) segs[s].visible = false
        }
    }

    /** Volume position of data point (x, y) on the slice at time t. */
    private fun world(t: Float, x: Float, y: Float): P3 {
        val c = along(t)
        val o = YAW.rotate(floatArrayOf(x.coerceIn(-RANGE, RANGE) * UNIT, y.coerceIn(-RANGE, RANGE) * UNIT, 0.004f))
        return P3(c.x + o[0], c.y + o[1], c.z + o[2])
    }

    // ---------------------------------------------------------------- readout + labels

    private fun samplerName() = if (ddim) "DDIM" else "DDPM"

    private fun updateReadout() {
        val ti = Math.round(tShow).coerceIn(0, T)
        val ab = model.abar[ti]; val sg = model.sigma(ti)
        val phase = when (mode) {
            Mode.FORWARD -> tr("前向：数据 → 噪声（随机游走）", "forward: data → noise (random walk)")
            Mode.GENERATING -> if (paused) tr("生成已暂停", "generation paused") else tr("反向生成中：噪声 → 数据", "generating: noise → data")
            Mode.DONE -> tr("生成完成 —— 拖动滑块回看", "generated — drag the handle to replay")
        }
        val r = run
        val calls = (r?.evals ?: 0L)
        val runSteps = r?.steps ?: steps; val runSampler = if (r?.ddim ?: ddim) "DDIM" else "DDPM"
        val q = (quality.q * 100).toInt()
        val l2 = if (mode == Mode.FORWARD)
            tr("迷你扩散模型：2-D 点云，MLP 去噪器（3×64）· 当前设置 %s %d 步 · 质量 %d %%", "Mini diffusion model: 2-D points, MLP denoiser (3×64) · set to %s %d steps · quality %d %%")
                .format(samplerName(), steps, q)
        else tr("迷你扩散模型：2-D 点云，MLP 去噪器（3×64）· %s %d 步 · 网络调用 %,d 次（%.0f ms）· 质量 %d %%", "Mini diffusion model: 2-D points, MLP denoiser (3×64) · %s %d steps · %,d network calls (%.0f ms) · quality %d %%")
            .format(runSampler, runSteps, calls, (r?.computeNs ?: 0L) / 1e6, q)
        val l3 = when {
            mode == Mode.FORWARD && tShow < 2f -> tr("生成 = 学会撤销噪声。先拖动时间滑块，看数据怎样一步步融化成噪声。", "Generation = learning to undo noise. Drag the time handle and watch the data melt into noise.")
            mode == Mode.FORWARD -> tr("每一步只加一点高斯噪声；t 时刻的分布正好是 N(√ᾱ_t·x₀, (1−ᾱ_t)·I)。", "Each step adds a little Gaussian noise; at time t the cloud is exactly N(√ᾱ_t·x₀, (1−ᾱ_t)·I).")
            mode == Mode.GENERATING -> tr("箭头 = 网络在这个噪声水平学到的推力（分数场）：噪声大时指向中间，噪声小时指向形状。", "Arrows = the push the network learned at this noise level (score field): to the middle at high noise, to the shape at low noise.")
            r != null && r.ddim && runSteps > 2 -> tr("DDIM 不加随机扰动：同样的网络，轨迹变成平滑曲线（一个常微分方程），少步数也稳。", "DDIM adds no random kicks: same network, the paths become smooth curves (an ODE) that tolerate few steps.")
            runSteps <= 2 -> tr("只有 %d 步：每步跨得太远，网络只能猜「平均位置」，点落在形状之间。".format(runSteps), "Only %d steps: each jump is too big, the network can only guess the average, points land between the arms.".format(runSteps))
            else -> tr("从同一种噪声出发，换一组随机数就是一批新的点 —— 模型学的是分布，不是这 300 个点。", "Same kind of noise, new random numbers, new points — the model learned the distribution, not these 300 points.")
        }
        readoutText = tr("t = %d / %d   ᾱ_t = %.3f   噪声 σ_t = √(1−ᾱ_t) = %.2f   %s\n", "t = %d / %d   ᾱ_t = %.3f   noise σ_t = √(1−ᾱ_t) = %.2f   %s\n")
            .format(ti, T, ab, sg, phase) + l2 + "\n" + l3
    }

    private fun updateLabels() {
        labels.clear()
        val ti = Math.round(tShow)
        labels += LabelSpec(tr("t = 0 · 数据", "t = 0 · data"), along(0f) + P3(0f, HS + 0.045f, 0f))
        labels += LabelSpec(tr("t = T · 捏这一片开始生成", "t = T · pinch this slice to generate"), along(T.toFloat()) + P3(0f, HS + 0.045f, 0f))
        labels += LabelSpec(tr("当前 t = %d · σ = %.2f", "now t = %d · σ = %.2f").format(ti, model.sigma(ti.coerceIn(0, T))),
            along(tShow) + P3(0f, HS + 0.045f, 0.01f), visible = tShow > 0.12f * T && tShow < 0.88f * T)
        labels += LabelSpec(tr("数据集 · 捏一块换形状", "Dataset · pinch a tile"), P3(0.60f, -0.04f, 0.30f))
        labels += LabelSpec(tr("质量 %d %%", "Quality %d %%").format((quality.q * 100).toInt()), P3(P0.x - HS - 0.035f, P0.y + HS + 0.045f, P0.z - 0.04f))
    }

    // ---------------------------------------------------------------- interaction

    override val tapTargets: List<TapTarget> get() = listOf(
        TapTarget("NoiseSlice", "噪声切片 → 从噪声生成", "Noise slice → generate"),
        TapTarget("TimeHandle", "时间滑块 → 扫过时间（可拖）", "Time handle → sweep t (drag)"),
        TapTarget("DotA", "粒子 A → 跟随", "Dot A → follow") { follow[0] },
        TapTarget("DotB", "粒子 B → 跟随", "Dot B → follow") { follow[1] },
        TapTarget("DotC", "粒子 C → 跟随", "Dot C → follow") { follow[2] },
        TapTarget("Tile0", "螺旋 → 载入", "Spiral → load") { datasetIdx == 0 },
        TapTarget("Tile1", "双月 → 载入", "Moons → load") { datasetIdx == 1 },
        TapTarget("Tile2", "光 字 → 载入", "Glyph → load") { datasetIdx == 2 },
    )

    override fun isTarget(name: String) = name == "NoiseSlice" || name == "TimeHandle" || name in TRACER_NAMES || name in TILE_NAMES
    override fun isDragTarget(name: String) = name == "TimeHandle"

    override fun onTap(name: String) {
        when {
            name == "NoiseSlice" -> {
                generate()
                lastMsg = tr("从纯噪声生成 %d 个点 · %s %d 步", "Generating %d points from pure noise · %s %d steps").format(N, samplerName(), steps)
            }
            name == "TimeHandle" -> when (mode) {
                Mode.GENERATING -> {
                    paused = !paused
                    lastMsg = if (paused) tr("暂停 —— 拖动滑块看已算好的步", "Paused — drag the handle through the computed steps") else tr("继续生成", "Generating again")
                }
                else -> {
                    tTarget = if (tShow < T / 2f) T.toFloat() else 0f
                    lastMsg = when {
                        mode == Mode.FORWARD && tTarget > 0f -> tr("t → T：看数据一步步融化成噪声", "t → T: watch the data melt into noise")
                        mode == Mode.FORWARD -> tr("t → 0：沿同一条随机游走退回数据", "t → 0: back along the same random walks")
                        tTarget > 0f -> tr("倒回 t = T：生成之前的纯噪声", "Rewind to t = T: the pure noise it started from")
                        else -> tr("回放生成：噪声 → 数据", "Replay: noise → data")
                    }
                }
            }
            name in TRACER_NAMES -> {
                val j = TRACER_NAMES.indexOf(name)
                follow[j] = !follow[j]; pathsDirty = true
                val l = "ABC"[j]
                lastMsg = if (follow[j]) tr("跟随粒子 $l：它穿过时间的路径", "Following dot $l: its path through time") else tr("不再跟随粒子 $l", "Stopped following dot $l")
            }
            name.startsWith("Tile") -> {
                val d = name.removePrefix("Tile").toIntOrNull() ?: return
                selectDataset(d)
                lastMsg = tr("数据集：%s —— 同一个网络，只换了条件标签", "Dataset: %s — same network, new condition label").format(dataName(d))
            }
        }
    }

    override fun tapMessage(name: String): String? = lastMsg

    override fun onDragStart(name: String) { dragging = true; tTarget = -1f }
    override fun onDrag(name: String, dx: Float, dy: Float, dz: Float) {
        val ax = P1 - P0; val l = ax.length
        val ds = (dx * ax.x + dy * ax.y + dz * ax.z) / l
        val lo = if (mode == Mode.GENERATING) run!!.tau[run!!.stepsDone].toFloat() else 0f
        // gain 2.4: a ~0.5 m sideways hand move sweeps the whole 1.2 m rail
        tShow = (tShow + ds / l * T * 2.4f).coerceIn(lo, T.toFloat())
    }
    override fun onDragEnd(name: String) { dragging = false }

    // ---------------------------------------------------------------- guided track (导学)

    private var genAtEnter = 0

    override val lesson: Lesson by lazy {
        Lesson(
            listOf(
                LessonStep(
                    say = Bi("扩散模型要「生成」一团新的点。你猜：它从什么开始？", "A diffusion model is about to generate a new point cloud. What does it start from?"),
                    options = listOf(Bi("一张模糊的螺旋草图", "A blurry sketch of the spiral"), Bi("纯随机噪声", "Pure random noise"), Bi("训练集里挑一个点", "A point picked from the training set")),
                    answer = 1,
                ),
                LessonStep(
                    say = Bi("先看反过来的过程：捏时间轴上的滑块（或拖动它），把 t 从 0 推到 T。", "First the opposite process: pinch the handle on the time axis (or drag it) to push t from 0 to T."),
                    target = "TimeHandle",
                    onEnter = { forwardPeak = if (mode == Mode.FORWARD) tShow else 0f },
                    done = { mode == Mode.FORWARD && forwardPeak >= 0.95f * T },
                    autoplay = { if (mode != Mode.FORWARD) resetToData(); onTap("TimeHandle") },
                    insight = Bi("前向过程只是一步步加一点高斯噪声。每片切片就是 t 时刻的分布：到 t = T，螺旋只剩纯噪声。", "The forward process just adds a little Gaussian noise per step. Each slice is the cloud at time t: by t = T only noise is left."),
                ),
                LessonStep(
                    say = Bi("现在捏最远处的噪声切片（t = T）：从纯噪声生成 300 个点，看箭头把它们推回去。", "Now pinch the far noise slice (t = T): 300 points start as pure noise — watch the arrows push them back."),
                    target = "NoiseSlice",
                    onEnter = { genAtEnter = genCount },
                    done = { genCount > genAtEnter && mode == Mode.DONE },
                    autoplay = { onTap("NoiseSlice") },
                    reveal = true,
                    insight = Bi("模型学的是每一步该往哪推 —— 分数场（箭头）。它不记任何一个点，只学会在每个噪声水平把点推向数据。", "The model learned which way to push at every step — the score field (arrows). It memorises no point; it learned to push towards the data at every noise level."),
                ),
                LessonStep(
                    say = Bi("采样步数从 200 减到 10 会怎样？", "What happens if sampling uses 10 steps instead of 200?"),
                    options = listOf(Bi("形状糊掉，点散得到处都是", "The shape blurs, points scatter"), Bi("几乎一样好，还快 20 倍", "Almost as good, and 20× faster"), Bi("完全生成不出来", "Nothing comes out at all")),
                    answer = 1,
                ),
                LessonStep(
                    say = Bi("在面板上把采样步数调到 10，再捏噪声切片生成一次，看质量。", "Set the sampling steps to 10 on the panel, pinch the noise slice again and read the quality."),
                    target = "NoiseSlice",
                    onEnter = { genAtEnter = genCount },
                    done = { genCount > genAtEnter && mode == Mode.DONE && lastDone?.steps == 10 },
                    autoplay = { setSteps(10); generate() },
                    reveal = true,
                    insight = Bi("10 步质量约 98 %，200 步约 100 %，网络调用却少 20 倍。但步子不能太大：只走 2 步时质量跌到约 40 %。", "10 steps score about 98 %, 200 steps about 100 % — with 20× fewer network calls. But not too few: with 2 steps quality drops to about 40 %."),
                ),
            ),
            challenge = Challenge(
                say = Bi("最少几步还能画好螺旋？用不超过 5 步生成螺旋，让质量 ≥ 80 %。", "How few steps still draw the spiral? Use at most 5 steps and reach quality ≥ 80 % on the spiral."),
                progress = {
                    val g = lastDone
                    if (g == null || g.dataset != 0) tr("先选螺旋，选步数，再捏噪声切片", "Pick the spiral, pick the steps, pinch the noise slice")
                    else tr("上一次：螺旋 · %d 步 · 质量 %d %%（目标：≤ 5 步，≥ 80 %%）", "Last run: spiral · %d steps · quality %d %% (goal: ≤ 5 steps, ≥ 80 %%)").format(g.steps, (g.q * 100).toInt())
                },
                done = { lastDone?.let { it.dataset == 0 && it.steps <= 5 && it.q >= 0.8f } == true },
                autoplay = { selectDataset(0); setSteps(5); generate() },
            ),
        )
    }

    // ---------------------------------------------------------------- review states

    override fun reviewVariants(): Map<String, () -> Unit> = mapOf(
        "generating" to { generate(); run!!.runToEnd(); tShow = 0.42f * T; paused = true },
        "done" to { follow[1] = true; follow[2] = true; generate(); run!!.runToEnd(); tShow = 0f },
        "ddim5" to { ddim = true; setSteps(5); follow.fill(true); generate(); run!!.runToEnd(); tShow = 0f },
        "melt" to { ddim = false; setSteps(200); resetToData(); follow.fill(true); tShow = 0.45f * T },
        "glyph" to { selectDataset(2); follow.fill(false); generate(); run!!.runToEnd(); tShow = 0f },
    )

    // ---------------------------------------------------------------- panel

    @Composable
    override fun Panel() {
        var si by FieldRef(::stepsIdx)
        var dd by FieldRef(::ddim)
        var sp by FieldRef(::speedIdx)
        var ar by FieldRef(::showArrows)
        var sl by FieldRef(::showSlices)
        LabPanel(readout) {
            LabRow {
                SegmentControl {
                    STEP_OPTIONS.forEachIndexed { k, s -> SegmentItem(selected = si == k, onClick = { si = k }, title = { Text(tr("$s 步", "$s steps")) }) }
                }
                Spacer(Modifier.width(20.dp))
                SegmentControl {
                    SegmentItem(selected = !dd, onClick = { dd = false }, title = { Text(tr("DDPM 随机", "DDPM")) })
                    SegmentItem(selected = dd, onClick = { dd = true }, title = { Text(tr("DDIM 确定", "DDIM")) })
                }
            }
            LabRow {
                SegmentControl {
                    SPEED_LABELS.forEachIndexed { k, l -> SegmentItem(selected = sp == k, onClick = { sp = k }, title = { Text(l) }) }
                }
                Spacer(Modifier.width(20.dp))
                LabSwitch(tr("分数场箭头", "Score arrows"), ar) { ar = it }
                LabSwitch(tr("时间切片", "Time slices"), sl) { sl = it }
                Button(onClick = { resetToData() }) { Text(tr("重置", "Reset")) }
            }
        }
    }

    private fun dim(c: Rgb, k: Float) = Rgb(c.r * k, c.g * k, c.b * k)
}
