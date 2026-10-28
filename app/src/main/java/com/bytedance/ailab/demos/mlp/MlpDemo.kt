// 神经网络 · 手写数字 (Rumelhart, Hinton & Williams 1986; presentation after 3Blue1Brown, "But what is a neural
// network?", 2017) — the app's opening demo.
//
// ONE IDEA: a network is layers of numbers. You write a digit and WATCH it light up layer by layer:
//   pad (your ink) -> input wall (the 28x28 the network reads: MNIST-normalised, centred by mass)
//   -> 64 neurons -> 32 neurons -> 10 output bars. Between walls only the strongest contributions w*a are drawn
//   as threads (blue = pushes up, red = pushes down), and a wave of light carries the new input through.
// Pinch a first-layer neuron: its 784 incoming weights appear as a 28x28 map — stroke pieces, not digits.
// Pinch an output bar: the chain of evidence that supports that answer lights up in gold.
//
// REAL MODEL: core/mlp (Mlp, StrokeRaster, TopK), weights trained by tools/train_mlp.py on synthetic digits
// (system fonts + pen-stroke templates); Kotlin == PyTorch to 3e-6 (MlpReferenceTest), held-out 99 %.
// Device budget: inference (52 k MACs) and thread selection only when the input changes; per frame only
// transforms / setMat; the input image is 4 vertex bands (U = brightness through a ramp texture, OptGeo's trick),
// re-uploaded at <= 7 Hz while writing; the weight map is one baked atlas, a neuron pick edits a 4-vertex quad.

package com.bytedance.ailab.demos.mlp

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import com.bytedance.ailab.core.P3
import com.bytedance.ailab.core.mlp.DigitAssets
import com.bytedance.ailab.core.mlp.Edge
import com.bytedance.ailab.core.mlp.Ink
import com.bytedance.ailab.core.mlp.Mlp
import com.bytedance.ailab.core.mlp.StrokeRaster
import com.bytedance.ailab.core.mlp.TopK
import com.bytedance.ailab.lab.About
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
import androidx.compose.ui.unit.sp
import com.pico.spatial.ui.design.PicoTheme
import com.pico.spatial.ui.design.Text
import java.io.File
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

object MlpDemo : SpecDemo() {
    override val id = "mlp"
    override val title = "神经网络 · 手写数字"  // i18n-ok (zh half of DemoRegistry.titleOf; English in TITLE_EN)
    override val credit = "Rumelhart · 1986"

    // ---------------------------------------------------------------- model + shipped data
    private val net by lazy { Mlp.load(LabAssets.bytes("mlp/mlp_w.bin")) }
    private val samples by lazy { DigitAssets.samples(LabAssets.bytes("mlp/samples.bin")) }
    private val glyphBytes by lazy { DigitAssets.glyphs(LabAssets.bytes("mlp/glyphs.bin")) }

    // ---------------------------------------------------------------- layout (volume metres)
    private const val EYE_Z = 2.5f
    private const val PAD = 0.27f                       // drawing pad side
    private const val CELLS = 5                         // gaze start cells per side (6 cm: >= eye-tracking error)
    private const val IN_S = 0.235f                     // input wall image side
    private const val PITCH = 0.044f                    // neuron spacing
    private const val OUT_PITCH = 0.046f
    private const val NEURON_R = 0.0135f
    private const val BAR_LEN = 0.19f
    private const val GAIN = 1.4f                       // hand metres -> pad metres
    // an "L": you write bottom-left, the network reads the image above it, then 64 -> 32 -> 10 to the right
    private val PAD_F = Frame(P3(-0.665f, -0.050f, 0.22f))
    private val IN_F = Frame(P3(-0.660f, 0.285f, -0.02f))
    private val H1_F = Frame(P3(-0.120f, 0.170f, -0.07f))
    private val H2_F = Frame(P3(0.250f, 0.170f, -0.21f))
    private val OUT_F = Frame(P3(0.640f, 0.170f, 0.00f))
    private val CARD_F = Frame(P3(-0.120f, -0.118f, 0.03f))
    private const val CARD_S = 0.158f
    private const val PLINTH_Y = -0.262f

    /** A wall turned to face the default viewer; local x right, y up, z toward the viewer. */
    private class Frame(val c: P3) {
        val yaw = atan2(-c.x, EYE_Z - c.z)
        val rot = Q.axisAngle(0f, 1f, 0f, yaw)
        private val rx = cos(yaw); private val rz = -sin(yaw); private val nx = sin(yaw); private val nz = cos(yaw)
        fun world(lx: Float, ly: Float, lz: Float = 0f) = P3(c.x + rx * lx + nx * lz, c.y + ly, c.z + rz * lx + nz * lz)
        /** Volume-space hand motion -> motion along this wall's own x axis. */
        fun alongX(dx: Float, dz: Float) = dx * rx + dz * rz
        val rightX get() = rx
        val rightZ get() = rz
    }

    // ---------------------------------------------------------------- palette (sRGB) — 3Blue1Brown-like
    private val INK = Rgb(0.90f, 0.97f, 1.00f)
    private val POS = Rgb(0.35f, 0.77f, 0.87f)          // 3b1b blue: pushes the next neuron UP
    private val NEG = Rgb(0.99f, 0.38f, 0.33f)          // 3b1b red: pushes it DOWN
    private val GOLD = Rgb(1.00f, 0.78f, 0.22f)
    private val N_OFF = Rgb(0.10f, 0.12f, 0.17f)
    private val N_ON = Rgb(0.88f, 0.98f, 1.00f)
    private val CYAN = Rgb(0.45f, 0.80f, 1.00f)

    private const val LEVELS = 16
    private val neuronMats = List(LEVELS + 1) { Mat.Unlit(ProcTex.okLerp(N_OFF, N_ON, (it / LEVELS.toFloat()).pow(0.8f))) }
    private val goldNeuron = Mat.Unlit(GOLD)
    private val glowMats = List(LEVELS + 1) { k -> Mat.Unlit(scale(N_ON, 0.85f * k / LEVELS), tex = null, blend = Blend.ADD) }
    private val goldGlow = Mat.Unlit(scale(GOLD, 0.75f), blend = Blend.ADD)
    // thread brightness: base levels 1..5 by |w a|, +2 while the idle "heartbeat" pulse passes
    private val THREAD_LV = 8
    private val threadMats = List(3) { kind -> List(THREAD_LV) { k ->
        val base = when (kind) { 0 -> POS; 1 -> NEG; else -> GOLD }
        Mat.Unlit(scale(base, 0.25f + 0.75f * (k + 1f) / THREAD_LV), alpha = 1f, blend = Blend.ADD) } }

    // ---------------------------------------------------------------- user-facing state
    var noiseOn = false
    var showWeights = false
    var speedIdx = 1
    private val WAVE_S = floatArrayOf(2.6f, 1.3f, 0.6f)
    private val SPEED_LABELS get() = listOf(tr("慢", "Slow"), tr("中", "Medium"), tr("快", "Fast"))
    private const val NOISE_SIGMA = 0.30f

    // ---------------------------------------------------------------- input state
    private val ink = Ink()
    private var inkDirty = false
    private var rasterAge = 0f
    private var inkFromSample = false
    private var inkByUser = false
    private val xClean = FloatArray(784)
    private val xShown = FloatArray(784)
    private val rInfo = StrokeRaster.Info()
    private var inputVersion = 0
    private var penDown = false
    private var penU = 0.5f; private var penV = 0.5f
    private var dragCell = ""

    // ---------------------------------------------------------------- network display state
    private val act by lazy { net.newAct() }
    private val actClean by lazy { net.newAct() }
    private var hasResult = false
    private val disp = arrayOf(FloatArray(64), FloatArray(32), FloatArray(10))
    private val from = arrayOf(FloatArray(64), FloatArray(32), FloatArray(10))
    private val tgt = arrayOf(FloatArray(64), FloatArray(32), FloatArray(10))
    private var waveT = 1f
    private var waveDur = 1.3f
    private var dimK = 1f
    private var selH1 = -1
    private var pathCls = -1
    private val onPath = arrayOf(BooleanArray(64), BooleanArray(32), BooleanArray(10))

    // threads: three groups (input->64, 64->32, 32->10), pooled
    private const val K0 = 40; private const val K1 = 32; private const val K2 = 12
    private const val NT = K0 + K1 + K2
    private val thrA = Array(NT) { P3(0f, 0f, 0f) }
    private val thrB = Array(NT) { P3(0f, 0f, 0f) }
    private val thrRot = Array(NT) { Q.identity() }
    private val thrR = FloatArray(NT)
    private val thrKind = IntArray(NT)
    private val thrLv = IntArray(NT)
    private var idleT = 0f
    private val thrOn = BooleanArray(NT)
    private val thrShown = FloatArray(NT) { -1f }
    private val thrCount = IntArray(3)

    // ---------------------------------------------------------------- scene nodes
    private lateinit var h1Nodes: Array<Node>
    private lateinit var h2Nodes: Array<Node>
    private lateinit var h1Glow: Array<Node>
    private lateinit var h2Glow: Array<Node>
    private lateinit var outNodes: Array<Node>
    private lateinit var outFill: Array<Node>
    private lateinit var outGlyph: Array<Node>
    private lateinit var thrNodes: Array<Node>
    private lateinit var inkDots: Array<Node>
    private lateinit var penTip: Node
    private lateinit var padGrid: Node
    private var padPing = 0f
    private val padGridMats by lazy { List(5) { k -> Mat.Unlit(scale(CYAN, 0.30f + 0.18f * k), tex = padGridTex, blend = Blend.ADD) } }
    private lateinit var selRing: Node
    private lateinit var card: Node
    private lateinit var cardMesh: MeshData
    private lateinit var cardLink: Node
    private lateinit var bands: Array<MeshData>
    private val bandRows = arrayOf(0..7, 7..14, 14..21, 21..27)
    private val bandShown = Array(4) { FloatArray(0) }

    // ink dots (light painting on the pad)
    private const val DOTS = 140
    private const val DOT_STEP = 0.028f                  // pad units between dots
    private var dotsUsed = 0
    private var dotCarry = 0f

    // ---------------------------------------------------------------- counters the guided track reads
    private var sampleLoads = 0
    private var wavesDone = 0
    private var userStrokes = 0
    private var neuronPicks = 0
    private var time = 0f
    private var introStarted = false

    // ---------------------------------------------------------------- textures
    private val softDisc by lazy {
        OptGeo.lumTex(64, 64) { u, v ->
            val r = sqrt((u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f)) * 2f
            (exp(-r * r * 5f) * (1f - smooth(0.85f, 1f, r))).coerceIn(0f, 1f)
        }
    }
    private val padGridTex by lazy {
        val n = 224; val img = Image(n, n)
        for (y in 0 until n) for (x in 0 until n) {
            val line = x % 8 == 0 || y % 8 == 0 || x == n - 1 || y == n - 1
            val major = x % 56 == 0 || y % 56 == 0
            val c = if (major) 0.55f else if (line) 0.30f else 0f
            img.px[y * n + x] = ProcTex.argb(1f, c, c, c)
        }
        img
    }
    private val rampTex by lazy { OptGeo.rampTex(1) { 1f } }
    private val glyphTex by lazy {
        List(10) { d ->
            val g = glyphBytes[d]; val n = DigitAssets.GLYPH
            Image(n, n).also { im -> for (k in 0 until n * n) { val a = (g[k].toInt() and 0xFF) / 255f; im.px[k] = ProcTex.argb(1f, a, a, a) } }
        }
    }
    /** All 64 first-layer weight maps as an 8 x 8 atlas of 30 px tiles (28 + 1 px black border). */
    private val atlas by lazy {
        val n = 240; val img = Image(n, n)
        for (j in 0 until 64) {
            var m = 1e-9f
            for (i in 0 until 784) m = max(m, abs(net.w1[j * 784 + i]))
            val ty = (j / 8) * 30 + 1; val tx = (j % 8) * 30 + 1
            for (r in 0 until 28) for (c in 0 until 28) {
                val w = net.w1[j * 784 + r * 28 + c] / m
                val a = abs(w).pow(0.75f)
                val col = if (w >= 0) POS else NEG
                img.px[(ty + r) * n + tx + c] = ProcTex.argb(1f, OptGeo.enc(ProcTex.toLinear(col.r) * a),
                    OptGeo.enc(ProcTex.toLinear(col.g) * a), OptGeo.enc(ProcTex.toLinear(col.b) * a))
            }
        }
        for (k in img.px.indices) if (img.px[k] == 0) img.px[k] = ProcTex.argb(1f, 0f, 0f, 0f)
        img
    }

    // ---------------------------------------------------------------- materials
    private val slabTex by lazy {
        ProcTex.bakePlanet(256, 128, Rgb(0.06f, 0.065f, 0.08f), Rgb(0.10f, 0.105f, 0.125f), Rgb(0.16f, 0.17f, 0.19f), 0f, 91)
    }
    private val slab get() = Mat.Pbr(Rgb(1f, 1f, 1f), metallic = 0f, roughness = 0.55f, tex = slabTex)
    private val alu = Mat.Pbr(Rgb(0.78f, 0.80f, 0.84f), metallic = 1f, roughness = 0.30f)
    private val blackGlass = Mat.Pbr(Rgb(0.02f, 0.024f, 0.035f), metallic = 0f, roughness = 0.12f)
    private val rimMat = Mat.Unlit(scale(CYAN, 0.55f), blend = Blend.ADD)

    // ================================================================ build

    override fun buildSpec(cacheDir: File): Node {
        resetState()
        val r = Node("MlpRoot")
        val unitCyl = MeshGen.cylinder(1f, 1f, 6)
        val sphere = MeshGen.sphere(1f, 10, 18, MeshGen.SphereUv.EQUIRECT)
        val disc = OptGeo.quad(1f, 1f)

        // ---- plinth: charcoal slab + aluminium edge, the bench every lab demo stands on
        r.add("Plinth", MeshGen.box(1.76f, 0.03f, 0.70f), slab, P3(0f, PLINTH_Y, 0.03f))
        r.add("PlinthBand", MeshGen.box(1.77f, 0.008f, 0.71f), alu, P3(0f, PLINTH_Y, 0.03f))

        // ---- drawing pad
        val pad = wall(r, "Pad", PAD_F, PAD + 0.016f, PAD + 0.016f)
        padGrid = pad.add("PadGrid", OptGeo.quad(PAD, PAD), padGridMats[0], P3(0f, 0f, 0.006f))
        val dotMesh = MeshGen.sphere(1f, 6, 10, MeshGen.SphereUv.EQUIRECT)
        for (row in 0 until CELLS) for (col in 0 until CELLS) {
            val u = (col + 0.5f) / CELLS; val v = (row + 0.5f) / CELLS
            pad.add("Pad_${row}_$col", dotMesh, Mat.Unlit(scale(CYAN, 0.55f), blend = Blend.ADD), P3(padX(u), padY(v), 0.008f)).also {
                it.uniformScale(0.0028f); it.tapRadius = 0.045f / 0.0028f
            }
        }
        inkDots = Array(DOTS) { k ->
            r.add("Ink$k", disc, Mat.Unlit(INK, blend = Blend.ADD, tex = softDisc)).also {
                it.rot = PAD_F.rot; it.uniformScale(0.034f); it.visible = false
            }
        }
        penTip = r.add("PenTip", disc, Mat.Unlit(scale(GOLD, 1.0f), blend = Blend.ADD, tex = softDisc)).also {
            it.rot = PAD_F.rot; it.uniformScale(0.055f); it.visible = false
        }

        // ---- sample digits (2 x 5 tiles under the pad) and the clear button
        for (d in 0 until 10) {
            val p = samplePos(d)
            val t = r.add(Node("Sample_$d")).also { it.pos = p; it.rot = PAD_F.rot }
            t.add("SampleBack_$d", MeshGen.box(0.050f, 0.050f, 0.006f), blackGlass, P3(0f, 0f, -0.004f))
            t.add("SampleGlyph_$d", OptGeo.quad(0.040f, 0.040f), Mat.Unlit(scale(CYAN, 0.95f), tex = glyphTex[d], blend = Blend.ADD), P3(0f, 0f, 0.0015f))
            t.tapRadius = 0.045f
        }
        val clear = r.add(Node("Clear")).also { it.pos = PAD_F.world(PAD / 2 + 0.087f, PAD / 2 + 0.058f, 0.03f); it.rot = PAD_F.rot }
        clear.tapRadius = 0.045f
        clear.add("ClearRing", MeshGen.annulus(0.019f, 0.023f, 40), Mat.Unlit(scale(NEG, 0.9f), blend = Blend.ADD)).also {
            it.rot = Q.axisAngle(1f, 0f, 0f, (PI / 2).toFloat())
        }
        for (s in listOf(1f, -1f)) clear.add("ClearX", unitCyl, Mat.Unlit(scale(NEG, 0.9f), blend = Blend.ADD)).also {
            it.scale = P3(0.0022f, 0.026f, 0.0022f); it.rot = Q.axisAngle(0f, 0f, 1f, s * (PI / 4).toFloat())
        }

        // ---- arrow pad -> input wall ("what the network reads")
        val a0 = PAD_F.world(-0.080f, PAD / 2 + 0.016f, 0.01f); val a1 = IN_F.world(-0.080f, -IN_S / 2 - 0.026f, 0.01f)
        stick(r, "Arrow", unitCyl, a0, a1, 0.0018f, Mat.Unlit(scale(CYAN, 0.7f), blend = Blend.ADD))
        for (s in listOf(1f, -1f)) stick(r, "ArrowHead", unitCyl, P3(a1.x + s * 0.014f, a1.y - 0.016f, a1.z + 0.004f), a1, 0.0018f, Mat.Unlit(scale(CYAN, 0.7f), blend = Blend.ADD))

        // ---- input wall: the 28x28 the network reads, as 4 vertex bands (U = brightness)
        val inW = wall(r, "In", IN_F, IN_S + 0.022f, IN_S + 0.022f)
        inW.add("InGrid", OptGeo.quad(IN_S, IN_S), Mat.Unlit(scale(CYAN, 0.16f), tex = padGridTex, blend = Blend.ADD), P3(0f, 0f, 0.005f))
        bands = Array(4) { b ->
            val rows = bandRows[b]; val md = OptGeo.grid(rows.last - rows.first + 1, 28)
            for ((k, row) in rows.withIndex()) for (c in 0 until 28) {
                val i = k * 28 + c
                OptGeo.setPos(md, i, inX(c), inY(row), 0.007f); OptGeo.setUv(md, i, OptGeo.rampU(0f), 0.5f)
            }
            inW.add("InBand$b", md, Mat.Unlit(INK, tex = rampTex, blend = Blend.ADD))
            md
        }
        // the MNIST 20x20 box the ink was scaled into
        val hb = IN_S * 10f / 28f; val box = Merge(); val zr = Q.axisAngle(0f, 0f, 1f, (PI / 2).toFloat()); val bs = P3(0.0009f, 2 * hb, 0.0009f)
        box.add(unitCyl, P3(0f, -hb, 0.006f), zr, bs); box.add(unitCyl, P3(0f, hb, 0.006f), zr, bs)
        box.add(unitCyl, P3(-hb, 0f, 0.006f), Q.identity(), bs); box.add(unitCyl, P3(hb, 0f, 0.006f), Q.identity(), bs)
        inW.add("Box20", box.build(), Mat.Unlit(scale(GOLD, 0.35f), blend = Blend.ADD))

        // ---- hidden layers
        val h1W = wall(r, "H1", H1_F, 8 * PITCH + 0.03f, 8 * PITCH + 0.03f)
        h1Nodes = Array(64) { j ->
            h1W.add("H1_$j", sphere, neuronMats[0], P3(h1X(j), h1Y(j), 0.012f)).also { it.uniformScale(NEURON_R); it.tapRadius = 0.045f / NEURON_R }
        }
        h1Glow = Array(64) { j -> h1W.add("H1Glow$j", disc, glowMats[0].copy(tex = softDisc), P3(h1X(j), h1Y(j), 0.016f)).also { it.uniformScale(0.066f); it.visible = false } }
        val h2W = wall(r, "H2", H2_F, 4 * PITCH + 0.03f, 8 * PITCH + 0.03f)
        h2Nodes = Array(32) { j -> h2W.add("H2n$j", sphere, neuronMats[0], P3(h2X(j), h2Y(j), 0.012f)).also { it.uniformScale(NEURON_R) } }
        h2Glow = Array(32) { j -> h2W.add("H2Glow$j", disc, glowMats[0].copy(tex = softDisc), P3(h2X(j), h2Y(j), 0.016f)).also { it.uniformScale(0.066f); it.visible = false } }
        // H2 is not pinchable (no automatic halo): the same thin ring, drawn, so all neurons look alike
        val ring = MeshGen.annulus(0.86f, 1.0f, 32); val rings = Merge(); val rs = NEURON_R * 1.6f
        for (j in 0 until 32) rings.add(ring, P3(h2X(j), h2Y(j), 0.014f), Q.axisAngle(1f, 0f, 0f, (PI / 2).toFloat()), P3(rs, rs, rs))
        h2W.add("H2Rings", rings.build(), Mat.Unlit(scale(CYAN, 0.22f), blend = Blend.ADD))
        selRing = h1W.add("SelRing", MeshGen.annulus(0.0205f, 0.0255f, 36), Mat.Unlit(GOLD, blend = Blend.ADD)).also {
            it.rot = Q.axisAngle(1f, 0f, 0f, (PI / 2).toFloat()); it.visible = false
        }

        // ---- output: 10 labelled bars, 0 at the top (like 3Blue1Brown's output column)
        val outW = wall(r, "Out", OUT_F, 0.32f, 10 * OUT_PITCH + 0.02f)
        val fillMesh = MeshGen.box(1f, 1f, 1f)
        outFill = Array(10) { d -> outW.add("OutFill_$d", fillMesh, barMat(0f, false), P3(barX0() + 0.001f, outY(d), 0.010f)).also { it.scale = P3(0.002f, 0.022f, 0.008f) } }
        for (d in 0 until 10) outW.add("OutTrack_$d", MeshGen.box(BAR_LEN, 0.024f, 0.003f), Mat.Unlit(Rgb(0.09f, 0.12f, 0.17f)), P3(barX0() + BAR_LEN / 2, outY(d), 0.006f))
        outGlyph = Array(10) { d -> outW.add("OutGlyph_$d", OptGeo.quad(0.046f, 0.046f), glyphMat(0f, false, d), P3(glyphX(), outY(d), 0.010f)) }
        outNodes = Array(10) { d ->
            outW.add("Out_$d", sphere, neuronMats[0], P3(beadX(), outY(d), 0.012f)).also { it.uniformScale(0.0125f); it.tapRadius = 0.045f / 0.0125f }
        }

        // ---- weight card (under the 64-neuron wall) + its link to the picked neuron
        card = r.add(Node("Card")).also { it.pos = CARD_F.c; it.rot = CARD_F.rot; it.visible = false }
        card.add("CardBack", MeshGen.box(CARD_S + 0.016f, CARD_S + 0.016f, 0.006f), blackGlass, P3(0f, 0f, -0.005f))
        frameBars(card, CARD_S + 0.016f, CARD_S + 0.016f, unitCyl, Mat.Unlit(scale(GOLD, 0.55f), blend = Blend.ADD))
        cardMesh = OptGeo.quad(CARD_S, CARD_S)
        card.add("CardMap", cardMesh, Mat.Unlit(Rgb(1f, 1f, 1f), tex = atlas, blend = Blend.ADD), P3(0f, 0f, 0.001f))
        cardLink = r.add("CardLink", unitCyl, Mat.Unlit(scale(GOLD, 0.6f), blend = Blend.ADD)).also { it.visible = false }

        // ---- thread pool
        thrNodes = Array(NT) { k -> r.add("Thr$k", unitCyl, threadMats[0][0]).also { it.visible = false } }

        bandShown.indices.forEach { bandShown[it] = FloatArray(0) }
        updateBands()
        updateLabels()
        return r
    }

    /**
     * A dark wall with an aluminium + glow rim, turned toward the viewer. The back is OPAQUE on purpose: a
     * see-through wall is sorted against the light threads by distance (per entity on the headset), so threads
     * crossing in front of it flickered between bright and swallowed; opaque, they simply depth-test.
     */
    private fun wall(r: Node, name: String, f: Frame, w: Float, h: Float): Node {
        val g = r.add(Node("${name}Wall")).also { it.pos = f.c; it.rot = f.rot }
        g.add("${name}Back", MeshGen.box(w, h, 0.008f), blackGlass, P3(0f, 0f, -0.004f))
        frameBars(g, w, h, MeshGen.cylinder(1f, 1f, 8), alu)
        g.add("${name}Rim", MeshGen.box(w + 0.006f, 0.0025f, 0.002f), rimMat, P3(0f, -h / 2 - 0.006f, 0.004f))
        // slim post down to the plinth
        val bottom = f.c.y - h / 2
        if (bottom - PLINTH_Y > 0.04f && name != "Pad")
            g.add("${name}Post", MeshGen.cylinder(1f, 1f, 12), alu, P3(0f, (bottom + PLINTH_Y) / 2 - f.c.y - 0.004f, -0.012f)).also { it.scale = P3(0.006f, bottom - PLINTH_Y, 0.006f) }
        return g
    }

    /** The four frame bars as ONE static mesh (node budget). */
    private fun frameBars(g: Node, w: Float, h: Float, cyl: MeshData, m: Mat) {
        val rr = 0.0032f; val zr = Q.axisAngle(0f, 0f, 1f, (PI / 2).toFloat())
        val b = Merge()
        b.add(cyl, P3(0f, h / 2, 0.002f), zr, P3(rr, w + 2 * rr, rr)); b.add(cyl, P3(0f, -h / 2, 0.002f), zr, P3(rr, w + 2 * rr, rr))
        b.add(cyl, P3(-w / 2, 0f, 0.002f), Q.identity(), P3(rr, h, rr)); b.add(cyl, P3(w / 2, 0f, 0.002f), Q.identity(), P3(rr, h, rr))
        g.add("Frame", b.build(), m)
    }

    /** Static geometry merged into one MeshData: parts placed by translation, rotation and (per-axis) scale. */
    private class Merge {
        private val pos = ArrayList<Float>(); private val nrm = ArrayList<Float>(); private val uv = ArrayList<Float>(); private val idx = ArrayList<Int>()
        fun add(m: MeshData, t: P3, q: Q, s: P3) {
            val base = pos.size / 3
            for (k in 0 until m.vertexCount) {
                val p = q.rotate(floatArrayOf(m.pos[k * 3] * s.x, m.pos[k * 3 + 1] * s.y, m.pos[k * 3 + 2] * s.z))
                pos += p[0] + t.x; pos += p[1] + t.y; pos += p[2] + t.z
                val n = q.rotate(floatArrayOf(m.nrm[k * 3] / s.x, m.nrm[k * 3 + 1] / s.y, m.nrm[k * 3 + 2] / s.z))
                val l = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).coerceAtLeast(1e-9f)
                nrm += n[0] / l; nrm += n[1] / l; nrm += n[2] / l
                uv += m.uv[k * 2]; uv += m.uv[k * 2 + 1]
            }
            for (i in m.idx) idx += base + i
        }
        fun build() = MeshData(pos.toFloatArray(), nrm.toFloatArray(), uv.toFloatArray(), idx.toIntArray())
    }

    private fun stick(r: Node, name: String, cyl: MeshData, a: P3, b: P3, rad: Float, m: Mat): Node {
        val d = b - a; val l = d.length
        return r.add(name, cyl, m, (a + b) * 0.5f).also { it.scale = P3(rad, l, rad); it.rot = Q.fromTo(0f, 1f, 0f, d.x, d.y, d.z) }
    }

    // ---------------------------------------------------------------- geometry helpers (local wall coordinates)
    private fun padX(u: Float) = (u - 0.5f) * PAD
    private fun padY(v: Float) = (0.5f - v) * PAD
    private fun inX(c: Int) = ((c + 0.5f) / 28f - 0.5f) * IN_S
    private fun inY(r: Int) = (0.5f - (r + 0.5f) / 28f) * IN_S
    private fun h1X(j: Int) = (j % 8 - 3.5f) * PITCH
    private fun h1Y(j: Int) = (3.5f - j / 8) * PITCH
    private fun h2X(j: Int) = (j % 4 - 1.5f) * PITCH
    private fun h2Y(j: Int) = (3.5f - j / 4) * PITCH
    private fun outY(d: Int) = (4.5f - d) * OUT_PITCH
    // output row: neuron bead (threads arrive here) | digit label | probability bar
    private fun beadX() = -0.135f
    private fun glyphX() = -0.097f
    private fun barX0() = -0.068f
    /** Keypad beside the pad: 0-4 in the left column, 5-9 in the right, top to bottom. */
    private fun samplePos(d: Int): P3 = PAD_F.world(PAD / 2 + 0.052f + (d / 5) * 0.070f, 0.124f - (d % 5) * 0.062f, 0.03f)

    private fun scale(c: Rgb, k: Float) = Rgb(c.r * k, c.g * k, c.b * k)
    private fun norm(p: P3): P3 { val l = p.length.coerceAtLeast(1e-9f); return P3(p.x / l, p.y / l, p.z / l) }
    private fun smooth(e0: Float, e1: Float, x: Float): Float { val t = ((x - e0) / (e1 - e0)).coerceIn(0f, 1f); return t * t * (3 - 2 * t) }
    private fun level(b: Float) = (b.coerceIn(0f, 1f) * LEVELS + 0.5f).toInt().coerceIn(0, LEVELS)

    private val barMats = List(LEVELS + 1) { k -> Mat.Unlit(ProcTex.okLerp(Rgb(0.20f, 0.30f, 0.40f), N_ON, k / LEVELS.toFloat())) }
    private val barGold = Mat.Unlit(GOLD)
    private fun barMat(p: Float, win: Boolean) = if (win) barGold else barMats[level(p)]
    private val glyphMatsCache = HashMap<Int, Mat>()
    private fun glyphMat(p: Float, win: Boolean, d: Int): Mat {
        val key = d * 100 + if (win) 99 else level(p)
        return glyphMatsCache.getOrPut(key) {
            val c = if (win) GOLD else ProcTex.okLerp(scale(CYAN, 0.75f), N_ON, p.coerceIn(0f, 1f))
            Mat.Unlit(c, tex = glyphTex[d], blend = Blend.ADD)
        }
    }

    // world positions of every drawable unit (thread endpoints)
    // threads leave the input from just BEHIND its wall: the image stays clean, they emerge past its edge
    private fun pixelW(i: Int) = IN_F.world(inX(i % 28), inY(i / 28), -0.014f)
    // ...and end just in FRONT of the neuron beads: a thread crossing a row of neurons reads as one line
    private fun h1W(j: Int) = H1_F.world(h1X(j), h1Y(j), 0.030f)
    private fun h2W(j: Int) = H2_F.world(h2X(j), h2Y(j), 0.030f)
    private fun outW(d: Int) = OUT_F.world(beadX(), outY(d), 0.030f)

    // ================================================================ state

    private fun resetState() {
        ink.clear(); inkDirty = false; rasterAge = 0f; inkFromSample = false; inkByUser = false
        xClean.fill(0f); xShown.fill(0f); inputVersion = 0; penDown = false; dragCell = ""
        hasResult = false
        for (a in disp + from + tgt) a.fill(0f)
        waveT = 1f; dimK = 1f; selH1 = -1; pathCls = -1
        for (a in onPath) a.fill(false)
        thrOn.fill(false); thrShown.fill(-1f); thrCount.fill(0); idleT = 0f
        dotsUsed = 0; dotCarry = 0f
        sampleLoads = 0; wavesDone = 0; userStrokes = 0; neuronPicks = 0
        time = 0f; introStarted = false; cardShownFor = -2; padPing = 0f
        ghost = null; hand = null; autoChallenge = false
        got.fill(false); challengePoll = 99f; celebrate = 0f; settle = 0f; judged = true
        noiseOn = false; showWeights = false; speedIdx = 1
        labels.clear(); readoutText = ""
    }

    // ---------------------------------------------------------------- ink
    private fun clearInk() {
        ink.clear(); inkDirty = true; inkFromSample = false; inkByUser = false
        for (k in 0 until dotsUsed) inkDots[k].visible = false
        dotsUsed = 0; dotCarry = 0f
    }

    /** Add a point to the current stroke (or start one) and lay ink dots along the new segment. */
    private fun inkPoint(u: Float, v: Float, newStroke: Boolean) {
        if (ink.pointCount >= 1500) return
        val last = if (newStroke) null else ink.last()
        if (newStroke) ink.begin(u, v) else ink.add(u, v)
        inkDirty = true
        if (last == null) { dot(u, v); dotCarry = 0f; return }
        val du = u - last.first; val dv = v - last.second; val l = sqrt(du * du + dv * dv)
        var s = DOT_STEP - dotCarry
        while (s <= l) { dot(last.first + du * s / l, last.second + dv * s / l); s += DOT_STEP }
        dotCarry = l - (s - DOT_STEP)
    }

    private fun dot(u: Float, v: Float) {
        if (dotsUsed >= DOTS) return
        val n = inkDots[dotsUsed++]
        n.pos = PAD_F.world(padX(u), padY(v), 0.010f); n.visible = true
    }

    /** Rasterise the ink (MNIST-style), add noise if on, refresh the input wall. */
    private fun rasterNow() {
        StrokeRaster.rasterise(ink, xClean, info = rInfo)
        inputVersion++
        if (noiseOn && !rInfo.empty) {
            val rnd = java.util.Random(1000L + inputVersion % 7)
            for (k in 0 until 784) xShown[k] = (xClean[k] + NOISE_SIGMA * rnd.nextGaussian().toFloat()).coerceIn(0f, 1f)
        } else xClean.copyInto(xShown)
        inkDirty = false; rasterAge = 0f
        updateBands()
    }

    private fun updateBands() {
        for (b in 0 until 4) {
            val rows = bandRows[b]; val md = bands[b]
            val n = (rows.last - rows.first + 1) * 28
            val vals = FloatArray(n) { xShown[rows.first * 28 + it] }
            if (vals.contentEquals(bandShown[b])) continue
            for (i in 0 until n) OptGeo.setUv(md, i, OptGeo.rampU(vals[i]), 0.5f)
            bandShown[b] = vals
            meshEdited(md)
        }
    }

    // ---------------------------------------------------------------- the network responds
    /** The input is complete (pen lifted / sample written / noise toggled): run the net, launch the wave. */
    private fun commit() {
        rasterNow()
        pathCls = -1
        if (rInfo.empty) { hasResult = false; startWave(fadeOut = true); return }
        net.forward(xShown, act)
        net.forward(xClean, actClean)
        hasResult = true
        if (showWeights && selH1 < 0) selH1 = mostActiveH1()
        tgt[0].let { t -> for (j in 0 until 64) t[j] = act.h1[j] / net.actScale1 }
        tgt[1].let { t -> for (j in 0 until 32) t[j] = act.h2[j] / net.actScale2 }
        act.probs.copyInto(tgt[2])
        chooseThreads()
        startWave(fadeOut = false)
        judged = false; settle = 0f
    }

    private fun startWave(fadeOut: Boolean) {
        for (l in 0 until 3) for (k in disp[l].indices) from[l][k] = disp[l][k] * dimK
        for (l in 0 until 3) disp[l].indices.forEach { disp[l][it] = from[l][it] }
        dimK = 1f
        if (fadeOut) { for (t in tgt) t.fill(0f); thrOn.fill(false) }
        waveDur = if (fadeOut) 0.5f else WAVE_S[speedIdx]
        waveT = 0f
        thrShown.fill(-1f)
        for (k in 0 until NT) thrNodes[k].visible = false
    }

    fun replayWave() { if (hasResult) { for (l in 0 until 3) disp[l].fill(0f); startWave(fadeOut = false) } }

    private val waveRunning get() = waveT < 1f
    /** Current answer (-1 = no input) and its calibrated probability — read by tests. */
    val prediction: Int get() = if (hasResult) act.predicted else -1
    val confidence: Float get() = if (hasResult) act.confidence else 0f

    private fun mostActiveH1(): Int { var b = 0; for (j in 1 until 64) if (act.h1[j] > act.h1[b]) b = j; return b }

    /** Fill the thread pool: top-k |w a| per layer pair, or the weight-mode / path-mode selections. */
    private fun chooseThreads() {
        thrOn.fill(false); thrCount.fill(0)
        for (a in onPath) a.fill(false)
        if (!hasResult) return
        if (pathCls >= 0) {
            val p = TopK.supportPath(net, act, pathCls, k2 = 4, k1 = 3, k0 = 3)
            setGroup(0, p.toH1.take(K0), 2) { pixelW(it.src) to h1W(it.dst) }
            setGroup(1, p.toH2.take(K1), 2) { h1W(it.src) to h2W(it.dst) }
            setGroup(2, p.toOut.take(K2), 2) { h2W(it.src) to outW(it.dst) }
            p.toH1.forEach { onPath[0][it.dst] = true }; p.toH2.forEach { onPath[1][it.dst] = true }
            p.toOut.forEach { onPath[1][it.src] = true }; p.toH2.forEach { onPath[0][it.src] = true }
            onPath[2][pathCls] = true
            return
        }
        // only contributions INTO neurons that fire (ReLU > 0) are drawn, and into the winning class at the end:
        // the threads then trace how the lit pixels light the lit neurons that light the answer
        val on1 = (0 until 64).filter { act.h1[it] > 0f }.toIntArray()
        val on2 = (0 until 32).filter { act.h2[it] > 0f }.toIntArray()
        val e0 = if (showWeights && selH1 >= 0) TopK.strongest(net.w1, act.x, 64, 784, K0 / 2, intArrayOf(selH1))
                 else TopK.strongest(net.w1, act.x, 64, 784, K0, on1)
        setGroup(0, e0, -1) { pixelW(it.src) to h1W(it.dst) }
        setGroup(1, TopK.strongest(net.w2, act.h1, 32, 64, K1, on2), -1) { h1W(it.src) to h2W(it.dst) }
        setGroup(2, TopK.strongest(net.w3, act.h2, 10, 32, K2, intArrayOf(act.predicted)), -1) { h2W(it.src) to outW(it.dst) }
    }

    private fun setGroup(g: Int, edges: List<Edge>, kind: Int, ends: (Edge) -> Pair<P3, P3>) {
        val base = when (g) { 0 -> 0; 1 -> K0; else -> K0 + K1 }
        val cap = when (g) { 0 -> K0; 1 -> K1; else -> K2 }
        val mx = edges.maxOfOrNull { abs(it.value) } ?: 1f
        for ((k, e) in edges.take(cap).withIndex()) {
            val i = base + k
            val (a, b) = ends(e)
            thrA[i] = a; thrB[i] = b
            val d = b - a
            thrRot[i] = Q.fromTo(0f, 1f, 0f, d.x, d.y, d.z)
            val m = abs(e.value) / mx
            thrR[i] = 0.0010f + 0.0024f * m
            val kd = if (kind >= 0) kind else if (e.value >= 0f) 0 else 1
            thrKind[i] = kd; thrLv[i] = 1 + (m * 4.99f).toInt()
            thrNodes[i].setMat(threadMats[kd][thrLv[i]])
            thrOn[i] = true
        }
        thrCount[g] = min(cap, edges.size)
    }

    // ---------------------------------------------------------------- ghost pen (sample digits)
    private class Ghost(val strokes: List<FloatArray>, val dur: Float) {
        val lens = strokes.map { s -> var l = 0f; for (i in 1 until s.size / 2) l += hypot(s[i * 2] - s[i * 2 - 2], s[i * 2 + 1] - s[i * 2 - 1]); l }
        val total = lens.sum().coerceAtLeast(1e-6f)
        var t = 0f; var stroke = 0; var idx = 0
        companion object { fun hypot(a: Float, b: Float) = sqrt(a * a + b * b) }
    }
    private var ghost: Ghost? = null
    val ghostBusy get() = ghost != null

    /** Pinch a sample tile: the pen writes that digit on the pad (same ink path as a hand), then the wave. */
    fun loadSample(d: Int, dur: Float = 0.9f) {
        hand = null; penDown = false
        clearInk(); inkFromSample = true
        ghost = Ghost(samples[d], dur)
        sampleLoads++
        dimK = 0.35f; thrOn.fill(false); for (k in 0 until NT) thrNodes[k].visible = false
    }

    private fun stepGhost(dt: Float) {
        val g = ghost ?: return
        g.t += dt
        val target = if (g.t >= g.dur) Float.MAX_VALUE else g.total * g.t / g.dur
        var walked = g.lens.take(g.stroke).sum()
        while (g.stroke < g.strokes.size) {
            val s = g.strokes[g.stroke]; val n = s.size / 2
            if (g.idx == 0) { inkPoint(s[0], s[1], true); g.idx = 1 }
            var acc = walked
            for (i in 1 until g.idx) acc += Ghost.hypot(s[i * 2] - s[i * 2 - 2], s[i * 2 + 1] - s[i * 2 - 1])
            while (g.idx < n) {
                val seg = Ghost.hypot(s[g.idx * 2] - s[g.idx * 2 - 2], s[g.idx * 2 + 1] - s[g.idx * 2 - 1])
                if (acc + seg > target + 1e-6f) break
                acc += seg; inkPoint(s[g.idx * 2], s[g.idx * 2 + 1], false); g.idx++
            }
            if (g.idx < n) { penU = s[(g.idx - 1) * 2]; penV = s[(g.idx - 1) * 2 + 1]; break }
            walked += g.lens[g.stroke]; g.stroke++; g.idx = 0
        }
        if (g.stroke >= g.strokes.size) { ghost = null; commit() }
        else if (inkDirty && rasterAge > 0.14f) rasterNow()
    }

    // ---------------------------------------------------------------- pinch-and-drag writing
    override fun isDragTarget(name: String) = name.startsWith("Pad_")

    override fun onDragStart(name: String) {
        val rc = name.removePrefix("Pad_").split("_").mapNotNull { it.toIntOrNull() }
        if (rc.size != 2) return
        if (ghost != null || inkFromSample) { ghost = null; clearInk() }
        dragCell = name
        penU = (rc[1] + 0.5f) / CELLS; penV = (rc[0] + 0.5f) / CELLS
        penDown = true; inkByUser = true; pathCls = -1
        inkPoint(penU, penV, true)
        thrOn.fill(false); for (k in 0 until NT) thrNodes[k].visible = false
    }

    override fun onDrag(name: String, dx: Float, dy: Float, dz: Float) {
        if (!penDown) return
        val du = PAD_F.alongX(dx, dz) * GAIN / PAD
        val dv = -dy * GAIN / PAD
        penU = (penU + du).coerceIn(0.02f, 0.98f); penV = (penV + dv).coerceIn(0.02f, 0.98f)
        val l = ink.last()
        if (l == null || abs(penU - l.first) + abs(penV - l.second) > 0.004f) inkPoint(penU, penV, false)
    }

    override fun onDragEnd(name: String) {
        if (!penDown) return
        penDown = false; dragCell = ""
        userStrokes++
        commit()
    }

    /** What a hand would do: write [strokes] stroke by stroke through the drag API (lesson autoplay). */
    private class Hand(val strokes: List<FloatArray>) { var stroke = 0; var idx = 0; var pause = 0.2f; var cell = ""; var offU = 0f; var offV = 0f }
    private var hand: Hand? = null

    fun autoWrite(d: Int) { ghost = null; clearInk(); inkFromSample = false; hand = Hand(samples[d]) }

    private fun stepHand(dt: Float) {
        val h = hand ?: return
        if (h.pause > 0f) { h.pause -= dt; return }
        if (h.stroke >= h.strokes.size) { hand = null; return }
        val s = h.strokes[h.stroke]; val n = s.size / 2
        if (h.idx == 0) {
            val col = (s[0] * CELLS).toInt().coerceIn(0, CELLS - 1); val row = (s[1] * CELLS).toInt().coerceIn(0, CELLS - 1)
            h.cell = "Pad_${row}_$col"
            onDragStart(h.cell)
            h.offU = penU - s[0]; h.offV = penV - s[1]; h.idx = 1
            return
        }
        // ~1.2 pad units per second, as a person writing in the air
        var budget = 1.2f * dt
        while (h.idx < n && budget > 0f) {
            val tu = s[h.idx * 2] + h.offU; val tv = s[h.idx * 2 + 1] + h.offV
            val du = tu - penU; val dv = tv - penV
            budget -= sqrt(du * du + dv * dv)
            val mx = du * PAD / GAIN
            onDrag(h.cell, mx * PAD_F.rightX, -dv * PAD / GAIN, mx * PAD_F.rightZ)
            h.idx++
        }
        if (h.idx >= n) { onDragEnd(h.cell); h.stroke++; h.idx = 0; h.pause = 0.25f }
    }

    // ================================================================ per frame

    override fun update(dt: Float) {
        time += dt
        if (!introStarted) { introStarted = true; loadSample(3, dur = 0.6f) }   // the opening flourish: a "3" writes itself
        stepGhost(dt)
        stepHand(dt)
        rasterAge += dt
        if (penDown && inkDirty && rasterAge > 0.14f) rasterNow()

        // the network waits (dimmed) while the pen is down
        if (penDown || ghost != null) dimK = max(0.35f, dimK - dt * 3f)

        if (waveT < 1f) {
            waveT = (waveT + dt / waveDur).coerceAtMost(1f)
            if (waveT >= 1f && hasResult) wavesDone++
        }
        val e = floatArrayOf(ease(win(0.22f, 0.40f)), ease(win(0.52f, 0.66f)), ease(win(0.80f, 1.0f)))
        for (l in 0 until 3) {
            val bump = sin(PI.toFloat() * e[l]) * 0.35f
            for (k in disp[l].indices) disp[l][k] = from[l][k] + (tgt[l][k] - from[l][k]) * e[l] + bump * tgt[l][k]
        }
        applyNeurons()
        applyThreads(dt)
        applyOutputs()
        applyCard()
        // a ping on any start cell (lesson pointer / a tap on the pad) lights the whole grid: the 4 mm cell rings
        // alone are too small to find from 2.5 m
        padPing = max(0f, padPing - dt)
        padGrid.setMat(padGridMats[(padPing / 0.9f * 4.99f).toInt().coerceIn(0, 4)])
        penTip.visible = penDown || ghost != null
        if (penTip.visible) penTip.pos = PAD_F.world(padX(penU), padY(penV), 0.016f)
        stepChallenge(dt)
        updateLabels()
        updateReadout()
    }

    private fun win(a: Float, b: Float) = ((waveT - a) / (b - a)).coerceIn(0f, 1f)
    private fun ease(t: Float) = t * t * (3 - 2 * t)

    private fun applyNeurons() {
        val path = pathCls >= 0
        for (j in 0 until 64) {
            var b = disp[0][j] * dimK
            if (path && !onPath[0][j]) b *= 0.22f
            h1Nodes[j].setMat(if (path && onPath[0][j] && waveT >= 0.4f) goldNeuron else neuronMats[level(b)])
            val lv = level(b * 0.9f)
            h1Glow[j].visible = lv > 1
            if (lv > 1) h1Glow[j].setMat(glowCache(lv, path && onPath[0][j]))
        }
        for (j in 0 until 32) {
            var b = disp[1][j] * dimK
            if (path && !onPath[1][j]) b *= 0.22f
            h2Nodes[j].setMat(if (path && onPath[1][j] && waveT >= 0.66f) goldNeuron else neuronMats[level(b)])
            val lv = level(b * 0.9f)
            h2Glow[j].visible = lv > 1
            if (lv > 1) h2Glow[j].setMat(glowCache(lv, path && onPath[1][j]))
        }
        selRing.visible = showWeights && selH1 >= 0
        if (selRing.visible) selRing.pos = P3(h1X(selH1), h1Y(selH1), 0.020f)
    }

    private val glowMatCache = HashMap<Int, Mat>()
    private fun glowCache(lv: Int, gold: Boolean): Mat = glowMatCache.getOrPut(lv * 2 + if (gold) 1 else 0) {
        (if (gold) goldGlow else glowMats[lv]).copy(tex = softDisc)
    }

    private fun applyThreads(dt: Float) {
        val grow = floatArrayOf(win(0.0f, 0.28f), win(0.36f, 0.58f), win(0.62f, 0.84f))
        // idle heartbeat: once the answer is in, a soft pulse runs input -> 64 -> 32 -> 10 every 3.2 s
        idleT = if (waveT >= 1f && hasResult && !penDown && ghost == null) idleT + dt else 0f
        val ph = if (idleT < 1.2f) -1f else (idleT - 1.2f) % 3.2f
        for (i in 0 until NT) {
            if (thrOn[i] && thrShown[i] >= 1f) {
                val g = if (i < K0) 0 else if (i < K0 + K1) 1 else 2
                val x = (ph - g * 0.30f) / 0.45f
                val boost = if (ph < 0f || x < 0f || x > 1f) 0 else (2.4f * sin(PI.toFloat() * x)).toInt()
                thrNodes[i].setMat(threadMats[thrKind[i]][(thrLv[i] + boost).coerceAtMost(THREAD_LV - 1)])
            }
            val g = if (!thrOn[i] || penDown || ghost != null) 0f else grow[if (i < K0) 0 else if (i < K0 + K1) 1 else 2]
            val q = (g * 40f).toInt() / 40f
            if (q == thrShown[i]) continue
            thrShown[i] = q
            val n = thrNodes[i]
            if (q <= 0f) { n.visible = false; continue }
            val a = thrA[i]; val b = thrB[i]; val d = b - a; val l = d.length * q
            n.visible = true
            n.rot = thrRot[i]
            val u = norm(d)
            n.pos = P3(a.x + u.x * l / 2, a.y + u.y * l / 2, a.z + u.z * l / 2)
            n.scale = P3(thrR[i], max(l, 1e-4f), thrR[i])
        }
    }

    private fun applyOutputs() {
        val done = hasResult && waveT >= 1f && !penDown && ghost == null
        val win = if (done) act.predicted else -1
        for (d in 0 until 10) {
            var p = disp[2][d].coerceIn(0f, 1.2f) * dimK
            if (pathCls >= 0 && d != pathCls) p *= 0.3f
            val len = max(0.002f, p.coerceAtMost(1f) * BAR_LEN)
            outFill[d].scale = P3(len, 0.022f, 0.008f)
            outFill[d].pos = P3(barX0() + len / 2, outY(d), 0.010f)
            val isWin = d == win && pathCls < 0 || d == pathCls && waveT >= 0.8f
            outFill[d].setMat(barMat(p, isWin))
            outGlyph[d].setMat(glyphMat(p, isWin, d))
            outFill[d].visible = p > 0.01f
            outNodes[d].setMat(if (isWin) goldNeuron else neuronMats[level(p)])
        }
    }

    private var cardShownFor = -2
    private fun applyCard() {
        val on = showWeights && selH1 >= 0
        card.visible = on; cardLink.visible = on
        if (!on || cardShownFor == selH1) return
        cardShownFor = selH1
        val tx = (selH1 % 8) * 30 + 1; val ty = (selH1 / 8) * 30 + 1
        val u0 = (tx + 0.5f) / 240f; val u1 = (tx + 27.5f) / 240f; val v0 = (ty + 0.5f) / 240f; val v1 = (ty + 27.5f) / 240f
        OptGeo.setUv(cardMesh, 0, u0, v0); OptGeo.setUv(cardMesh, 1, u1, v0); OptGeo.setUv(cardMesh, 2, u0, v1); OptGeo.setUv(cardMesh, 3, u1, v1)
        meshEdited(cardMesh)
        val a = CARD_F.world(0f, CARD_S / 2 + 0.010f, 0f); val b = H1_F.world(h1X(selH1), h1Y(selH1) - 0.016f, 0.014f)
        val d = b - a
        cardLink.pos = (a + b) * 0.5f; cardLink.scale = P3(0.0012f, d.length, 0.0012f); cardLink.rot = Q.fromTo(0f, 1f, 0f, d.x, d.y, d.z)
    }

    // ================================================================ interaction

    override fun flash(name: String) {
        super.flash(name)
        if (name.startsWith("Pad_")) padPing = 0.9f
    }

    override fun isTarget(name: String) = name.startsWith("Pad_") || name.startsWith("Sample_") || name == "Clear" ||
        name.startsWith("H1_") || name.startsWith("Out_")

    override val tapTargets get() = listOf(
        TapTarget("Clear", "清除 → 擦掉手写板", "Clear → wipe the pad"),
        TapTarget("Sample_0", "写 0", "Write 0"), TapTarget("Sample_1", "写 1", "Write 1"),
        TapTarget("Sample_2", "写 2", "Write 2"), TapTarget("Sample_3", "写 3", "Write 3"),
        TapTarget("Sample_4", "写 4", "Write 4"), TapTarget("Sample_5", "写 5", "Write 5"),
        TapTarget("Sample_6", "写 6", "Write 6"), TapTarget("Sample_7", "写 7", "Write 7"),
        TapTarget("Sample_8", "写 8", "Write 8"), TapTarget("Sample_9", "写 9", "Write 9"),
        TapTarget("H1_${nextNeuron()}", "隐藏神经元 → 看权重", "Hidden neuron → its weights") { showWeights },
        TapTarget("Out_${if (hasResult) act.predicted else 3}", "输出条 → 追溯证据", "Output bar → trace evidence") { pathCls >= 0 },
    )

    /** Panel twin of "pinch a neuron": the next most active first-layer neuron after the picked one. */
    private fun nextNeuron(): Int {
        if (!hasResult) return 27
        val order = (0 until 64).sortedByDescending { act.h1[it] }
        val k = order.indexOf(selH1)
        return order[(k + 1) % 8]
    }

    override fun onTap(name: String) {
        when {
            name == "Clear" -> { ghost = null; hand = null; penDown = false; clearInk(); commit() }
            name.startsWith("Sample_") -> name.removePrefix("Sample_").toIntOrNull()?.let { loadSample(it.coerceIn(0, 9)) }
            name.startsWith("H1_") -> name.removePrefix("H1_").toIntOrNull()?.let { pickNeuron(it.coerceIn(0, 63)) }
            name.startsWith("Out_") -> name.removePrefix("Out_").toIntOrNull()?.let { togglePath(it.coerceIn(0, 9)) }
        }
    }

    fun pickNeuron(j: Int) {
        selH1 = j; neuronPicks++
        showWeights = true; pathCls = -1
        if (hasResult) { chooseThreads(); for (i in 0 until K0) thrShown[i] = -1f }
    }

    fun applyShowWeights(on: Boolean) {
        showWeights = on
        if (on && selH1 < 0) selH1 = if (hasResult) mostActiveH1() else 27
        if (on) neuronPicks++
        if (hasResult) { chooseThreads(); thrShown.fill(-1f) }
    }

    fun togglePath(c: Int) {
        pathCls = if (pathCls == c) -1 else c
        if (pathCls >= 0) showWeights = false
        if (hasResult) { chooseThreads(); thrShown.fill(-1f) }
    }

    fun setNoise(on: Boolean) { noiseOn = on; if (!penDown && ghost == null) commit() }

    override fun tapMessage(name: String): String? = when {
        name == "Clear" -> tr("手写板已清空 —— 注视一个格点，捏住并移动手来写", "Pad cleared — look at a dot, pinch and move your hand to write")
        name.startsWith("Sample_") -> name.removePrefix("Sample_").let { tr("示例 $it：看笔写出它，再看光波穿过网络", "Sample $it: watch the pen write it, then the wave") }
        name.startsWith("H1_") -> tr("神经元 #%d：它的 784 个权重在下方 —— 蓝 = 有墨加分，红 = 减分", "Neuron #%d: its 784 weights below — blue adds, red subtracts").format(selH1)
        name.startsWith("Out_") -> if (pathCls >= 0) tr("金线：支持「%d」的最强证据链", "Gold: the strongest chain of evidence for \"%d\"").format(pathCls)
                                   else tr("证据链已关闭", "Evidence chain off")
        name.startsWith("Pad_") -> tr("捏住不放、移动手来写 —— 从你注视的格点起笔", "Pinch, hold and move your hand — the stroke starts at the dot you look at")
        else -> null
    }

    // ================================================================ readout + labels

    private fun updateReadout() {
        val l1 = when {
            penDown -> tr("正在书写…  松开捏合，网络才开始计算", "Writing…  release the pinch and the network computes")
            ghost != null -> tr("示例数字正在写到手写板上…", "The sample digit is being written on the pad…")
            !hasResult -> tr("还没有输入 —— 注视手写板，捏住并移动手写一个数字，或捏一个示例", "No input yet — look at the pad, pinch and move to write a digit, or pinch a sample")
            waveT < 0.8f -> tr("计算中：光波正在一层层传过网络…", "Computing: the wave is passing through the layers…")
            act.confidence >= 0.5f -> tr("预测：%d   置信度 %s   第二可能：%d（%s）", "Prediction: %d   confidence %s   runner-up: %d (%s)")
                .format(act.predicted, pct(act.confidence), act.runnerUp, pct(act.probs[act.runnerUp]))
            else -> tr("预测：%d？只有 %s 把握 —— 网络拿不准（第二可能 %d：%s）", "Prediction: %d? only %s sure — the network is unsure (next: %d at %s)")
                .format(act.predicted, pct(act.confidence), act.runnerUp, pct(act.probs[act.runnerUp]))
        }
        val a1 = if (hasResult) act.h1.count { it > 0f } else 0
        val a2 = if (hasResult) act.h2.count { it > 0f } else 0
        val nThr = thrCount.sum()
        val l2 = tr("784 → 64 → 32 → 10   被激活的神经元：%d/64 · %d/32   只画最强的 %d 条连线 w·a", "784 → 64 → 32 → 10   active neurons: %d/64 · %d/32   only the %d strongest links w·a drawn")
            .format(a1, a2, nThr)
        val l3 = when {
            pathCls >= 0 -> tr("金线 = 支持「%d」的最强正贡献链：像素 → 64 → 32 → 输出", "Gold = the strongest positive chain behind \"%d\": pixels → 64 → 32 → output").format(pathCls)
            showWeights && selH1 >= 0 -> tr("神经元 #%d 的权重：蓝 = 这里有墨就加分，红 = 减分 —— 它找的是笔画碎片", "Neuron #%d's weights: blue = ink here adds, red = subtracts — it looks for stroke pieces").format(selH1)
            noiseOn && hasResult && actClean.predicted == act.predicted -> tr("加噪声 σ=0.3：仍判为 %d（干净时 %s → 现在 %s）· 训练时见过带噪样例", "Noise σ=0.3: still %d (clean %s → now %s) · it saw noisy examples in training")
                .format(act.predicted, pct(actClean.probs[act.predicted]), pct(act.confidence))
            noiseOn && hasResult -> tr("加噪声 σ=0.3：答案从 %d 变成了 %d —— 噪声骗过了它", "Noise σ=0.3: the answer flipped from %d to %d — the noise fooled it")
                .format(actClean.predicted, act.predicted)
            else -> tr("每层都把上一层的特征重新组合；权重全从 8 万个合成数字学来（玩具模型，留出集 99%）", "Each layer recombines the last one's features; weights learned from 80k synthetic digits (toy, 99% held-out)")
        }
        readoutText = "$l1\n$l2\n$l3"
    }

    /** A calibrated probability, never rounded up to a false "100 %" or down to "0 %". */
    private fun pct(p: Float) = when { p > 0.995f -> ">99%"; p < 0.005f -> "<1%"; else -> "${(p * 100f + 0.5f).toInt()}%" }

    private fun updateLabels() {
        labels.clear()
        val top = { f: Frame, h: Float -> f.world(0f, h / 2 + 0.040f, 0.02f) }
        val padText = if (challengeActive) tr("请写：%d   已认对 %d/10", "Write: %d   done %d/10").format(ask(), got.count { it })
                      else tr("手写板 · 捏住移动", "Pad · pinch & move")
        labels += LabelSpec(padText, PAD_F.world(0.045f, PAD / 2 + 0.040f, 0.03f))
        labels += LabelSpec(tr("输入 28×28 = 784", "Input 28×28 = 784"), top(IN_F, IN_S))
        labels += LabelSpec(tr("隐藏层 64", "Hidden 64"), top(H1_F, 8 * PITCH))
        labels += LabelSpec(tr("隐藏层 32", "Hidden 32"), top(H2_F, 8 * PITCH))
        labels += LabelSpec(tr("输出 10", "Output 10"), top(OUT_F, 10 * OUT_PITCH))
        if (showWeights && selH1 >= 0)
            labels += LabelSpec(tr("#%d 的权重 · 蓝+ 红−", "#%d weights · blue+ red−").format(selH1), CARD_F.world(CARD_S / 2 + 0.12f, -0.02f, 0.03f))
    }

    // ================================================================ guided track (导学) + challenge

    private var lastWaves = 0; private var lastUser = 0; private var lastPicks = 0; private var lastLoads = 0

    // challenge: write 0..9 in turn, each recognised (from your own hand, not a sample)
    private val got = BooleanArray(10)
    private var challengePoll = 99f
    private val challengeActive get() = challengePoll < 0.6f
    private var autoChallenge = false
    private var celebrate = 0f
    private var settle = 0f
    private var judged = true
    private fun ask(): Int = (0 until 10).firstOrNull { !got[it] } ?: 9

    private fun stepChallenge(dt: Float) {
        challengePoll += dt
        if (!challengeActive) return
        if (celebrate > 0f) { celebrate -= dt; if (celebrate <= 0f) { clearInk(); commit() }; return }
        if (!judged && hasResult && !penDown && hand == null && waveT >= 1f) {
            settle += dt
            if (settle > 0.5f) {
                judged = true
                if (inkByUser && act.predicted == ask() && act.confidence >= 0.5f) { got[ask()] = true; celebrate = 1.0f }
            }
        }
        if (autoChallenge && hand == null && ghost == null && !penDown && celebrate <= 0f && ink.pointCount == 0 && got.any { !it }) autoWrite(ask())
    }

    // ---- background card (📖): context for a viewer who has never met this topic
    override val about = About(listOf(
        About.Page(
            Bi("这个实验在做什么？", "What is this experiment?"),
            Bi("左下是手写板：捏住、移动手就能写一个数字（或捏下方的示例 0–9）。它先变成上方 28×28 的灰度图，再一层层传过网络：784 个像素 → 64 个神经元 → 32 个 → 右边 10 根输出条。最亮的金色条就是网络的答案。面板可以加噪声、看权重、重放光波。", "Bottom left is a writing pad: pinch and move your hand to write a digit (or pinch a sample 0–9 below it). It becomes the 28×28 grey image above, then passes through the network layer by layer: 784 pixels → 64 neurons → 32 → the 10 output bars on the right. The brightest, gold bar is the answer. The panel adds noise, shows weights and replays the wave."),
        ),
        About.Page(
            Bi("先认识几个词", "A few words first"),
            Bi("「模型」是一台把输入变成答案的计算机器，这里是：图片进，数字出。机器里有许多可调的旋钮，叫「参数」（也叫权重），这个网络约有 5.3 万个。「训练」就是给它看大量带答案的例子，一点点拧旋钮，让它错得越来越少。没人告诉它 3 长什么样，它是从例子里自己调出来的。", "A “model” is a calculating machine that turns an input into an answer — here, picture in, digit out. Inside are many adjustable knobs called “parameters” (or weights); this network has about 53,000. “Training” means showing it lots of examples with known answers and nudging the knobs so it gets fewer wrong. Nobody tells it what a 3 looks like; it tunes itself from examples."),
        ),
        About.Page(
            Bi("背景：一个老想法", "Background: an old idea"),
            Bi("1958 年，罗森布拉特造出「感知机」：一层神经元能学会简单的图案，再难一点就不行了。多加几层本可以更强，难题是没人知道中间层的旋钮该怎么调。1986 年，鲁梅尔哈特、辛顿和威廉姆斯推广了「反向传播」，多层网络才真正训练得动。到 1990 年代末，类似的网络已在银行里读支票上的手写数字。", "In 1958 Frank Rosenblatt built the “perceptron”: one layer of neurons that could learn simple patterns, but nothing harder. More layers would be stronger — yet nobody knew how to tune the knobs of the middle layers. In 1986 Rumelhart, Hinton and Williams popularised “backpropagation”, and multi-layer networks became trainable. By the late 1990s similar networks read handwritten digits on bank cheques."),
        ),
        About.Page(
            Bi("核心想法：层层拼图", "Key idea: assemble the pieces"),
            Bi("每个神经元只是一个数：把上一层所有的数各乘一个权重再相加，负数变成 0（ReLU）。第一层的神经元各自找几段笔画；后面的层把碎片拼成「一个圈」「一条竖」；最后一层看哪些部件到齐了，就投票给哪个数字。就像认人：先看到眼睛、鼻子，再拼成一张脸。", "Each neuron is just one number: multiply every number in the previous layer by a weight, add them up, and turn negatives into 0 (ReLU). First-layer neurons each look for bits of stroke; later layers combine bits into “a loop” or “an upright”; the last layer checks which parts are present and votes for a digit. Like recognising a face: eyes and nose first, then the whole face."),
        ),
        About.Page(
            Bi("怎么看这个 3D 场景", "How to read the 3-D scene"),
            Bi("上方的方块墙 = 网络读到的 784 个像素；每个圆点 = 一个神经元，越亮数值越大；层间的细线 = 「权重 × 激活」最强的那些：蓝线把下一个神经元往上推，红线往下拉；光波 = 一次计算从左传到右。捏一个隐藏神经元，下方出现它的 784 个权重图；捏一根输出条，金线标出支持这个答案的证据链。", "The square wall above = the 784 pixels the network reads; each dot = a neuron, brighter = bigger value; threads between layers = the strongest weight × activity terms: blue pushes the next neuron up, red pulls it down; the wave of light = one computation travelling left to right. Pinch a hidden neuron to see its 784 weights as a map; pinch an output bar and gold threads trace the evidence for that answer."),
        ),
        About.Page(
            Bi("有多大？今天在哪里？", "How big? Where is it today?"),
            Bi("这是个玩具：约 5.3 万个参数，只在 8 万个电脑合成的数字上训练过，没见过真人笔迹，写得太潦草它也会认错。今天的大模型有上千亿个参数，但最基本的砖块仍是这里的「加权求和 + 非线性」。手机上的手写输入、快递单和支票的识别，都用到这一类网络。", "This is a toy: about 53,000 parameters, trained only on 80,000 computer-generated digits — no real handwriting — so very messy writing can fool it. Today's large models have hundreds of billions of parameters, yet their basic brick is still this “weighted sum + non-linearity”. Handwriting input on phones and the reading of parcel labels and cheques use networks of this kind."),
        ),
    ))

    override val lesson: Lesson by lazy {
        Lesson(
            listOf(
                LessonStep(
                    say = Bi("先猜一猜：第一个隐藏层里的一个神经元，在“看”输入图像的什么？", "First, a guess: what does ONE neuron of the first hidden layer \"look\" at in the image?"),
                    options = listOf(
                        Bi("一整个数字，比如完整的 3", "A whole digit, like a complete 3"),
                        Bi("几段笔画、局部的明暗块", "Stroke pieces and local patches"),
                        Bi("某一个像素", "One single pixel"),
                    ),
                    answer = 1,
                ),
                LessonStep(
                    say = Bi("捏一个示例数字（手写板下方，或面板上的“写 0–9”），看笔把它写出来。", "Pinch a sample digit (under the pad, or \"Write 0–9\" on the panel) and watch the pen write it."),
                    target = "Sample_7",
                    onEnter = { lastLoads = sampleLoads },
                    done = { sampleLoads > lastLoads && ghost == null },
                    autoplay = { onTap("Sample_7") },
                ),
                LessonStep(
                    say = Bi("看光波一层层传过去：输入 → 64 → 32 → 10。最亮的金色输出条就是答案。", "Watch the wave go layer by layer: input → 64 → 32 → 10. The gold output bar is the answer."),
                    target = "Out_7",
                    onEnter = { lastWaves = wavesDone },
                    done = { hasResult && !waveRunning && (wavesDone > lastWaves || waveT >= 1f) },
                    autoplay = { if (!hasResult) onTap("Sample_7") },
                    insight = Bi("每个神经元只是一个数：上一层所有数的加权和，再经过 ReLU（负数变 0）。", "Each neuron is just a number: a weighted sum of the whole previous layer, through ReLU (negatives become 0)."),
                ),
                LessonStep(
                    say = Bi("轮到你：注视手写板上要起笔的格点，捏住，移动手写一个数字，再松开。", "Your turn: look at the dot where your stroke starts, pinch, move your hand to write a digit, release."),
                    target = "Pad_1_2",
                    onEnter = { lastUser = userStrokes },
                    done = { userStrokes > lastUser && !penDown && hand == null && hasResult && !waveRunning },
                    autoplay = { autoWrite(4) },
                    insight = Bi("你的笔迹先被缩放进 20×20 的框、按“质心”居中（MNIST 的做法），网络才认得出 —— 写大写小都一样。", "Your ink is scaled into a 20×20 box and centred by its centre of mass (as MNIST was), so big or small writing looks the same."),
                ),
                LessonStep(
                    say = Bi("捏一个“隐藏层 64”里的神经元（或面板按钮），看它的权重图：它在找什么？", "Pinch a neuron in \"Hidden 64\" (or the panel button) and look at its weight map: what is it looking for?"),
                    target = "H1_27",
                    onEnter = { lastPicks = neuronPicks },
                    done = { neuronPicks > lastPicks && showWeights },
                    autoplay = { onTap("H1_27") },
                    reveal = true,
                    insight = Bi("蓝 = 这里有墨就加分，红 = 减分：它找的是几段笔画和明暗块，不是整个数字；下一层再把碎片拼起来。这些图案全是从例子里学来的，比“干净的笔画”乱得多。", "Blue = ink here adds, red = subtracts: it looks for stroke pieces, not a whole digit; the next layer assembles them. All learned from examples — messier than clean strokes."),
                ),
            ),
            challenge = Challenge(
                say = Bi("挑战：从 0 写到 9，每个都让网络认对（写错就捏“清除”重写）。", "Challenge: write 0 to 9, one at a time, and get each one recognised (pinch Clear to retry)."),
                progress = {
                    challengePoll = 0f
                    if (got.all { it }) tr("已认对 10/10 ✓", "Recognised 10/10 ✓")
                    else tr("已认对 %d/10 · 现在写：%d", "Recognised %d/10 · now write: %d").format(got.count { it }, ask())
                },
                done = { got.all { it } },
                autoplay = { autoChallenge = true; clearInk(); commit() },
            ),
        )
    }

    // ================================================================ panel

    @Composable
    override fun Panel() {
        var noise by FieldRef(::noiseOn)
        var wts by FieldRef(::showWeights)
        var si by FieldRef(::speedIdx)
        LabPanel(readout) {
            LabRow {
                LabSwitch(tr("加噪声", "Add noise"), noise) { noise = it; setNoise(it) }
                LabSwitch(tr("显示权重", "Show weights"), wts) { applyShowWeights(it); wts = it }
                Text(tr("光波速度", "Wave speed"), color = PicoTheme.colorScheme.labelSecondary, style = PicoTheme.typography.bodyLarge.copy(fontSize = 22.sp))
                SegmentControl {
                    SPEED_LABELS.forEachIndexed { k, l -> SegmentItem(selected = si == k, onClick = { si = k }, title = { Text(l) }) }
                }
                Button(onClick = { replayWave() }, enabled = hasResult) { Text(tr("重放光波", "Replay wave")) }
            }
        }
    }

    // ================================================================ review states

    override fun reviewVariants(): Map<String, () -> Unit> = mapOf(
        "weights" to { pickNeuron(if (hasResult) mostActiveH1() else 27) },
        "path" to { togglePath(if (hasResult) act.predicted else 3) },
        "noise" to { togglePath(-1); pathCls = -1; showWeights = false; setNoise(true) },
        "writing" to { setNoise(false); loadSample(8, dur = 3f); repeat(50) { update(1f / 60f) } },
    )
}
