// 反向传播 · 链式法则 (Rumelhart, Hinton & Williams, Nature 1986).
//
// ONE IDEA: learning = blame flowing backwards. A sample goes FORWARD (white light: input -> output p), the error
// p - y appears at the output, and flows BACKWARD (magenta) along every connection, scaled by the weights and by
// each neuron's slope f'(z) — the chain rule. Each weight then moves a little against its own gradient
// dL/dw = (activity at its input) x (blame at its output), and the decision surface over the input plane bends.
//
// Scene (left -> right): the input plane on a lectern, tilted toward the viewer, with the training points as
// beads riding on the decision SURFACE (height and colour = the network's p; white line = p = 0.5); the network in
// the air (2 -> 4|8 -> 4|8 -> 1, tanh, tanh, sigmoid): neurons as spheres (blue = +, red = -), weights as threads
// (thickness |w|, colour = sign), gradients as magenta sleeves (thickness |dL/dw|), blame delta as magenta rings.
// Front right: the dataset shelf (XOR, circles, spirals).
//
// REAL MODEL, trained live: core/backprop (Net, Trainer, Datasets) — plain per-sample SGD in double precision;
// checked against finite differences, Mazur's textbook example, a hand calculation and PyTorch (BackpropTest).
// Device budget: training <= 100 SGD steps/frame (~0.06 ms on the Mac); the 841-vertex sheet (4 strips) and the
// bead heights are re-evaluated only when the weights changed, at <= 5 Hz; threads/neurons move by transform/setMat.

package com.bytedance.ailab.demos.backprop

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.ui.unit.sp
import com.bytedance.ailab.core.P3
import com.bytedance.ailab.core.backprop.DataKind
import com.bytedance.ailab.core.backprop.Datasets
import com.bytedance.ailab.core.backprop.Net
import com.bytedance.ailab.core.backprop.Trainer
import com.bytedance.ailab.lab.About
import com.bytedance.ailab.lab.Bi
import com.bytedance.ailab.lab.Challenge
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
import com.bytedance.ailab.render.MeshData
import com.bytedance.ailab.render.MeshGen
import com.bytedance.ailab.render.OptGeo
import com.bytedance.ailab.render.ProcTex
import com.bytedance.ailab.render.Rgb
import com.pico.spatial.ui.design.PicoTheme
import com.pico.spatial.ui.design.Text
import java.io.File
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tanh

object BackpropDemo : SpecDemo() {
    override val id = "backprop"
    override val title = "反向传播 · 链式法则"  // i18n-ok (zh half of DemoRegistry.titleOf; English in TITLE_EN)
    override val credit = "Rumelhart, Hinton & Williams · 1986"

    // ---------------------------------------------------------------- the live model
    private val trainer = Trainer()
    private val net: Net get() = trainer.net

    // ---------------------------------------------------------------- layout (volume metres)
    private val PC = P3(-0.43f, -0.05f, 0.02f)          // input-plane centre
    private const val TILT = 0.663f                       // 38 deg: a lectern facing the viewer
    private const val HALF = 0.25f                        // input range [-1, 1] -> +-0.25 m
    private const val HGT = 0.10f                         // p = 1 -> 10 cm above the plane
    private const val G = 29                              // sheet vertices per side
    private const val BEAD_R = 0.0095f
    private val COL_X = floatArrayOf(0.06f, 0.30f, 0.54f, 0.78f)
    private const val NET_Y = 0.15f
    private const val NET_Z = -0.07f
    private const val NEURON_R = 0.019f
    private const val PLINTH_Y = -0.262f
    private val SHELF_X = floatArrayOf(0.22f, 0.39f, 0.56f)
    private const val SHELF_Y = -0.188f
    private const val SHELF_Z = 0.30f
    private val U = P3(1f, 0f, 0f)
    private val V = P3(0f, sin(TILT), -cos(TILT))
    private val N = P3(0f, cos(TILT), sin(TILT))
    /** Input plane (x1, x2 in [-1, 1], height h in metres) -> volume space. */
    private fun planeW(x1: Float, x2: Float, h: Float) = PC + U * (x1 * HALF) + V * (x2 * HALF) + N * h

    // ---------------------------------------------------------------- palette (sRGB), MLP family
    private val POS = Rgb(0.35f, 0.77f, 0.87f)
    private val NEG = Rgb(0.99f, 0.38f, 0.33f)
    private val GOLD = Rgb(1.00f, 0.78f, 0.22f)
    private val MAG = Rgb(1.00f, 0.22f, 0.86f)
    private val N_OFF = Rgb(0.10f, 0.12f, 0.17f)
    private val CYAN = Rgb(0.45f, 0.80f, 1.00f)
    private const val LV = 8
    private val wMats = List(2) { s -> List(LV + 4) { k -> Mat.Unlit(sc(if (s == 0) POS else NEG, 0.34f + 0.66f * k / (LV + 3f)), blend = Blend.ADD) } }
    private val gMats = List(LV) { k -> Mat.Unlit(sc(MAG, 0.30f + 0.70f * (k + 1f) / LV), blend = Blend.ADD) }
    private val gTop = Mat.Unlit(GOLD, blend = Blend.ADD)
    private val ghostMats = List(5) { k -> Mat.Unlit(sc(Rgb(0.9f, 0.95f, 1f), 0.08f * k), blend = Blend.ADD) }
    private val nMats = List(2) { s -> List(LV + 1) { k -> Mat.Unlit(ProcTex.okLerp(N_OFF, if (s == 0) Rgb(0.80f, 0.95f, 1f) else Rgb(1f, 0.62f, 0.55f), k / LV.toFloat())) } }
    private val nGlow = List(2) { s -> List(LV + 1) { k -> Mat.Unlit(sc(if (s == 0) POS else NEG, 0.9f * k / LV), tex = BpGfx.softDisc, blend = Blend.ADD) } }
    private val dMats = List(LV + 1) { k -> Mat.Unlit(sc(MAG, 0.25f + 0.75f * k / LV), blend = Blend.ADD) }
    private val beadMats = listOf(Mat.Pbr(BpGfx.CLS0, roughness = 0.35f, emissive = sc(BpGfx.CLS0, 0.85f)), Mat.Pbr(BpGfx.CLS1, roughness = 0.35f, emissive = sc(BpGfx.CLS1, 0.85f)))
    private val beadCurMat = Mat.Pbr(Rgb(1f, 1f, 1f), roughness = 0.3f, emissive = Rgb(0.9f, 0.9f, 0.9f))
    private val linkMats = List(4) { k -> Mat.Unlit(sc(CYAN, 0.20f + 0.25f * k), blend = Blend.ADD) }
    private val slabTex by lazy { ProcTex.bakePlanet(256, 128, Rgb(0.06f, 0.065f, 0.08f), Rgb(0.10f, 0.105f, 0.125f), Rgb(0.16f, 0.17f, 0.19f), 0f, 91) }
    private val alu = Mat.Pbr(Rgb(0.78f, 0.80f, 0.84f), metallic = 1f, roughness = 0.30f)
    private val blackGlass = Mat.Pbr(Rgb(0.02f, 0.024f, 0.035f), metallic = 0f, roughness = 0.12f)
    private val rimMat = Mat.Unlit(sc(CYAN, 0.55f), blend = Blend.ADD)
    private val sheetTex by lazy { BpGfx.sheetTex() }
    private val gridTex by lazy { BpGfx.gridTex() }

    // ---------------------------------------------------------------- user-facing state
    var kindIdx = 0
    var wide = false
    var lrIdx = 1
    var speedIdx = 1
    var showGrad = true
    var training = false; private set
    private val SPEED_LABELS get() = listOf(tr("慢", "Slow"), tr("中", "Medium"), tr("快", "Fast"))
    private val ANIM_S = floatArrayOf(4.5f, 2.6f, 1.4f)
    private val KIND_LABELS get() = listOf(tr("XOR", "XOR"), tr("圆环", "Circles"), tr("螺旋", "Spirals"))

    // ---------------------------------------------------------------- dynamic state
    private var cur = 0                    // current sample
    private var selCol = -1; private var selIdx = 0   // picked neuron (column 1..3)
    private var animT = -1f                // single-step animation 0..1 (< 0 = idle)
    private var applied = false
    private var lastStepShown = false      // a single step's gradient is on display
    private var stepAcc = 0f
    private var sheetTimer = 0f; private var evalTimer = 0f; private var probeTimer = 0f
    private var sheetDirty = true
    private var time = 0f
    private var introDone = false
    private var beat = 0f
    private val oldW = Array(3) { DoubleArray(64) }
    private var view: Net.Trace? = null   // what the nodes show (step trace while animating, else a probe)
    private val pGrid = FloatArray(G * G)
    private val beadP = FloatArray(2 * Datasets.PER_CLASS)

    // lesson counters
    private var stepsDone = 0
    private var neuronPicks = 0

    // ---------------------------------------------------------------- nodes
    private lateinit var rig: Node
    private lateinit var strips: Array<MeshData>
    private val stripRows = arrayOf(0..7, 7..14, 14..21, 21..28)
    private lateinit var beads: Array<Node>
    private lateinit var curNode: Node
    private lateinit var errBar: Node
    private lateinit var tgtRing: Node
    private lateinit var links: Array<Node>
    private lateinit var neurons: Array<Array<Node>>
    private lateinit var glows: Array<Array<Node>>
    private lateinit var dRings: Array<Array<Node>>
    private lateinit var selRing: Node
    private lateinit var wThr: Array<Node>
    private lateinit var gThr: Array<Node>
    private lateinit var shelf: Array<Node>
    private lateinit var shelfFrames: Array<Node>
    private val NT = 2 * 8 + 8 * 8 + 8
    private val thrA = Array(NT) { P3(0f, 0f, 0f) }
    private val thrB = Array(NT) { P3(0f, 0f, 0f) }
    private val thrRot = Array(NT) { Q.identity() }
    private val wShownR = FloatArray(NT); private val wShownM = IntArray(NT)
    private val gShown = FloatArray(NT); private val gShownR = FloatArray(NT); private val gShownM = IntArray(NT)
    private var nUsed = 0

    private fun sc(c: Rgb, k: Float) = Rgb(c.r * k, c.g * k, c.b * k)

    // ================================================================ build

    override fun buildSpec(cacheDir: File): Node {
        resetState()
        val r = Node("BackpropRoot")
        val cyl = MeshGen.cylinder(1f, 1f, 6)
        val cyl12 = MeshGen.cylinder(1f, 1f, 12)
        val sphere = MeshGen.sphere(1f, 10, 18, MeshGen.SphereUv.EQUIRECT)
        val disc = OptGeo.quad(1f, 1f)
        val ringMesh = MeshGen.annulus(0.80f, 1.0f, 40)
        val toZ = Q.axisAngle(1f, 0f, 0f, (PI / 2).toFloat())

        // ---- plinth
        r.add("Plinth", MeshGen.box(1.76f, 0.03f, 0.70f), Mat.Pbr(Rgb(1f, 1f, 1f), roughness = 0.55f, tex = slabTex), P3(0f, PLINTH_Y, 0.03f))
        r.add("PlinthBand", MeshGen.box(1.77f, 0.008f, 0.71f), alu, P3(0f, PLINTH_Y, 0.03f))

        // ---- the input plane on a lectern
        rig = r.add(Node("PlaneRig")).also { it.pos = PC; it.rot = Q.axisAngle(1f, 0f, 0f, TILT - (PI / 2).toFloat()) }
        val side = 2 * HALF
        rig.add("PlaneBack", MeshGen.box(side + 0.03f, side + 0.03f, 0.010f), blackGlass, P3(0f, 0f, -0.007f))
        rig.add("PlaneGrid", OptGeo.quad(side, side), Mat.Unlit(sc(CYAN, 0.30f), tex = gridTex, blend = Blend.ADD), P3(0f, 0f, 0.0005f))
        frameBars(rig, side + 0.03f, side + 0.03f, cyl12, alu)
        rig.add("PlaneRim", MeshGen.box(side + 0.04f, 0.003f, 0.003f), rimMat, P3(0f, -HALF - 0.022f, 0.004f))
        for (x in listOf(-0.62f, -0.24f)) {      // two posts from the back edge down to the plinth + a front ledge
            val top = planeW(if (x < -0.4f) -0.76f else 0.76f, 1f, -0.01f)
            stick(r, "LecternPost", cyl12, P3(top.x, PLINTH_Y + 0.015f, top.z), top, 0.006f, alu)
        }
        val fl = planeW(0f, -1.06f, -0.01f)
        r.add("Ledge", MeshGen.box(side + 0.05f, 0.012f, 0.02f), alu, P3(fl.x, fl.y, fl.z)).also { it.rot = rig.rot }

        strips = Array(4) { s ->
            val rows = stripRows[s]; val md = OptGeo.grid(rows.last - rows.first + 1, G)
            rig.add("Sheet$s", md, Mat.Unlit(Rgb(1f, 1f, 1f), tex = sheetTex)); md
        }
        beads = Array(2 * Datasets.PER_CLASS) { i ->
            rig.add("Pt_$i", sphere, beadMats[0]).also { it.uniformScale(BEAD_R); it.tapRadius = 0.045f / BEAD_R }
        }
        curNode = rig.add(Node("Cur")).also { it.billboard = true; it.tapRadius = 0.05f }
        curNode.add("CurRing", ringMesh, Mat.Unlit(GOLD, blend = Blend.ADD)).also { it.rot = toZ; it.uniformScale(0.020f) }
        errBar = rig.add("ErrBar", cyl12, Mat.Unlit(MAG, blend = Blend.ADD)).also { it.rot = toZ }
        tgtRing = rig.add("Target", ringMesh, Mat.Unlit(sc(MAG, 0.9f), blend = Blend.ADD)).also { it.rot = toZ; it.uniformScale(0.016f) }
        links = Array(2) { k -> r.add("Link$k", cyl, linkMats[0]) }

        // ---- the network: four slim walls (MLP family) + neurons + threads
        for (c in 0 until 4) {
            val g = r.add(Node("Col$c")).also { it.pos = P3(COL_X[c], NET_Y, NET_Z - 0.03f) }
            g.add("ColBack$c", MeshGen.box(0.075f, 0.50f, 0.008f), blackGlass, P3(0f, 0f, -0.004f))
            frameBars(g, 0.075f, 0.50f, cyl12, alu)
            g.add("ColRim$c", MeshGen.box(0.08f, 0.0025f, 0.002f), rimMat, P3(0f, -0.256f, 0.004f))
            val bottom = NET_Y - 0.25f
            g.add("ColPost$c", cyl12, alu, P3(0f, (bottom + PLINTH_Y) / 2 - NET_Y, -0.012f)).also { it.scale = P3(0.006f, bottom - PLINTH_Y, 0.006f) }
        }
        wThr = Array(NT) { k -> r.add("W$k", cyl, wMats[0][1]).also { it.visible = false } }
        gThr = Array(NT) { k -> r.add("G$k", cyl, gMats[0]).also { it.visible = false } }
        neurons = Array(4) { c -> Array(if (c == 0) 2 else if (c == 3) 1 else 8) { j ->
            r.add("N_${c}_$j", sphere, nMats[0][0]).also { it.uniformScale(NEURON_R); if (c > 0) it.tapRadius = 0.045f / NEURON_R }
        } }
        glows = Array(4) { c -> Array(neurons[c].size) { j -> r.add("Glow_${c}_$j", disc, nGlow[0][0]).also { it.billboard = true; it.uniformScale(0.07f); it.visible = false } } }
        dRings = Array(4) { c -> Array(neurons[c].size) { j ->
            r.add(Node("DRing_${c}_$j")).also { it.billboard = true; it.visible = false }.also { g -> g.add("DRingM_${c}_$j", ringMesh, dMats[LV]).also { it.rot = toZ } }
        } }
        selRing = r.add(Node("SelRing")).also { it.billboard = true; it.visible = false }
        selRing.add("SelRingM", ringMesh, Mat.Unlit(GOLD, blend = Blend.ADD)).also { it.rot = toZ; it.uniformScale(0.030f) }

        // ---- dataset shelf (front right)
        shelfFrames = Array(3) { Node("x") }
        shelf = Array(3) { k ->
            val t = r.add(Node("Shelf_$k")).also { it.pos = P3(SHELF_X[k], SHELF_Y, SHELF_Z); it.rot = Q.axisAngle(1f, 0f, 0f, -0.35f); it.tapRadius = 0.05f }
            t.add("ShelfBack$k", MeshGen.box(0.115f, 0.115f, 0.008f), blackGlass, P3(0f, 0f, -0.005f))
            t.add("ShelfPic$k", OptGeo.quad(0.10f, 0.10f), Mat.Unlit(Rgb(1f, 1f, 1f), tex = BpGfx.dataTex(Datasets.make(DataKind.values()[k]))), P3(0f, 0f, 0.0005f))
            shelfFrames[k] = Node("ShelfFrame$k").also { t.add(it) }
            frameBars(shelfFrames[k], 0.115f, 0.115f, cyl12, alu)
            t
        }
        r.add("ShelfBase", MeshGen.box(0.56f, 0.012f, 0.07f), alu, P3(0.39f, PLINTH_Y + 0.02f, SHELF_Z))

        layoutNetwork()
        sheetDirty = true; refreshSheet()
        refreshView()
        update(0f)
        return r
    }

    private fun frameBars(g: Node, w: Float, h: Float, cyl: MeshData, m: Mat) {
        val rr = 0.0032f; val zr = Q.axisAngle(0f, 0f, 1f, (PI / 2).toFloat()); val b = BpGfx.Merge()
        b.add(cyl, P3(0f, h / 2, 0.002f), zr, P3(rr, w + 2 * rr, rr)); b.add(cyl, P3(0f, -h / 2, 0.002f), zr, P3(rr, w + 2 * rr, rr))
        b.add(cyl, P3(-w / 2, 0f, 0.002f), Q.identity(), P3(rr, h, rr)); b.add(cyl, P3(w / 2, 0f, 0.002f), Q.identity(), P3(rr, h, rr))
        g.add("Frame", b.build(), m)
    }

    private fun stick(r: Node, name: String, cyl: MeshData, a: P3, b: P3, rad: Float, m: Mat): Node {
        val d = b - a
        return r.add(name, cyl, m, (a + b) * 0.5f).also { it.scale = P3(rad, d.length, rad); it.rot = Q.fromTo(0f, 1f, 0f, d.x, d.y, d.z) }
    }

    // ================================================================ state

    private fun resetState() {
        kindIdx = 0; wide = false; lrIdx = 1; speedIdx = 1; showGrad = true; training = false
        trainer.reset(DataKind.XOR, Trainer.NARROW, Trainer.SEED); trainer.lr = Trainer.LRS[lrIdx]
        cur = trainer.worstSample(); selCol = -1; selIdx = 0
        animT = -1f; applied = false; lastStepShown = false; stepAcc = 0f
        sheetTimer = 0f; evalTimer = 0f; probeTimer = 0f; sheetDirty = true; time = 0f; introDone = false; beat = 0f
        stepsDone = 0; neuronPicks = 0; view = null
        wShownR.fill(-1f); wShownM.fill(-1); gShown.fill(-1f); gShownR.fill(-1f); gShownM.fill(-1)
        labels.clear(); readoutText = ""
    }

    private val sizes get() = trainer.sizes
    private fun colSize(c: Int) = sizes[c]

    private fun neuronPos(c: Int, j: Int): P3 {
        val n = colSize(c)
        val span = when (n) { 1 -> 0f; 2 -> 0.14f; 4 -> 0.27f; else -> 0.42f }
        val y = if (n == 1) NET_Y else NET_Y + span / 2 - span * j / (n - 1)
        return P3(COL_X[c], y, NET_Z)
    }

    private fun wIndex(l: Int, j: Int, i: Int): Int {
        val w = sizes[1]
        val off = when (l) { 0 -> 0; 1 -> 2 * w; else -> 2 * w + w * w }
        return off + j * sizes[l] + i
    }

    /** Positions of neurons and threads for the current width (transform only; on width change). */
    private fun layoutNetwork() {
        for (c in 0 until 4) for (j in neurons[c].indices) {
            val on = j < colSize(c)
            neurons[c][j].visible = on; if (c > 0) neurons[c][j].tapRadius = if (on) 0.045f / NEURON_R else 0f
            if (on) { val p = neuronPos(c, j); neurons[c][j].pos = p; glows[c][j].pos = P3(p.x, p.y, p.z + 0.004f); dRings[c][j].pos = P3(p.x, p.y, p.z + 0.006f) }
            else { glows[c][j].visible = false; dRings[c][j].visible = false }
        }
        nUsed = 0
        for (l in 0 until 3) for (j in 0 until sizes[l + 1]) for (i in 0 until sizes[l]) {
            val k = wIndex(l, j, i); nUsed = max(nUsed, k + 1)
            val a = neuronPos(l, i); val b = neuronPos(l + 1, j)
            thrA[k] = a; thrB[k] = b; val d = b - a
            thrRot[k] = Q.fromTo(0f, 1f, 0f, d.x, d.y, d.z)
            wThr[k].rot = thrRot[k]; gThr[k].rot = thrRot[k]
            wThr[k].pos = (a + b) * 0.5f
        }
        for (k in 0 until NT) { wThr[k].visible = k < nUsed; if (k >= nUsed) gThr[k].visible = false }
        wShownR.fill(-1f); wShownM.fill(-1); gShown.fill(-1f); gShownR.fill(-1f); gShownM.fill(-1)
    }

    // ---------------------------------------------------------------- user actions (the panel and pinches call these)

    fun setDataset(k: Int) {
        finishAnim()
        kindIdx = k.coerceIn(0, 2); training = false
        resetNet(Trainer.SEED)
    }

    fun setWidth(w8: Boolean) { finishAnim(); wide = w8; training = false; resetNet(Trainer.SEED) }

    fun resetWeights() { finishAnim(); training = false; resetNet(trainer.seed + 1) }

    private fun resetNet(seed: Long) {
        trainer.reset(DataKind.values()[kindIdx], if (wide) Trainer.WIDE else Trainer.NARROW, seed)
        trainer.lr = Trainer.LRS[lrIdx]
        cur = trainer.worstSample(); selCol = -1; lastStepShown = false
        layoutNetwork(); sheetDirty = true; refreshSheet(); refreshView()
    }

    fun setLr(k: Int) { lrIdx = k.coerceIn(0, 2); trainer.lr = Trainer.LRS[lrIdx] }

    fun runTraining(on: Boolean) {
        finishAnim()
        training = on; stepAcc = 0f; lastStepShown = false
    }

    /** One animated SGD step on sample [i]: forward, error, backward, update. */
    fun stepSample(i: Int) {
        finishAnim(); training = false
        cur = i.coerceIn(0, trainer.data.n - 1)
        trainer.probe(cur)
        for (l in 0 until 3) net.w[l].copyInto(oldW[l])
        view = trainer.trace
        animT = 0f; applied = false; lastStepShown = true
        gShown.fill(-1f)
    }

    private fun finishAnim() {
        if (animT < 0f) return
        if (!applied) applyNow()
        animT = -1f
    }

    private fun applyNow() {
        trainer.applyStep(trainer.trace); applied = true; stepsDone++
        sheetDirty = true; refreshSheet(); trainer.evaluate()
    }

    fun pickNeuron(c: Int, j: Int) {
        if (selCol == c && selIdx == j) { selCol = -1; return }
        selCol = c; selIdx = j; neuronPicks++
    }

    // ================================================================ per frame

    override fun update(dt: Float) {
        time += dt
        if (!introDone && time >= 0.3f) { introDone = true; stepSample(cur) }

        if (training) {
            stepAcc += Trainer.STEPS_PER_S[speedIdx] * dt
            val n = min(stepAcc.toInt(), 120)
            stepAcc -= n
            if (n > 0) { trainer.trainSteps(n); sheetDirty = true }
            evalTimer += dt
            if (evalTimer >= 0.1f) { evalTimer = 0f; trainer.evaluate() }
        }
        sheetTimer += dt
        if (sheetDirty && (sheetTimer >= 0.2f || !training)) refreshSheet()

        if (animT >= 0f) {
            animT += dt / ANIM_S[speedIdx]
            if (animT >= 0.85f && !applied) applyNow()
            if (animT >= 1f) animT = -1f
        }
        probeTimer += dt
        if (animT < 0f && (probeTimer >= 0.1f || view == null)) { probeTimer = 0f; refreshView() }

        beat = (beat + dt) % 1.6f
        applyThreads()
        applyNeurons()
        applySample()
        applyShelf()
        updateLabels()
        updateReadout()
    }

    /** Re-evaluate the decision surface and the bead heights (only when the weights changed). */
    private fun refreshSheet() {
        if (!sheetDirty) return
        sheetDirty = false; sheetTimer = 0f
        for (r in 0 until G) for (c in 0 until G) {
            val x1 = -1.0 + 2.0 * c / (G - 1); val x2 = 1.0 - 2.0 * r / (G - 1)
            pGrid[r * G + c] = trainer.predict(x1, x2).toFloat()
        }
        val d = 2f * HALF / (G - 1)
        // light from upper left, toward the viewer (local frame of the plane)
        val lx = -0.40f; val ly = 0.50f; val lz = 0.75f; val ll = sqrt(lx * lx + ly * ly + lz * lz)
        for (s in 0 until 4) {
            val rows = stripRows[s]; val md = strips[s]
            for ((k, r) in rows.withIndex()) for (c in 0 until G) {
                val i = k * G + c; val p = pGrid[r * G + c]
                val hx = (pGrid[r * G + min(c + 1, G - 1)] - pGrid[r * G + max(c - 1, 0)]) * HGT / (d * (min(c + 1, G - 1) - max(c - 1, 0)))
                val hy = (pGrid[max(r - 1, 0) * G + c] - pGrid[min(r + 1, G - 1) * G + c]) * HGT / (d * (min(r + 1, G - 1) - max(r - 1, 0)))
                val nl = sqrt(hx * hx + hy * hy + 1f)
                val lam = ((-hx * lx - hy * ly + lz) / (nl * ll)).coerceAtLeast(0f)
                val shade = (0.28f + 0.72f * lam / 0.76f).coerceIn(0f, 1f)
                OptGeo.setPos(md, i, -HALF + c * d, HALF - r * d, p * HGT + 0.003f)
                OptGeo.setUv(md, i, OptGeo.rampU(p), BpGfx.shadeV(shade))
            }
            meshEdited(md)
        }
        val data = trainer.data
        for (i in 0 until data.n) {
            val p = trainer.predict(data.x1(i), data.x2(i)).toFloat(); beadP[i] = p
            beads[i].pos = P3(data.x1(i).toFloat() * HALF, data.x2(i).toFloat() * HALF, p * HGT + BEAD_R + 0.003f)
            beads[i].setMat(if (i == cur) beadCurMat else beadMats[if (data.y[i] > 0.5) 1 else 0])
            beads[i].visible = true
        }
        for (i in data.n until beads.size) { beads[i].visible = false; beads[i].tapRadius = 0f }
    }

    /** Forward + backward on the current sample under the current weights (what the idle network shows). */
    private fun refreshView() { view = trainer.probe(cur, trainer.probeTrace) }

    // ---------------------------------------------------------------- threads

    private fun wRadius(w: Double) = 0.0009f + 0.0032f * tanh(abs(w) / 1.5).toFloat()

    private fun applyThreads() {
        val t = view ?: return
        val a = animT
        val fwd = if (a in 0f..0.40f) a / 0.40f else -1f
        val bwd = when { a < 0f -> 1f; a < 0.5f -> -1f; a < 0.85f -> (a - 0.5f) / 0.35f; else -> 1f }
        val ghostK = if (a >= 0.85f) 1f - (a - 0.85f) / 0.15f else 0f
        // gradient scale: one step = relative to its biggest gradient; training = absolute (they fade as it learns)
        val stepMode = animT >= 0f || (lastStepShown && !training)
        var gMax = 1e-12
        if (stepMode) for (l in 0 until 3) for (g in t.gw[l]) gMax = max(gMax, abs(g))
        val top = topGradient(t)
        val trainBeat = training && showGrad
        for (l in 0 until 3) {
            val nIn = sizes[l]
            for (j in 0 until sizes[l + 1]) for (i in 0 until nIn) {
                val k = wIndex(l, j, i)
                // ---- weight thread: thickness |w|, colour sign, brightness |w| (+ the forward pulse)
                val w = net.w[l][j * nIn + i]
                val rad = wRadius(w)
                var lv = (tanh(abs(w) / 1.5) * (LV - 1)).toInt()
                if (fwd >= 0f && fwd * 3f >= l && fwd * 3f < l + 1) lv += 4
                if (trainBeat && beat < 0.8f && ((beat / 0.8f) * 3f).toInt() == l) lv += 2
                val m = (if (w >= 0) 0 else 1) * 100 + lv
                val n = wThr[k]
                if (abs(rad - wShownR[k]) > 0.00004f) { wShownR[k] = rad; n.scale = P3(rad, (thrB[k] - thrA[k]).length, rad) }
                if (m != wShownM[k]) { wShownM[k] = m; n.setMat(wMats[if (w >= 0) 0 else 1][min(lv, LV + 3)]) }
                // ---- gradient sleeve (magenta, grows back from the output end) or the ghost of the old weight
                val gn = gThr[k]
                var q: Float; var gr: Float; var gm: Mat
                if (ghostK > 0f) {
                    q = 1f; gr = wRadius(oldW[l][j * nIn + i]) + 0.0009f; gm = ghostMats[(ghostK * 4.99f).toInt().coerceIn(0, 4)]
                    if (ghostK * 4.99f < 1f) q = 0f
                } else if (!showGrad && animT < 0f) {
                    q = 0f; gr = 0f; gm = gMats[0]
                } else {
                    val g = abs(t.gw[l][j * nIn + i])
                    val rel = if (stepMode) (g / gMax).toFloat() else tanh(trainer.gAvg[l][j * nIn + i] / 0.12).toFloat()
                    q = if (bwd < 0f) 0f else ((bwd * 3f - (2 - l)).coerceIn(0f, 1f))
                    if (!stepMode && !training) q = if (showGrad) 1f else 0f
                    gr = if (stepMode) 0.0010f + 0.0050f * rel else 0.0008f + 0.0034f * rel
                    var glv = (rel * (LV - 0.01f)).toInt().coerceIn(0, LV - 1)
                    if (trainBeat && beat >= 0.8f && (((beat - 0.8f) / 0.8f) * 3f).toInt() == 2 - l) glv = min(LV - 1, glv + 2)
                    gm = if (stepMode && k == top) gTop else gMats[glv]
                    if (rel < 0.02f) q = 0f
                }
                val qq = (q * 30f).toInt() / 30f
                if (qq <= 0f) { if (gn.visible) gn.visible = false; gShown[k] = 0f; continue }
                gn.visible = true
                if (qq != gShown[k] || abs(gr - gShownR[k]) > 0.00004f) {
                    gShown[k] = qq; gShownR[k] = gr
                    val A = thrA[k]; val B = thrB[k]; val d = B - A; val L = d.length; val u = d * (1f / L)
                    gn.scale = P3(gr, L * qq, gr)
                    gn.pos = B - u * (L * qq / 2f)
                }
                gn.setMat(gm)
            }
        }
    }

    /** Index of the weight with the largest |dL/dw| in [t] (-1 if none). */
    private fun topGradient(t: Net.Trace): Int {
        var best = -1; var bv = 0.0
        for (l in 0 until 3) for (j in 0 until sizes[l + 1]) for (i in 0 until sizes[l]) {
            val g = abs(t.gw[l][j * sizes[l] + i]); if (g > bv) { bv = g; best = wIndex(l, j, i) }
        }
        return best
    }

    // ---------------------------------------------------------------- neurons

    private fun neuronVal(t: Net.Trace, c: Int, j: Int): Double = if (c == 3) 2 * t.a[3][0] - 1 else t.a[c][j]

    private fun applyNeurons() {
        val t = view ?: return
        val a = animT
        val fwd = if (a < 0f) 1f else (a / 0.40f).coerceIn(0f, 1f)
        val bwd = when { a < 0f -> if (showGrad) 1f else -1f; a < 0.4f -> -1f; a < 0.5f -> 0f; else -> ((a - 0.5f) / 0.35f).coerceIn(0f, 1f) }
        var dMax = 1e-12
        for (c in 1..3) for (j in 0 until colSize(c)) dMax = max(dMax, abs(t.delta[c][j]))
        for (c in 0 until 4) for (j in 0 until colSize(c)) {
            val lit = fwd * 3f >= c - 0.001f
            val v = neuronVal(t, c, j)
            val s = if (v >= 0) 0 else 1
            val lv = if (lit) (abs(v).coerceIn(0.0, 1.0) * LV).toInt() else 0
            neurons[c][j].setMat(nMats[s][lv])
            glows[c][j].visible = lit && lv >= 2
            if (glows[c][j].visible) glows[c][j].setMat(nGlow[s][lv])
            // blame ring: appears when the backward wave reaches this column
            val dr = dRings[c][j]
            val on = c > 0 && bwd >= 0f && bwd * 3f >= 3 - c - 0.001f
            dr.visible = on
            if (on) {
                val rel = sqrt(abs(t.delta[c][j]) / dMax).toFloat()
                dr.uniformScale(0.020f + 0.020f * rel)
                dr.children[0].setMat(dMats[(rel * LV).toInt().coerceIn(1, LV)])
            }
        }
        selRing.visible = selCol > 0 && selIdx < colSize(selCol)
        if (selRing.visible) { val p = neuronPos(selCol, selIdx); selRing.pos = P3(p.x, p.y, p.z + 0.008f) }
    }

    // ---------------------------------------------------------------- current sample, error bar, links

    private fun applySample() {
        val d = trainer.data
        val x1 = d.x1(cur).toFloat(); val x2 = d.x2(cur).toFloat(); val y = d.y[cur].toFloat()
        val p = (if (animT >= 0f && !applied) trainer.trace.output else beadP[cur].toDouble()).toFloat()
        curNode.pos = P3(x1 * HALF, x2 * HALF, beadP[cur] * HGT + BEAD_R + 0.003f)
        val showErr = animT < 0f || animT >= 0.40f
        val a = p * HGT + 0.003f; val b = y * HGT + 0.003f
        errBar.visible = showErr && abs(b - a) > 0.002f
        val pulse = if (animT in 0.40f..0.5f) 1.6f else 1f
        errBar.pos = P3(x1 * HALF, x2 * HALF, (a + b) / 2f); errBar.scale = P3(0.0032f * pulse, max(abs(b - a), 0.001f), 0.0032f * pulse)
        tgtRing.visible = errBar.visible
        tgtRing.pos = P3(x1 * HALF, x2 * HALF, b)
        // the sample enters the network: x1 -> input neuron 0, x2 -> input neuron 1
        val from = planeW(x1, x2, beadP[cur] * HGT + BEAD_R)
        val lv = if (animT in 0f..0.15f) 3 else 1
        for (k in 0..1) {
            val to = neuronPos(0, k); val dd = to - from
            links[k].pos = (from + to) * 0.5f; links[k].scale = P3(0.0011f, dd.length, 0.0011f)
            links[k].rot = Q.fromTo(0f, 1f, 0f, dd.x, dd.y, dd.z); links[k].setMat(linkMats[lv])
        }
        for (i in 0 until d.n) beads[i].setMat(if (i == cur) beadCurMat else beadMats[if (d.y[i] > 0.5) 1 else 0])
    }

    private fun applyShelf() {
        for (k in 0 until 3) {
            shelf[k].pos = P3(SHELF_X[k], SHELF_Y + if (k == kindIdx) 0.015f else 0f, SHELF_Z)
            shelfFrames[k].children[0].setMat(if (k == kindIdx) Mat.Unlit(GOLD) else alu)
        }
    }

    // ================================================================ interaction

    override fun isTarget(name: String) = name.startsWith("Pt_") || name == "Cur" || name.startsWith("N_") || name.startsWith("Shelf_")

    override val tapTargets get() = listOf(
        TapTarget("Cur", "训练点 → 单步", "Point → one step"),
        TapTarget(nextNeuronName(), "神经元 → 看 δ", "Neuron → its δ") { selCol > 0 },
        TapTarget("Shelf_${(kindIdx + 1) % 3}", "数据架 → 换数据", "Shelf → next dataset"),
    )

    private fun nextNeuronName(): String {
        val order = ArrayList<Pair<Int, Int>>()
        for (c in 1..3) for (j in 0 until colSize(c)) order += c to j
        val k = order.indexOf(selCol to selIdx)
        val n = order[(k + 1) % order.size]
        return "N_${n.first}_${n.second}"
    }

    override fun onTap(name: String) {
        when {
            name == "Cur" -> stepSample(cur)
            name.startsWith("Pt_") -> name.removePrefix("Pt_").toIntOrNull()?.let { stepSample(it) }
            name.startsWith("N_") -> name.removePrefix("N_").split("_").mapNotNull { it.toIntOrNull() }.let { if (it.size == 2 && it[0] in 1..3) pickNeuron(it[0], it[1].coerceIn(0, colSize(it[0]) - 1)) }
            name.startsWith("Shelf_") -> name.removePrefix("Shelf_").toIntOrNull()?.let { setDataset(it) }
        }
    }

    override fun tapMessage(name: String): String? = when {
        name == "Cur" || name.startsWith("Pt_") -> tr("单步：先看白光前向，再看品红误差往回传", "One step: white light forward, then magenta blame flows back")
        name.startsWith("N_") -> if (selCol > 0) tr("%s 的 δ = f′(z) · Σ w·δ —— 数字在面板上", "%s: δ = f′(z) · Σ w·δ — numbers on the panel").format(neuronName(selCol, selIdx))
                                 else tr("已取消选中神经元", "Neuron deselected")
        name.startsWith("Shelf_") -> tr("数据集：%s —— 权重已重置，按「训练」", "Dataset: %s — weights reset, press Train").format(KIND_LABELS[kindIdx])
        else -> null
    }

    private fun neuronName(c: Int, j: Int) = when (c) { 0 -> "x${if (j == 0) "₁" else "₂"}"; 3 -> tr("输出", "out"); else -> "H$c·${j + 1}" }

    // ================================================================ readout + labels

    private fun f(x: Double, d: Int = 2) = "%.${d}f".format(x).replace("-", "−")
    private fun pct(x: Double) = "${(x * 100 + 0.5).toInt()}%"

    private fun updateReadout() {
        val w = sizes[1]
        val l1 = tr("%s · 2→%d→%d→1 · η = %s · 第 %d 轮 · 损失 %s · 准确率 %s · %s", "%s · 2→%d→%d→1 · η = %s · epoch %d · loss %s · accuracy %s · %s")
            .format(KIND_LABELS[kindIdx], w, w, Trainer.LRS[lrIdx].toString(), trainer.epoch, f(trainer.loss, 3), pct(trainer.accuracy),
                if (training) tr("训练中", "training") else tr("暂停", "paused"))
        val t = view
        val l2 = when {
            t == null -> ""
            selCol == 3 -> tr("输出：δ = ∂L/∂p · σ′(z) = p − y = %s − %d = %s（σ + 交叉熵，斜率相消）", "Output: δ = ∂L/∂p · σ′(z) = p − y = %s − %d = %s (σ + cross-entropy: slopes cancel)")
                .format(f(t.a[3][0]), trainer.data.y[cur].toInt(), f(t.delta[3][0]))
            selCol in 1..2 -> tr("%s：δ = f′(z) · Σ w·δ下游 = (1 − a²) · Σ = %s × %s = %s", "%s: δ = f′(z) · Σ w·δ downstream = (1 − a²) · Σ = %s × %s = %s")
                .format(neuronName(selCol, selIdx), f(net.slope(t, selCol, selIdx)), f(net.blameSum(t, selCol, selIdx), 3), f(t.delta[selCol][selIdx], 3))
            else -> {
                val top = topGradient(t)
                var tl = 0; var tj = 0; var ti = 0
                for (l in 0 until 3) for (j in 0 until sizes[l + 1]) for (i in 0 until sizes[l]) if (wIndex(l, j, i) == top) { tl = l; tj = j; ti = i }
                val d = trainer.data
                tr("样本 (%s, %s) 标签 %d：p = %s，误差 p−y = %s · 最大梯度 %s→%s：a×δ = %s×%s = %s", "Sample (%s, %s) label %d: p = %s, error p−y = %s · top gradient %s→%s: a×δ = %s×%s = %s")
                    .format(f(d.x1(cur)), f(d.x2(cur)), d.y[cur].toInt(), f(t.a[3][0]), f(t.a[3][0] - d.y[cur]),
                        neuronName(tl, ti), neuronName(tl + 1, tj), f(t.a[tl][ti]), f(t.delta[tl + 1][tj]), f(t.gw[tl][tj * sizes[tl] + ti], 3))
            }
        }
        val l3 = when {
            animT in 0f..0.40f -> tr("① 前向：x₁, x₂ 一层层算到输出 p（白光）", "① Forward: x₁, x₂ are computed layer by layer up to p (white light)")
            animT in 0.40f..0.5f -> tr("误差 = 品红竖线：p 到目标高度的距离", "The error = the magenta bar: from p to the target height")
            animT in 0.5f..0.85f -> tr("② 反向：误差乘以权重和斜率，一层层往回传（品红越粗，∂L/∂w 越大）", "② Backward: the error, times weights and slopes, flows back layer by layer (thicker magenta = bigger ∂L/∂w)")
            animT >= 0.85f -> tr("③ 更新：每根连线 w ← w − η·∂L/∂w（白色虚影 = 旧值），曲面跟着弯", "③ Update: every weight w ← w − η·∂L/∂w (white ghost = old value); the surface bends")
            training && kindIdx == 2 && !wide -> tr("螺旋要弯很多次：试试把隐藏层加宽到 8", "Spirals need many bends: try a wider hidden layer (8)")
            training && trainer.accuracy >= 1.0 -> tr("全部分对：曲面在每个点的那一侧都对了，梯度（品红）随之变细", "All correct: every bead is on its side of the surface; the gradients (magenta) thin out")
            training -> tr("训练 = 单步重复成千上万次：每个点都把曲面往自己那边拉一点", "Training = the single step, thousands of times: each point pulls the surface its way a little")
            else -> tr("学习 = 责任往回流：输出的误差沿每根连线分摊回去，每根连线按自己的那份去改", "Learning = blame flowing backwards: the output error is shared back along every connection")
        }
        readoutText = "$l1\n$l2\n$l3"
    }

    private fun updateLabels() {
        labels.clear()
        val w = sizes[1]
        labels += LabelSpec(tr("输入平面 (x₁, x₂) · 高度/颜色 = p", "Input plane (x₁, x₂) · height/colour = p"), planeW(0f, 1.15f, 0.06f))
        labels += LabelSpec(tr("2 → %d → %d → 1   tanh · tanh · σ", "2 → %d → %d → 1   tanh · tanh · σ").format(w, w), P3(0.42f, 0.42f, NET_Z))
        val phase = when {
            animT in 0f..0.40f -> tr("① 前向", "① Forward")
            animT in 0.40f..0.5f -> tr("误差 p − y", "Error p − y")
            animT in 0.5f..0.85f -> tr("② 反向 · δ 往回传", "② Backward · δ flows back")
            animT >= 0.85f -> tr("③ 更新 w", "③ Update w")
            training -> tr("训练中 · 第 %d 轮", "Training · epoch %d").format(trainer.epoch)
            else -> tr("捏一个点 = 单步", "Pinch a point = one step")
        }
        labels += LabelSpec(phase, P3(0.42f, -0.115f, NET_Z + 0.06f))
        val d = trainer.data
        val err = beadP[cur] - d.y[cur]
        labels += LabelSpec(tr("p − y = %s", "p − y = %s").format(f(err.toDouble())), planeW(d.x1(cur).toFloat(), d.x2(cur).toFloat(), HGT + 0.05f))
        labels += LabelSpec(tr("数据架", "Datasets"), P3(0.39f, -0.255f, SHELF_Z + 0.08f))
    }

    // ================================================================ guided track (导学) + challenge
    // KEY INSIGHT: learning = blame flowing backwards; a weight changes by (activity at its input) x (blame at its output).

    private var baseSteps = 0
    private var basePicks = 0
    private var baseTrain = 0L

    // ---- background card (📖): context for a viewer who has never met this topic
    override val about = About(listOf(
        About.Page(
            Bi("这个实验在做什么？", "What is this experiment?"),
            Bi("左边讲台上的斜面是「输入平面」，上面的珠子是两类训练数据。空中悬着一个小网络：2 个输入 → 两层隐藏神经元 → 1 个输出。捏金圈里的点，网络就学一步：白光前向，品红的「误差」往回流，然后每根连线改一点。按「训练」，这一步重复成千上万次。右前方的数据架可换 XOR、圆环、螺旋。", "On the left lectern, the tilted “input plane” holds the training data as beads of two classes. In the air floats a small network: 2 inputs → two hidden layers → 1 output. Pinch the point in the gold ring and the network learns one step: white light forward, magenta “blame” flowing back, then every connection changes a little. Press Train to repeat it thousands of times. The shelf (front right) swaps XOR, circles and spirals."),
        ),
        About.Page(
            Bi("先认识几个词", "A few words first"),
            Bi("网络（模型）对每个点给出一个概率 p：它有多大把握这是第 1 类。正确答案叫「标签」y（0 或 1）；p 离 y 有多远，用「损失」打分。每根连线上的数叫「权重」，是要学的参数。「梯度」∂L/∂w 回答：这根权重稍微变大一点，损失会变大还是变小、变多少？知道了它，就知道往哪边改。", "The network (the model) gives each point a probability p: how sure it is that the point is class 1. The right answer is the “label” y (0 or 1); the “loss” scores how far p is from y. The number on each connection is a “weight”, a parameter to be learned. The “gradient” ∂L/∂w answers: if this weight grew a little, would the loss go up or down, and by how much? Knowing it tells you which way to turn."),
        ),
        About.Page(
            Bi("背景：1986 年的突破", "Background: the 1986 breakthrough"),
            Bi("多层网络早有人提出，却几十年训练不好：中间层的权重出了错，该怪谁、怪多少？1986 年，鲁梅尔哈特、辛顿和威廉姆斯在《自然》上展示：用微积分的「链式法则」把误差从输出一层层往回分摊，一次就能算出所有权重的梯度。这个方法此前也有人独立想到，但正是这篇论文让神经网络第一次真正学会了东西。", "Multi-layer networks had been proposed long before, but for decades nobody could train them well: when a middle-layer weight is wrong, whose fault is it, and how much? In 1986 Rumelhart, Hinton and Williams showed in Nature that the calculus “chain rule” shares the error back from the output, layer by layer, giving every weight's gradient in one pass. Others had found it earlier, but this paper made networks really learn."),
        ),
        About.Page(
            Bi("核心想法：往回追责", "Key idea: tracing the blame"),
            Bi("想象流水线做坏了一件产品，质检员从最后一道工序往回问：你这一步让结果偏了多少？每个工人的「责任」= 他自己的放大倍数 × 下游传回来的责任之和。每根连线该改多少 = 起点有多活跃 × 终点担多少责。一次反向就把全部算完，代价和前向差不多。", "Picture an assembly line that made a faulty product. The inspector walks back from the last station asking: how much did your step push the result off? Each worker's “blame” = their own amplification × the blame passed back from downstream. How much a connection should change = how active its start is × how much blame its end carries. One backward pass computes it all, at about the cost of the forward pass."),
        ),
        About.Page(
            Bi("怎么看这个 3D 场景", "How to read the 3-D scene"),
            Bi("输入平面上的曲面 = 网络在每一处给出的 p（高度和颜色），白线是 p = 0.5 的分界，珠子 = 训练点。空中的球 = 神经元（蓝正红负），连线 = 权重（越粗 |w| 越大，颜色 = 正负）。品红套筒越粗，这根权重的梯度越大；品红圆环 = 神经元的责任 δ；白色虚影 = 更新前的旧权重。", "The surface over the input plane = the network's p at every spot (height and colour); the white line is the p = 0.5 boundary; beads = training points. Spheres in the air = neurons (blue +, red −); threads = weights (thicker = larger |w|, colour = sign). A thicker magenta sleeve = a bigger gradient for that weight; magenta rings = a neuron's blame δ; white ghosts = the weights before the update."),
        ),
        About.Page(
            Bi("有多大？今天在哪里？", "How big? Where is it today?"),
            Bi("这里的网络最多只有 105 个参数（2→8→8→1），在你眼前实时训练。GPT-3 有 1750 亿个参数，但训练它用的仍是同一个反向传播，只是交给显卡以惊人的速度做乘加。PyTorch 等框架会自动完成这一步（「自动求导」）—— 今天几乎每个 AI 产品的训练都靠它。", "The network here has at most 105 parameters (2→8→8→1) and trains live in front of you. GPT-3 has 175 billion parameters, yet it was trained with the same backpropagation — just with graphics chips doing multiply-adds at enormous speed. Frameworks such as PyTorch do this step automatically (“autograd”); almost every AI product today is trained with it."),
        ),
    ))

    override val lesson: Lesson by lazy {
        Lesson(
            listOf(
                LessonStep(
                    say = Bi("先猜：误差从输出往回传时，哪根连线该改得最多？", "Guess first: as the error flows back from the output, which connection should change the most?"),
                    options = listOf(
                        Bi("最粗的（权重最大）", "The thickest (largest weight)"),
                        Bi("对误差贡献最大的", "Most to blame for the error"),
                        Bi("离输出最近的", "The one nearest the output"),
                        Bi("每根改得一样多", "All change equally"),
                    ),
                    answer = 1,
                ),
                LessonStep(
                    say = Bi("捏金圈里的训练点（或面板「训练点 → 单步」）：先看白光前向，再看品红反向。", "Pinch the point in the gold ring (or \"Point → one step\"): white light forward, then magenta backward."),
                    target = "Cur",
                    onEnter = { baseSteps = stepsDone },
                    done = { stepsDone > baseSteps && animT < 0f },
                    autoplay = { onTap("Cur") },
                    reveal = true,
                    insight = Bi("∂L/∂w = 起点的激活 a × 终点的责任 δ。起点越活跃、下游担责越多，这根改得越多 —— 金色那根；权重粗细本身不决定。", "∂L/∂w = activity a at its input × blame δ at its output: the gold one changes most. Its own thickness doesn't decide it."),
                ),
                LessonStep(
                    say = Bi("捏一个隐藏层神经元（或面板「神经元 → 看 δ」），看面板上它的 δ 怎么算出来。", "Pinch a hidden neuron (or \"Neuron → its δ\") and read on the panel how its δ is computed."),
                    target = "N_2_0",
                    onEnter = { basePicks = neuronPicks },
                    done = { neuronPicks > basePicks && selCol > 0 },
                    autoplay = { onTap("N_2_0") },
                    insight = Bi("链式法则：我的责任 = 我的斜率 f′(z) × 下游责任的加权和。一次反向就算出所有梯度，代价和前向差不多。", "Chain rule: my blame = my slope f′(z) × the weighted blame downstream. One backward pass gives every gradient, at about the cost of the forward."),
                ),
                LessonStep(
                    say = Bi("再猜：XOR（对角同色的四团点）能被一条直线分开吗？", "Guess again: can XOR (four clusters, diagonal ones alike) be split by one straight line?"),
                    options = listOf(
                        Bi("能", "Yes"),
                        Bi("不能", "No"),
                    ),
                    answer = 1,
                ),
                LessonStep(
                    say = Bi("选数据架上的 XOR，按「训练」，看曲面弯折，直到准确率 100%。", "Pick XOR on the dataset shelf, press Train, and watch the surface fold until accuracy is 100%."),
                    target = "Shelf_0",
                    onEnter = { baseTrain = trainer.steps },
                    done = { kindIdx == 0 && trainer.steps > baseTrain && trainer.accuracy >= 1.0 },
                    autoplay = { onTap("Shelf_0"); runTraining(true) },
                    reveal = true,
                    insight = Bi("一条直线最多分对 3/4。隐藏层先把平面弯折，输出再切一刀 —— 曲面上长出两块对角的高台。", "One line gets at most 3 of 4 clusters. The hidden layers fold the plane, then the output cuts — two diagonal plateaus rise."),
                ),
            ),
            challenge = Challenge(
                say = Bi("挑战：把两个螺旋训练到 ≥ 95% 准确率。自己选隐藏层宽度和学习率 η。", "Challenge: train the two spirals to ≥ 95% accuracy. Choose the hidden width and the learning rate η yourself."),
                progress = {
                    if (kindIdx != 2) tr("先在数据架上选「螺旋」", "First pick \"Spirals\" on the shelf")
                    else tr("螺旋 · 2→%d→%d→1 · η = %s · 第 %d 轮 · 准确率 %s · 损失 %s", "Spirals · 2→%d→%d→1 · η = %s · epoch %d · accuracy %s · loss %s")
                        .format(sizes[1], sizes[1], Trainer.LRS[lrIdx].toString(), trainer.epoch, pct(trainer.accuracy), f(trainer.loss, 3))
                },
                done = { kindIdx == 2 && trainer.accuracy >= 0.95 },
                autoplay = { setDataset(2); setWidth(true); setLr(1); speedIdx = 2; runTraining(true) },
            ),
        )
    }

    // ================================================================ panel

    @Composable
    override fun Panel() {
        var ki by FieldRef(::kindIdx)
        var wd by FieldRef(::wide)
        var li by FieldRef(::lrIdx)
        var si by FieldRef(::speedIdx)
        var sg by FieldRef(::showGrad)
        val st = PicoTheme.typography.bodyLarge.copy(fontSize = 22.sp)
        LabPanel(readout) {
            LabRow {
                Button(onClick = { stepSample(cur) }) { Text(tr("单步", "Step")) }
                Button(onClick = { runTraining(!training) }) { Text(if (training) tr("⏸ 暂停", "⏸ Pause") else tr("▶ 训练", "▶ Train")) }
                Button(onClick = { resetWeights() }) { Text(tr("重置权重", "Reset weights")) }
                Text(tr("学习率 η", "Rate η"), color = PicoTheme.colorScheme.labelSecondary, style = st)
                SegmentControl { Trainer.LRS.forEachIndexed { k, v -> SegmentItem(selected = li == k, onClick = { setLr(k); li = k }, title = { Text(v.toString()) }) } }
            }
            LabRow {
                Text(tr("数据", "Data"), color = PicoTheme.colorScheme.labelSecondary, style = st)
                SegmentControl { KIND_LABELS.forEachIndexed { k, l -> SegmentItem(selected = ki == k, onClick = { setDataset(k) }, title = { Text(l) }) } }
                Text(tr("隐藏层", "Hidden"), color = PicoTheme.colorScheme.labelSecondary, style = st)
                SegmentControl {
                    SegmentItem(selected = !wd, onClick = { setWidth(false) }, title = { Text("4") })
                    SegmentItem(selected = wd, onClick = { setWidth(true) }, title = { Text("8") })
                }
                LabSwitch(tr("显示梯度", "Gradients"), sg) { sg = it }
                Text(tr("速度", "Speed"), color = PicoTheme.colorScheme.labelSecondary, style = st)
                SegmentControl { SPEED_LABELS.forEachIndexed { k, l -> SegmentItem(selected = si == k, onClick = { si = k }, title = { Text(l) }) } }
            }
        }
    }

    // ================================================================ review states

    override fun reviewVariants(): Map<String, () -> Unit> = mapOf(
        "neuron" to { stepSample(cur); repeat(100) { update(1f / 60f) }; pickNeuron(2, 1) },
        "xor_trained" to { setDataset(0); trainer.trainEpochs(40); trainer.evaluate(); sheetDirty = true; refreshSheet(); refreshView() },
        "spirals" to { setDataset(2); setWidth(true); trainer.trainEpochs(300); trainer.evaluate(); sheetDirty = true; refreshSheet(); refreshView(); runTraining(true) },
    )
}
