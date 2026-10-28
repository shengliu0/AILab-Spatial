// 残差网络 · 跳连 (He, Zhang, Ren & Sun 2015, "Deep Residual Learning for Image Recognition").
//
// ONE IDEA: making a PLAIN network deeper can make it train WORSE — its training loss goes UP (the degradation
// problem), because the gradient has to pass back through every layer and fades to nothing on the way. A residual
// block outputs x + F(x); backwards, d/dx = 1 + dF/dx, and the "1" is a gradient highway: 56 layers train as
// easily as 8.
//
// Scene (left -> right): two towers of layer slabs on the bench — PLAIN (coral) and RESIDUAL (cyan, skip arcs on
// its right side; pinch the arcs to switch them off). Loss = the gold ball on top, input at the bottom. On every
// slab a magenta bar = the measured gradient norm ||dL/dW_l|| of that layer (log scale, 10^-6 .. 10^-0.5); a
// backward pulse runs from the loss down and its glow follows the same numbers. Right: the training-loss chart
// (both towers + plain-8 for reference, gold dashes = chance ln 2) with a scrub cursor, and below it the two
// trained nets' decision maps over the spiral data. Front: the depth knob (4 / 8 / 16 / 32 / 56).
//
// REAL DATA: nothing is simulated — the demo REPLAYS the logs of tools/train_resnet.py (10 runs, seeded,
// deterministic): per-epoch loss, accuracy and per-layer gradient norms. The decision maps are a live Kotlin
// forward pass of the 10 trained nets (== PyTorch within 4e-6, ResnetCoreTest). Budget: per frame transforms and
// setMat only; the maps (441 points x 2) are recomputed only on a depth / skip change, spread over frames.

package com.bytedance.ailab.demos.resnet

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.ui.unit.sp
import com.bytedance.ailab.core.P3
import com.bytedance.ailab.core.resnet.MlpNet
import com.bytedance.ailab.core.resnet.ResnetLogs
import com.bytedance.ailab.core.resnet.Run
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
import kotlin.math.floor
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

object ResnetDemo : SpecDemo() {
    override val id = "resnet"
    override val title = "残差网络 · 跳连"  // i18n-ok (zh half of DemoRegistry.titleOf; English in TITLE_EN)
    override val credit = "He, Zhang, Ren & Sun · 2015"

    // ---------------------------------------------------------------- the measurements + the trained nets
    private val logs by lazy { ResnetLogs(LabAssets.bytes(ResnetLogs.PATH)) }
    private val nets by lazy { MlpNet.loadAll(LabAssets.bytes(MlpNet.PATH)) }
    private val DEPTHS = intArrayOf(4, 8, 16, 32, 56)
    private const val MAXD = 56
    private const val EPOCHS = 60

    // ---------------------------------------------------------------- layout (volume metres)
    private val TX = floatArrayOf(-0.58f, -0.14f)          // tower centres: 0 = plain, 1 = right (residual)
    private const val TZ = 0.02f
    private const val Y0 = -0.165f                          // lowest slab (layer 0, next to the input)
    private const val HMAX = 0.52f                          // tower height at 56 layers
    private const val SLAB_W = 0.20f
    private const val SLAB_D = 0.10f
    private const val PLINTH_Y = -0.262f
    private val KNOB = P3(-0.36f, -0.185f, 0.27f)
    private val CHART_C = P3(0.50f, 0.185f, -0.10f)
    private const val CHART_YAW = -0.20f
    private const val CHART_W = 0.66f
    private const val CHART_H = 0.36f
    private const val PLOT_X0 = -0.285f; private const val PLOT_X1 = 0.300f
    private const val PLOT_Y0 = -0.150f; private const val PLOT_Y1 = 0.150f
    private const val LOSS_MAX = 0.8f
    private val MAP_C = arrayOf(P3(0.32f, -0.158f, 0.17f), P3(0.68f, -0.158f, 0.17f))
    private const val MAP_S = 0.15f
    private const val MAP_TILT = -0.75f
    private const val MAP_R = 1.15f                          // input range shown: [-1.15, 1.15]^2
    private const val G = 21                                 // map vertices per side

    // ---------------------------------------------------------------- palette (sRGB)
    private val PLAIN_C = Rgb(1.00f, 0.46f, 0.38f)
    private val RES_C = Rgb(0.36f, 0.86f, 1.00f)
    private val MAG = Rgb(1.00f, 0.22f, 0.86f)
    private val GOLD = Rgb(1.00f, 0.78f, 0.22f)
    private const val LV = 12
    private val barMats = List(LV) { k -> Mat.Unlit(sc(MAG, 0.22f + 0.78f * k / (LV - 1f)), blend = Blend.ADD) }
    private val hotMats = List(LV) { k -> Mat.Unlit(ProcTex.okLerp(sc(MAG, 0.5f + 0.5f * k / (LV - 1f)), Rgb(1f, 0.85f, 1f), 0.15f + 0.45f * k / (LV - 1f)), blend = Blend.ADD) }
    private val beadMats = List(LV) { k -> Mat.Unlit(sc(Rgb(1f, 0.55f, 0.95f), 0.25f + 0.75f * k / (LV - 1f)), tex = ResGfx.softDisc, blend = Blend.ADD) }
    private val arcOn = Mat.Unlit(sc(RES_C, 0.95f), blend = Blend.ADD)
    private val arcHot = Mat.Unlit(Rgb(1f, 0.45f, 0.95f), blend = Blend.ADD)
    private val arcOff = Mat.Unlit(Rgb(0.16f, 0.17f, 0.20f), blend = Blend.ADD)
    private val slabMats = listOf(PLAIN_C, RES_C).map { c -> Mat.Pbr(Rgb(0.13f, 0.15f, 0.20f), metallic = 0.2f, roughness = 0.35f, emissive = sc(c, 0.10f)) }
    private val slabPlainOnRight = slabMats[0]
    private val rimMats = listOf(PLAIN_C, RES_C).map { c -> Mat.Unlit(sc(c, 0.75f), blend = Blend.ADD) }
    private val alu = Mat.Pbr(Rgb(0.78f, 0.80f, 0.84f), metallic = 1f, roughness = 0.30f)
    private val blackGlass = Mat.Pbr(Rgb(0.02f, 0.024f, 0.035f), metallic = 0f, roughness = 0.12f)
    private val lossMat = Mat.Pbr(GOLD, metallic = 0.6f, roughness = 0.25f, emissive = sc(GOLD, 0.55f))
    private val slabTex by lazy { ProcTex.bakePlanet(256, 128, Rgb(0.06f, 0.065f, 0.08f), Rgb(0.10f, 0.105f, 0.125f), Rgb(0.16f, 0.17f, 0.19f), 0f, 91) }

    // ---------------------------------------------------------------- user-facing state
    var depthIdx = 4
    var skipOn = true
    var speedIdx = 1
    var playing = false; private set
    var epochF = EPOCHS.toFloat(); private set
    var selLayer = 0; private set
    private val EPS = floatArrayOf(2f, 5f, 12f)               // replay speed, epochs per second
    private val SPEED_LABELS get() = listOf(tr("慢", "Slow"), tr("中", "Medium"), tr("快", "Fast"))
    val depth get() = DEPTHS[depthIdx]

    // ---------------------------------------------------------------- dynamic state
    private var pulse = 0f
    private var replayEnds = 0
    private val plainSeen = HashSet<Int>()
    private var knobAcc = 0f; private var probeAcc = 0f
    private var probeTaps = 0
    private var curveKey = -1
    private var layoutDepth = -1
    private val barLv = Array(2) { IntArray(MAXD) { -2 } }
    private val barW = Array(2) { FloatArray(MAXD) { -1f } }
    private val arcState = IntArray(MAXD / 2) { -1 }
    // decision maps: progressive recompute
    private val mapP = Array(2) { FloatArray(G * G) }
    private var mapJob = -1                                   // next point index (0 .. 2 G^2), -1 = idle
    private var mapNets: Array<MlpNet>? = null

    // ---------------------------------------------------------------- nodes
    private lateinit var slabs: Array<Array<Node>>
    private lateinit var bars: Array<Array<Node>>
    private lateinit var arcs: Array<Node>
    private lateinit var spines: Array<Node>
    private lateinit var lossBalls: Array<Node>
    private lateinit var beads: Array<Node>
    private lateinit var probes: Array<Node>
    private lateinit var skipHit: Node
    private lateinit var knobDial: Node
    private lateinit var knobTicks: Array<Node>
    private lateinit var curves: Array<MeshData>
    private lateinit var curveNodes: Array<Node>
    private lateinit var cursor: Node
    private lateinit var maps: Array<MeshData>
    private val curveXY = FloatArray((EPOCHS + 1) * 2)

    private fun sc(c: Rgb, k: Float) = Rgb(c.r * k, c.g * k, c.b * k)

    // ================================================================ build

    override fun buildSpec(cacheDir: File): Node {
        resetState()
        val r = Node("ResnetRoot")
        val box = MeshGen.box(1f, 1f, 1f)
        val cyl = MeshGen.cylinder(1f, 1f, 12)
        val sphere = MeshGen.sphere(1f, 12, 20, MeshGen.SphereUv.EQUIRECT)
        val disc = OptGeo.quad(1f, 1f)

        r.add("Plinth", MeshGen.box(1.76f, 0.03f, 0.70f), Mat.Pbr(Rgb(1f, 1f, 1f), roughness = 0.55f, tex = slabTex), P3(0f, PLINTH_Y, 0.03f))
        r.add("PlinthBand", MeshGen.box(1.77f, 0.008f, 0.71f), alu, P3(0f, PLINTH_Y, 0.03f))

        // ---- the two towers
        for (t in 0..1) {
            r.add("Base$t", MeshGen.box(SLAB_W + 0.05f, 0.03f, SLAB_D + 0.05f), alu, P3(TX[t], PLINTH_Y + 0.03f, TZ))
            r.add("BaseRim$t", MeshGen.box(SLAB_W + 0.052f, 0.004f, 0.004f), rimMats[t], P3(TX[t], PLINTH_Y + 0.03f, TZ + (SLAB_D + 0.05f) / 2 + 0.002f))
        }
        spines = Array(2) { t -> r.add("Spine$t", cyl, alu).also { it.pos = P3(TX[t], 0f, TZ) } }
        slabs = Array(2) { t -> Array(MAXD) { l -> r.add("Slab_${t}_$l", box, slabMats[t]).also { it.pos = P3(TX[t], 0f, TZ) } } }
        bars = Array(2) { t -> Array(MAXD) { l -> r.add("Bar_${t}_$l", box, barMats[0]).also { it.visible = false } } }
        arcs = Array(MAXD / 2) { b -> r.add("Arc_$b", ResGfx.arcMesh, arcOn).also { it.visible = false } }
        lossBalls = Array(2) { t -> r.add("Loss_$t", sphere, lossMat) }
        beads = Array(2) { t -> r.add("Bead_$t", disc, beadMats[0]).also { it.billboard = true; it.visible = false } }
        val tri = OptGeo.triPrism(doubleArrayOf(-0.5, 0.5, -0.5, -0.5, 0.5, 0.0), 0.3f)
        probes = Array(2) { t -> r.add("Probe${if (t == 0) "P" else "R"}", tri, Mat.Unlit(GOLD, blend = Blend.ADD)).also { it.uniformScale(0.028f); if (t == 1) it.tapRadius = 0.05f / 0.028f } }
        skipHit = r.add("SkipHit", sphere, Mat.Pbr(RES_C, roughness = 0.3f, emissive = sc(RES_C, 0.8f))).also { it.uniformScale(0.012f); it.tapRadius = 0.055f / 0.012f }

        // ---- depth knob (front, between the towers): a dial facing the viewer, 5 ticks for 4/8/16/32/56
        val faceZ = Q.axisAngle(1f, 0f, 0f, (PI / 2).toFloat())
        r.add("KnobPost", cyl, alu, P3(KNOB.x, (PLINTH_Y + KNOB.y) / 2, KNOB.z - 0.03f)).also { it.scale = P3(0.008f, KNOB.y - PLINTH_Y, 0.008f) }
        r.add("KnobBack", cyl, blackGlass, P3(KNOB.x, KNOB.y, KNOB.z - 0.012f)).also { it.rot = faceZ; it.scale = P3(0.058f, 0.012f, 0.058f) }
        knobDial = r.add(Node("Knob")).also { it.pos = KNOB; it.tapRadius = 0.06f }
        knobDial.add("KnobCap", cyl, alu).also { it.rot = faceZ; it.scale = P3(0.040f, 0.02f, 0.040f) }
        knobDial.add("KnobMark", box, Mat.Unlit(GOLD, blend = Blend.ADD), P3(0f, 0.022f, 0.012f)).also { it.scale = P3(0.007f, 0.026f, 0.004f) }
        knobTicks = Array(5) { k ->
            val a = tickAngle(k)
            r.add("KnobTick$k", sphere, Mat.Unlit(sc(GOLD, 0.35f), blend = Blend.ADD), P3(KNOB.x - 0.052f * sin(a), KNOB.y + 0.052f * cos(a), KNOB.z)).also { it.uniformScale(0.0045f) }
        }

        // ---- the chart (right, behind)
        val chart = r.add(Node("Chart")).also { it.pos = CHART_C; it.rot = Q.axisAngle(0f, 1f, 0f, CHART_YAW) }
        val tw = 396; val th = 216
        fun px(x: Float) = ((x / CHART_W + 0.5f) * (tw - 1)).roundToInt()
        fun py(y: Float) = ((0.5f - y / CHART_H) * (th - 1)).roundToInt()
        chart.add("ChartPlate", OptGeo.quad(CHART_W, CHART_H), Mat.Unlit(Rgb(1f, 1f, 1f),
            tex = ResGfx.chartTex(tw, th, px(PLOT_X0), px(PLOT_X1), py(PLOT_Y1), py(PLOT_Y0), LOSS_MAX, EPOCHS)))
        chart.add("ChartBack", MeshGen.box(CHART_W + 0.016f, CHART_H + 0.016f, 0.008f), alu, P3(0f, 0f, -0.006f))
        chart.add("ChartPost", cyl, alu, P3(0f, (PLINTH_Y - CHART_C.y - CHART_H / 2) / 2, -0.012f)).also { it.scale = P3(0.007f, CHART_C.y - CHART_H / 2 - PLINTH_Y, 0.007f) }
        val curveCols = listOf(sc(Rgb(0.9f, 0.92f, 1f), 0.45f), PLAIN_C, RES_C)       // plain-8 reference, plain-D, right tower
        curves = Array(3) { OptGeo.grid(2, EPOCHS + 1) }
        curveNodes = Array(3) { k -> chart.add("Curve_$k", curves[k], Mat.Unlit(curveCols[k], blend = Blend.ADD)) }
        cursor = chart.add("Cursor", OptGeo.quad(0.006f, PLOT_Y1 - PLOT_Y0 + 0.02f), Mat.Unlit(GOLD, alpha = 0.9f, blend = Blend.ADD)).also { it.tapRadius = 0.05f }

        // ---- decision maps (front right, on lecterns): the trained nets' P(class 1) over the input plane
        val dots = ResGfx.dotsTex(160, MAP_R)
        maps = Array(2) { t ->
            val g = r.add(Node("Map$t")).also { it.pos = MAP_C[t]; it.rot = Q.axisAngle(1f, 0f, 0f, MAP_TILT) }
            g.add("MapBack$t", MeshGen.box(MAP_S + 0.016f, MAP_S + 0.016f, 0.008f), blackGlass, P3(0f, 0f, -0.006f))
            g.add("MapRim$t", MeshGen.box(MAP_S + 0.016f, 0.006f, 0.004f), rimMats[t], P3(0f, -MAP_S / 2 - 0.011f, 0f))
            val md = OptGeo.grid(G, G)
            for (rr in 0 until G) for (c in 0 until G) OptGeo.setPos(md, rr * G + c, -MAP_S / 2 + MAP_S * c / (G - 1), MAP_S / 2 - MAP_S * rr / (G - 1), 0f)
            g.add("MapSheet$t", md, Mat.Unlit(Rgb(1f, 1f, 1f), tex = ResGfx.mapRamp))
            g.add("MapDots$t", OptGeo.quad(MAP_S, MAP_S), Mat.Unlit(Rgb(1f, 1f, 1f), tex = dots, blend = Blend.TRANSPARENT), P3(0f, 0f, 0.0015f))
            val top = P3(MAP_C[t].x, MAP_C[t].y - 0.02f, MAP_C[t].z - 0.03f)
            r.add("MapPost$t", cyl, alu, P3(top.x, (top.y + PLINTH_Y) / 2, top.z)).also { it.scale = P3(0.006f, top.y - PLINTH_Y, 0.006f) }
            md
        }

        applyLayout()
        startMaps(); while (mapJob >= 0) stepMaps(Int.MAX_VALUE)
        update(0f)
        return r
    }

    private fun tickAngle(k: Int) = (-1.2f + 2.4f * k / 4f)

    private fun resetState() {
        depthIdx = 4; skipOn = true; speedIdx = 1; playing = false; epochF = EPOCHS.toFloat(); selLayer = 0
        pulse = 0f; replayEnds = 0; plainSeen.clear(); knobAcc = 0f; probeAcc = 0f; probeTaps = 0
        curveKey = -1; layoutDepth = -1; mapJob = -1; mapNets = null
        for (t in 0..1) { barLv[t].fill(-2); barW[t].fill(-1f) }
        arcState.fill(-1)
        labels.clear(); readoutText = ""
    }

    // ================================================================ geometry helpers

    private fun height(d: Int) = HMAX * sqrt((d - 1) / 55f)
    private fun layerY(l: Int, d: Int = depth) = Y0 + height(d) * l / (d - 1)
    private fun pitch(d: Int = depth) = height(d) / (d - 1)
    private fun runOf(t: Int): Run = logs.run(t == 1 && skipOn, depth)

    /** Gradient norm -> 0..1 on the display's log scale (10^-6 .. 10^-0.5). */
    private fun gScale(g: Float): Float = ((log10(max(g, 1e-12f)) + 6f) / 5.5f).coerceIn(0f, 1f)

    /** Tower layout for the current depth (transforms only; on a depth change). */
    private fun applyLayout() {
        val d = depth
        layoutDepth = d
        val th = min(0.0065f, 0.45f * pitch(d))
        val top = layerY(d - 1, d)
        for (t in 0..1) {
            for (l in 0 until MAXD) {
                val on = l < d
                slabs[t][l].visible = on; if (!on) bars[t][l].visible = false
                if (on) { slabs[t][l].pos = P3(TX[t], layerY(l, d), TZ); slabs[t][l].scale = P3(SLAB_W, th, SLAB_D) }
            }
            barLv[t].fill(-2); barW[t].fill(-1f)
            spines[t].pos = P3(TX[t], (PLINTH_Y + 0.045f + top) / 2, TZ); spines[t].scale = P3(0.006f, top - PLINTH_Y - 0.045f, 0.006f)
        }
        val nb = (d - 2) / 2
        for (b in arcs.indices) {
            val on = b < nb; arcs[b].visible = on; arcState.fill(-1)
            if (on) {
                val y0 = layerY(2 * b, d); val y1 = layerY(2 * b + 2, d)
                arcs[b].pos = P3(TX[1] + SLAB_W / 2 + 0.004f, (y0 + y1) / 2, TZ + SLAB_D / 2 - 0.01f)
                val bulge = 0.022f + 0.18f * (y1 - y0)
                arcs[b].scale = P3(bulge, y1 - y0, bulge)
            }
        }
        skipHit.pos = P3(TX[1] + SLAB_W / 2 + 0.05f, layerY(d / 2, d), TZ + SLAB_D / 2 - 0.01f)
        selLayer = selLayer.coerceIn(0, d - 1)
        val a = tickAngle(depthIdx)
        knobDial.rot = Q.axisAngle(0f, 0f, 1f, a)
        for (k in 0 until 5) knobTicks[k].setMat(Mat.Unlit(sc(GOLD, if (k == depthIdx) 1f else 0.35f), blend = Blend.ADD))
        curveKey = -1
    }

    // ================================================================ user actions (panel, pinches and the lesson call these)

    fun chooseDepth(k: Int) {
        val n = k.coerceIn(0, DEPTHS.size - 1)
        if (n == depthIdx && layoutDepth == depth) return
        depthIdx = n; applyLayout(); startMaps()
    }

    fun setSkip(on: Boolean) {
        if (on == skipOn) return
        skipOn = on; arcState.fill(-1); for (l in 0 until MAXD) barLv[1][l] = -2
        curveKey = -1; startMaps()
    }

    fun play(on: Boolean = true) {
        if (on && epochF >= EPOCHS - 1e-3f) epochF = 0f
        playing = on
    }

    fun restart() { epochF = 0f; playing = true }
    fun toEnd() { epochF = EPOCHS.toFloat(); playing = false }
    fun scrubTo(e: Float) { epochF = e.coerceIn(0f, EPOCHS.toFloat()); playing = false }

    /** Probe stops: input layer, quarters, the layer under the loss. */
    fun nextProbe() {
        val d = depth
        val stops = listOf(0, d / 4, d / 2, 3 * d / 4, d - 1).distinct()
        selLayer = stops.firstOrNull { it > selLayer } ?: 0
        probeTaps++
    }

    // ================================================================ decision maps (live forward pass, spread over frames)

    private fun startMaps() {
        mapNets = arrayOf(nets.getValue(false to depth), nets.getValue((skipOn) to depth))
        mapJob = 0
    }

    /** Compute up to [budget] layer-evaluations' worth of map points; upload a map when it is complete. */
    private fun stepMaps(budget: Int) {
        val ns = mapNets ?: return
        if (mapJob < 0) return
        var left = budget
        val scratch = FloatArray(2)
        while (mapJob < 2 * G * G && left > 0) {
            val t = mapJob / (G * G); val i = mapJob % (G * G)
            val rr = i / G; val c = i % G
            val x1 = -MAP_R + 2 * MAP_R * c / (G - 1); val x2 = MAP_R - 2 * MAP_R * rr / (G - 1)
            mapP[t][i] = ns[t].p1(x1, x2, scratch)
            left -= ns[t].depth
            mapJob++
            if (mapJob % (G * G) == 0) {
                val md = maps[t]
                for (k in 0 until G * G) OptGeo.setUv(md, k, OptGeo.rampU(mapP[t][k]), 0.5f)
                meshEdited(md)
            }
        }
        if (mapJob >= 2 * G * G) mapJob = -1
    }

    // ================================================================ per frame

    override fun update(dt: Float) {
        if (layoutDepth != depth) applyLayout()
        if (playing) {
            epochF += dt * EPS[speedIdx]
            if (epochF >= EPOCHS) { epochF = EPOCHS.toFloat(); playing = false; replayEnds++ }
        }
        if (epochF >= EPOCHS - 1e-3f) plainSeen += depth
        pulse = (pulse + dt / 1.8f) % 1f
        if (mapJob >= 0) stepMaps(1800)
        applyTowers()
        applyChart()
        updateLabels()
        updateReadout()
    }

    private fun applyTowers() {
        val d = depth
        val e = epochF
        // the backward pulse: from the loss (top) down to layer 0 in 80 % of the period, then a short rest
        val pl = (d - 1) * (1f - (pulse / 0.8f)).coerceIn(-0.2f, 1f)
        val pulseOn = pulse < 0.8f
        for (t in 0..1) {
            val run = runOf(t)
            val slabM = if (t == 1 && !skipOn) slabPlainOnRight else slabMats[t]
            for (l in 0 until d) {
                slabs[t][l].setMat(slabM)
                val s = gScale(run.gradAt(l, e))
                val hot = pulseOn && abs(l - pl) < max(1.2f, d / 18f) && s > 0.06f
                val lv = if (s <= 0.02f) -1 else ((s * (LV - 1)).roundToInt().coerceIn(0, LV - 1) + if (hot) 100 else 0)
                val w = (0.008f + 0.182f * s).let { (it * 400f).roundToInt() / 400f }
                val bar = bars[t][l]
                if (lv != barLv[t][l]) {
                    barLv[t][l] = lv
                    bar.visible = lv >= 0
                    if (lv >= 0) bar.setMat(if (lv >= 100) hotMats[lv - 100] else barMats[lv])
                }
                if (w != barW[t][l]) {
                    barW[t][l] = w
                    val th = min(0.0065f, 0.45f * pitch(d)) * 0.9f
                    bar.pos = P3(TX[t], layerY(l, d), TZ + SLAB_D / 2 + 0.002f); bar.scale = P3(w, th, 0.004f)
                }
            }
            // bead riding the pulse: its glow = the gradient where it is
            val bead = beads[t]
            val li = pl.roundToInt().coerceIn(0, d - 1)
            val s = gScale(run.gradAt(li, e))
            bead.visible = pulseOn && s > 0.04f && pl >= -0.1f
            if (bead.visible) {
                bead.pos = P3(TX[t], layerY(0, d) + (pl.coerceAtLeast(0f) / (d - 1)) * height(d), TZ + SLAB_D / 2 + 0.008f)
                bead.uniformScale(0.02f + 0.07f * s)
                bead.setMat(beadMats[(s * (LV - 1)).roundToInt().coerceIn(0, LV - 1)])
            }
            // the loss on top: radius ~ training loss
            val loss = run.lossAt(e)
            lossBalls[t].pos = P3(TX[t], layerY(d - 1, d) + 0.04f, TZ)
            lossBalls[t].uniformScale(0.008f + 0.028f * (loss / 0.75f).coerceIn(0f, 1.3f))
            // probe pointer at the selected layer, left of each tower
            probes[t].pos = P3(TX[t] - SLAB_W / 2 - 0.022f, layerY(selLayer, d), TZ + SLAB_D / 2)
        }
        // skip arcs: cyan when on, magenta while the pulse passes their block, dark when switched off
        val nb = (d - 2) / 2
        for (b in 0 until nb) {
            val st = if (!skipOn) 0 else if (pulseOn && abs(2 * b + 1 - pl) < max(1.5f, d / 14f)) 2 else 1
            if (st != arcState[b]) { arcState[b] = st; arcs[b].setMat(when (st) { 0 -> arcOff; 2 -> arcHot; else -> arcOn }) }
        }
    }

    private fun applyChart() {
        val ei = floor(epochF).toInt()
        val key = ((depthIdx * 2 + if (skipOn) 1 else 0) * 100 + ei)
        val x = PLOT_X0 + (PLOT_X1 - PLOT_X0) * epochF / EPOCHS
        cursor.pos = P3(x, (PLOT_Y0 + PLOT_Y1) / 2, 0.003f)
        if (key == curveKey) return
        curveKey = key
        val runs = arrayOf(logs.run(false, 8), logs.run(false, depth), logs.run(skipOn, depth))
        for (k in 0..2) {
            val r = runs[k]; val m = ei + 1
            for (j in 0 until m) {
                curveXY[j * 2] = PLOT_X0 + (PLOT_X1 - PLOT_X0) * j / EPOCHS
                curveXY[j * 2 + 1] = PLOT_Y0 + (PLOT_Y1 - PLOT_Y0) * (r.loss[j] / LOSS_MAX).coerceIn(0f, 1f)
            }
            polylineInto(curves[k], curveXY, m, if (k == 0) 0.0035f else 0.0055f, 0.001f + 0.001f * k)
            meshEdited(curves[k])
        }
        curveNodes[0].visible = depth != 8
    }

    /** Rewrite a 2 x K grid as a flat ribbon of width [w] through the first [count] points of [xy]; in place. */
    private fun polylineInto(md: MeshData, xy: FloatArray, count: Int, w: Float, z: Float) {
        val k = md.vertexCount / 2
        val m = count.coerceIn(1, k)
        for (i in 0 until k) {
            val p = min(i, m - 1)
            val x = xy[p * 2]; val y = xy[p * 2 + 1]
            val pa = max(p - 1, 0); val pb = min(p + 1, m - 1)
            var dx = xy[pb * 2] - xy[pa * 2]; var dy = xy[pb * 2 + 1] - xy[pa * 2 + 1]
            val l = sqrt(dx * dx + dy * dy)
            if (l < 1e-6f) { dx = 1f; dy = 0f } else { dx /= l; dy /= l }
            val ox = -dy * w / 2; val oy = dx * w / 2
            OptGeo.setPos(md, i, x + ox, y + oy, z); OptGeo.setUv(md, i, 0.5f, 0f)
            OptGeo.setPos(md, k + i, x - ox, y - oy, z); OptGeo.setUv(md, k + i, 0.5f, 1f)
        }
    }

    // ================================================================ interaction

    override fun isTarget(name: String) = name == "Knob" || name == "SkipHit" || name == "ProbeR" || name == "Cursor"
    override fun isDragTarget(name: String) = name == "Knob" || name == "ProbeR" || name == "Cursor"

    override val tapTargets get() = listOf(
        TapTarget("Knob", "深度旋钮 → 下一档", "Depth knob → next depth"),
        TapTarget("SkipHit", "跳连 → 开/关", "Skip arcs → on/off") { skipOn },
        TapTarget("ProbeR", "层探针 → 换一层", "Layer probe → next layer"),
        TapTarget("Cursor", "时间轴 → 回放/暂停", "Timeline → replay/pause") { playing },
    )

    override fun onTap(name: String) {
        when (name) {
            "Knob" -> chooseDepth((depthIdx + 1) % DEPTHS.size)
            "SkipHit" -> setSkip(!skipOn)
            "ProbeR" -> nextProbe()
            "Cursor" -> play(!playing)
        }
    }

    override fun tapMessage(name: String): String? = when (name) {
        "Knob" -> tr("深度 %d 层：两座塔都换成 %d 层的真实记录", "Depth %d: both towers now replay the %d-layer logs").format(depth, depth)
        "SkipHit" -> if (skipOn) tr("跳连已接通：右塔 = 残差网络 x + F(x)", "Skips ON: the right tower is residual, x + F(x)")
                     else tr("跳连已断开：右塔也成了普通网络", "Skips OFF: the right tower is a plain net too")
        "ProbeR" -> tr("探针在第 %d 层：两座塔这一层的梯度在面板上", "Probe at layer %d: both towers' gradients are on the panel").format(selLayer + 1)
        "Cursor" -> if (playing) tr("回放训练：第 %d 轮起", "Replaying training from epoch %d").format(epochF.toInt()) else tr("已暂停在第 %d 轮", "Paused at epoch %d").format(epochF.toInt())
        else -> null
    }

    override fun onDragStart(name: String) { knobAcc = 0f; probeAcc = 0f }

    override fun onDrag(name: String, dx: Float, dy: Float, dz: Float) {
        when (name) {
            "Knob" -> {          // turn by moving the hand sideways or up: 4 cm per notch
                knobAcc += dx - dy
                while (knobAcc > 0.04f) { knobAcc -= 0.04f; chooseDepth(depthIdx + 1) }
                while (knobAcc < -0.04f) { knobAcc += 0.04f; chooseDepth(depthIdx - 1) }
            }
            "ProbeR" -> {
                probeAcc += dy
                val p = max(pitch(), 0.012f)
                while (probeAcc > p) { probeAcc -= p; selLayer = min(selLayer + 1, depth - 1) }
                while (probeAcc < -p) { probeAcc += p; selLayer = max(selLayer - 1, 0) }
            }
            "Cursor" -> scrubTo(epochF + dx / cos(CHART_YAW) / (PLOT_X1 - PLOT_X0) * EPOCHS)
        }
    }

    // ================================================================ readout + labels

    private val SUP = mapOf('-' to '⁻', '0' to '⁰', '1' to '¹', '2' to '²', '3' to '³', '4' to '⁴', '5' to '⁵', '6' to '⁶', '7' to '⁷', '8' to '⁸', '9' to '⁹')
    /** 3.8×10⁻⁷ style. */
    fun sci(v: Float): String {
        if (v <= 0f) return "0"
        val ex = floor(log10(v.toDouble())).toInt()
        if (ex in -2..1) return "%.3f".format(v)
        val m = v / 10.0.pow(ex)
        return "%.1f×10%s".format(m, ex.toString().map { SUP[it] ?: it }.joinToString(""))
    }
    private fun ratio(a: Float, b: Float): String {
        val r = max(a, b) / max(min(a, b), 1e-12f)
        return if (r < 100f) "%.0f".format(r) else sci(r)
    }
    private fun pct(x: Float) = "${(x * 100 + 0.5f).toInt()}%"

    private fun rightName() = if (skipOn) tr("残差", "residual") else tr("普通(跳连关)", "plain (skips off)")

    private fun updateReadout() {
        val p = logs.run(false, depth); val r = runOf(1); val e = epochF
        val l1 = tr("深度 %d · 第 %d/%d 轮 · 训练损失 普通 %.3f | %s %.3f · 准确率 %s | %s · 回放真实训练记录",
            "Depth %d · epoch %d/%d · train loss plain %.3f | %s %.3f · acc %s | %s · real training logs")
            .format(depth, e.toInt(), EPOCHS, p.lossAt(e), rightName(), r.lossAt(e), pct(p.accAt(e)), pct(r.accAt(e)))
        val gp = p.gradAt(selLayer, e); val gr = r.gradAt(selLayer, e)
        val l2 = tr("第 %d 层（共 %d，1 = 最靠输入）‖∂L/∂W‖：普通 %s · %s %s · 相差 %s 倍",
            "Layer %d of %d (1 = next to the input) ‖∂L/∂W‖: plain %s · %s %s · ratio %s")
            .format(selLayer + 1, depth, sci(gp), rightName(), sci(gr), ratio(gp, gr))
        val l3 = when {
            depth <= 16 && !skipOn -> tr("浅网也能学会；继续加深，普通网络的训练误差会不降反升", "Shallow nets learn fine; go deeper and the plain net's training error goes UP")
            depth <= 16 -> tr("浅网两种都能学会；拨到 32 或 56 层，差别才出现", "Shallow: both learn; turn the knob to 32 or 56 to see the difference")
            !skipOn -> tr("普通深网：梯度要逐层相乘，传到底层几乎为零 —— 学不动，训练误差反而更高（退化）", "Deep plain net: the gradient is multiplied layer after layer and vanishes — training error goes UP (degradation)")
            else -> tr("跳连 x + F(x)：反传时 ∂/∂x = 1 + ∂F/∂x，那个「1」是梯度高速路，直达底层", "Skip x + F(x): backwards ∂/∂x = 1 + ∂F/∂x — the \"1\" is a gradient highway to the bottom")
        }
        readoutText = "$l1\n$l2\n$l3"
    }

    private fun updateLabels() {
        labels.clear()
        val top = layerY(depth - 1) + 0.10f
        labels += LabelSpec(tr("普通网络 · %d 层", "Plain · %d layers").format(depth), P3(TX[0], top, TZ))
        labels += LabelSpec(if (skipOn) tr("残差网络 x + F(x) · %d 层", "Residual x + F(x) · %d layers").format(depth)
                            else tr("跳连已断开 = 普通 · 捏右侧接通", "Skips off = plain · pinch right side"), P3(TX[1] + 0.02f, top, TZ))
        labels += LabelSpec(tr("训练损失 · 第 %d/%d 轮 · 白 = 普通 8 层", "Training loss · epoch %d/%d · white = plain 8").format(epochF.toInt(), EPOCHS),
            P3(CHART_C.x, CHART_C.y + CHART_H / 2 + 0.04f, CHART_C.z))
        val gp = logs.run(false, depth).gradAt(selLayer, epochF); val gr = runOf(1).gradAt(selLayer, epochF)
        labels += LabelSpec(tr("第 %d 层梯度 %s | %s", "Layer %d gradient %s | %s").format(selLayer + 1, sci(gp), sci(gr)),
            P3((TX[0] + TX[1]) / 2, layerY(selLayer) + 0.035f, TZ + 0.08f))
        labels += LabelSpec(tr("深度 %d", "Depth %d").format(depth), P3(KNOB.x, KNOB.y + 0.085f, KNOB.z))
        labels += LabelSpec(tr("训练后的决策图 · 普通 | 右塔", "Decision maps after training · plain | right"), P3((MAP_C[0].x + MAP_C[1].x) / 2, MAP_C[0].y + 0.115f, MAP_C[0].z))
    }

    // ================================================================ guided track (导学) + challenge
    // KEY INSIGHT: deeper plain nets train WORSE (degradation); x + F(x) gives the gradient a highway (d/dx = 1 + dF/dx).

    private var baseEnds = 0
    private var baseProbe = 0
    private fun fin(res: Boolean, d: Int) = logs.run(res, d).loss[EPOCHS]
    private fun finA(res: Boolean, d: Int) = logs.run(res, d).acc[EPOCHS]

    // ---- background card (📖): context for a viewer who has never met this topic
    override val about = About(listOf(
        About.Page(
            Bi("这个实验在做什么？", "What is this experiment?"),
            Bi("工作台上有两座「层塔」：每一片是网络的一层，底部是输入，顶上的金球是损失。左塔（珊瑚色）是普通网络，右塔（青色）侧面多了跳连弧线。前面的旋钮把两座塔加深到 4、8、16、32、56 层；按 ▶ 回放真实的训练记录，右边看损失曲线和训练后的决策图。捏跳连可以断开它。", "Two “layer towers” stand on the bench: each slab is one layer, the input at the bottom, the loss a gold ball on top. The left tower (coral) is a plain network; the right one (cyan) has skip arcs along its side. The knob in front deepens both to 4, 8, 16, 32 or 56 layers; press ▶ to replay real training logs and watch the loss curves and trained decision maps on the right. Pinch the arcs to cut them."),
        ),
        About.Page(
            Bi("先认识几个词", "A few words first"),
            Bi("「深度」= 网络叠了多少层。训练靠反向传播：误差从顶上的损失往下传，每层收到一个「梯度」，告诉它该怎么改；梯度太小，这一层就等于没在学。这里的任务是分开两条交缠的螺旋（2000 个点）。只有两类，所以准确率 50% 就等于瞎猜。", "“Depth” = how many layers a network stacks. Training uses backpropagation: the error flows down from the loss at the top, and each layer receives a “gradient” telling it how to change; if the gradient is tiny, that layer is not really learning. The task here is to separate two intertwined spirals (2,000 points). With only two classes, 50 % accuracy means guessing."),
        ),
        About.Page(
            Bi("背景：越深反而越差？", "Background: deeper but worse?"),
            Bi("2015 年前后，人们发现一件怪事：把普通网络从 20 层加深到 56 层，连训练集上的误差都变高了。这不是死记硬背（过拟合），而是根本学不动，叫「退化」。微软亚洲研究院的何恺明、张祥雨、任少卿和孙剑提出残差网络，用 152 层的网络赢下当年的 ImageNet 比赛，这篇论文成了被引用最多的论文之一。", "Around 2015 researchers noticed something odd: deepening a plain network from 20 to 56 layers made even its training error worse. Not memorising (overfitting) — it simply could not learn; this was called “degradation”. Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun at Microsoft Research Asia proposed residual networks and won that year's ImageNet contest with 152 layers; the paper became one of the most cited ever."),
        ),
        About.Page(
            Bi("核心想法：给梯度修高速路", "Key idea: a highway for gradients"),
            Bi("普通网络像传话游戏：一句话经过 56 个人转述，到最后早已面目全非；反过来，误差信号也要逐层相乘，传到底层几乎为零。残差块的输出是 x + F(x)：原话照抄一份，每层只补一点修改。反向时多出的那个「1」，让梯度可以沿跳连直接流到底层。", "A plain network is like a game of telephone: a message re-told by 56 people arrives mangled; backwards, the error signal is multiplied layer after layer and is almost zero by the bottom. A residual block outputs x + F(x): a copy of the original plus a small correction. Backwards, the extra “1” lets the gradient run straight down the skip path to the bottom."),
        ),
        About.Page(
            Bi("怎么看这个 3D 场景", "How to read the 3-D scene"),
            Bi("每片薄板 = 一层；板上的品红条 = 这一层实测的梯度大小（对数刻度，从 10⁻⁶ 到约 0.3）；从顶上往下跑的光脉冲 = 一次反向传播。青色弧线 = 跳连 x + F(x)；金色探针 = 正在比较的那一层。右边的曲线 = 训练损失随轮数的变化（白线 = 普通 8 层作参照，金色虚线 = 瞎猜水平）；下方两张图 = 训练后各自怎么划分螺旋。", "Each slab = one layer; its magenta bar = that layer's measured gradient size (log scale, from 10⁻⁶ to about 0.3); the pulse running down from the top = one backward pass. Cyan arcs = skips x + F(x); the gold probe = the layer being compared. On the right, curves = training loss over the epochs (white = plain-8 for reference, gold dashes = guessing); the two maps below = how each trained net splits the spirals."),
        ),
        About.Page(
            Bi("关键数字 · 今天", "Key numbers · today"),
            Bi("记录是真的：56 层普通网络训练 60 轮，准确率只有 51%（等于瞎猜），第 1 层梯度约 4×10⁻⁷；加上跳连的 56 层准确率 99%，第 1 层梯度约 0.1，大了二十多万倍。这里每层只有 32 个神经元，ResNet-152 则有约 6000 万参数。今天的 Transformer（ChatGPT 的骨架）每一层都带同样的跳连。", "These are real logs: after 60 epochs the plain 56-layer net reaches only 51 % accuracy (a coin flip) and its first-layer gradient is about 4×10⁻⁷; with skips the 56-layer net reaches 99 % and its first-layer gradient is about 0.1 — over 200,000 times larger. Each layer here has just 32 neurons; ResNet-152 has about 60 million parameters. Today's Transformers (the backbone of ChatGPT) carry the same skips in every layer."),
        ),
    ))

    override val lesson: Lesson by lazy {
        val p8 = "%.3f".format(fin(false, 8)); val p56 = "%.3f".format(fin(false, 56)); val r56 = "%.3f".format(fin(true, 56))
        val a8 = pct(finA(false, 8)); val a56 = pct(finA(false, 56)); val ar56 = pct(finA(true, 56))
        val gP = sci(logs.run(false, 56).g[EPOCHS][0]); val gR = sci(logs.run(true, 56).g[EPOCHS][0])
        Lesson(
            listOf(
                LessonStep(
                    say = Bi("先猜：把普通网络从 8 层加深到 56 层（同样宽度、同样训练 60 轮），它在训练集上的误差会？", "Guess first: make a plain net 56 layers deep instead of 8 (same width, same 60 epochs). Its TRAINING error will be…"),
                    options = listOf(
                        Bi("更低（层多更强）", "Lower (more layers, more power)"),
                        Bi("差不多", "About the same"),
                        Bi("更高", "Higher"),
                    ),
                    answer = 2,
                    onEnter = { setSkip(false); chooseDepth(1); toEnd() },
                ),
                LessonStep(
                    say = Bi("捏前面的深度旋钮拨到 56（或面板选 56），按 ▶ 回放训练，盯住左边的普通塔和曲线。", "Turn the depth knob to 56 (or pick 56 on the panel), press ▶ Replay and watch the plain tower and its curve."),
                    target = "Knob",
                    onEnter = { baseEnds = replayEnds },
                    done = { depth == 56 && replayEnds > baseEnds },
                    autoplay = { chooseDepth(4); speedIdx = 2; play(true) },
                    reveal = true,
                    insight = Bi("真实记录：普通 8 层 损失 $p8（准确率 $a8），56 层 $p56（$a56 = 瞎猜）。连训练集都学不会 —— 不是过拟合，是「退化」。看塔底：品红梯度条几乎消失（第 1 层 $gP）。", "Real logs: plain-8 loss $p8 ($a8 acc), plain-56 $p56 ($a56 = guessing). It can't even fit the training set — not overfitting: degradation. The magenta bars vanish at the bottom (layer 1: $gP)."),
                ),
                LessonStep(
                    say = Bi("再猜：给 56 层加上跳连，每两层输出 x + F(x)，结果会？", "Guess again: add skip connections, every two layers output x + F(x). The 56-layer net will…"),
                    options = listOf(
                        Bi("还是学不会", "Still fail to learn"),
                        Bi("和浅网一样好", "Train as well as a shallow net"),
                        Bi("更糟", "Do even worse"),
                    ),
                    answer = 1,
                ),
                LessonStep(
                    say = Bi("捏右塔侧面的青色跳连（或面板「跳连」）接通，再按 ▶ 回放一次。", "Pinch the cyan skip arcs on the right tower (or the Skip switch) to connect them, then ▶ Replay again."),
                    target = "SkipHit",
                    onEnter = { baseEnds = replayEnds },
                    done = { skipOn && depth == 56 && replayEnds > baseEnds },
                    autoplay = { chooseDepth(4); setSkip(true); speedIdx = 2; play(true) },
                    reveal = true,
                    insight = Bi("残差 56 层：损失 $r56，准确率 $ar56。反传时 ∂(x + F(x))/∂x = 1 + ∂F/∂x：那个「1」让梯度不必层层相乘，沿跳连直达底层 —— 第 1 层梯度 $gR，普通网络只有 $gP。", "Residual-56: loss $r56, accuracy $ar56. Backwards ∂(x + F(x))/∂x = 1 + ∂F/∂x: the \"1\" lets the gradient skip the products and reach the bottom — layer 1 gets $gR, the plain net $gP."),
                ),
                LessonStep(
                    say = Bi("捏塔边的金色层探针换几层，比较两座塔同一层的梯度。", "Pinch the gold layer probe a few times and compare both towers at the same layer."),
                    target = "ProbeR",
                    onEnter = { baseProbe = probeTaps },
                    done = { probeTaps >= baseProbe + 2 },
                    autoplay = { onTap("ProbeR"); onTap("ProbeR") },
                    insight = Bi("越靠近输入，普通网络的梯度越小；残差网络各层梯度都在同一个量级。所以 He 等人能训练 152 层、甚至 1000 层的网络。", "The closer to the input, the smaller the plain net's gradient; the residual net's stays in one range at every layer. That is how He et al. trained 152 — even 1000 — layers."),
                ),
            ),
            challenge = Challenge(
                say = Bi("挑战：断开跳连，从浅到深逐档看普通网络的最终损失，停在它第一次「越深越差」的深度。", "Challenge: switch the skips off, step the plain net from shallow to deep, and stop at the first depth where deeper is WORSE."),
                progress = {
                    val seen = DEPTHS.filter { it in plainSeen }.joinToString(" · ") { "$it→%.3f".format(fin(false, it)) }
                    if (seen.isEmpty()) tr("把时间轴拉到最后（第 60 轮）再换深度", "Move the timeline to the end (epoch 60), then change depth")
                    else tr("已看的普通网络最终损失：%s · 当前 %d 层", "Plain final losses seen: %s · now %d layers").format(seen, depth)
                },
                done = { !skipOn && depth == 32 && 16 in plainSeen && 32 in plainSeen },
                autoplay = { setSkip(false); toEnd(); chooseDepth(2); update(0f); chooseDepth(3) },
            ),
        )
    }

    // ================================================================ panel

    @Composable
    override fun Panel() {
        var di by FieldRef(::depthIdx)
        var sk by FieldRef(::skipOn)
        var si by FieldRef(::speedIdx)
        val st = PicoTheme.typography.bodyLarge.copy(fontSize = 22.sp)
        LabPanel(readout) {
            LabRow {
                Button(onClick = { play(!playing) }) { Text(if (playing) tr("⏸ 暂停", "⏸ Pause") else tr("▶ 回放训练", "▶ Replay")) }
                Button(onClick = { restart() }) { Text(tr("⏮ 从头", "⏮ From 0")) }
                Button(onClick = { toEnd() }) { Text(tr("⏭ 结束", "⏭ End")) }
                Text(tr("深度", "Depth"), color = PicoTheme.colorScheme.labelSecondary, style = st)
                SegmentControl { DEPTHS.forEachIndexed { k, d -> SegmentItem(selected = di == k, onClick = { chooseDepth(k); di = k }, title = { Text("$d") }) } }
            }
            LabRow {
                LabSwitch(tr("跳连 x + F(x)", "Skips x + F(x)"), sk) { setSkip(it); sk = it }
                Text(tr("速度", "Speed"), color = PicoTheme.colorScheme.labelSecondary, style = st)
                SegmentControl { SPEED_LABELS.forEachIndexed { k, l -> SegmentItem(selected = si == k, onClick = { si = k }, title = { Text(l) }) } }
            }
        }
    }

    // ================================================================ review states

    override fun reviewVariants(): Map<String, () -> Unit> = mapOf(
        "skips_off" to { chooseDepth(4); setSkip(false); toEnd() },
        "init" to { chooseDepth(4); setSkip(true); scrubTo(0f) },
        "depth8" to { chooseDepth(1); setSkip(true); toEnd() },
    )
}
