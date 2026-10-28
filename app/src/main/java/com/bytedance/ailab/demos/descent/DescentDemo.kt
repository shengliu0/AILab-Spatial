// 梯度下降 · 损失曲面 (Cauchy, 1847) — training a network = a ball rolling downhill on the loss surface.
//
// LESSON: the parameters are a point on a landscape whose height is the loss; every step moves it along −∇f by
// η·|∇f|. The learning rate η is the step length: too small crawls, too big overshoots the valley and — once
// |1 − ηλmax| > 1 — bounces out of it (发散). Momentum (a heavier ball that keeps its velocity) and Adam (a ball that
// rescales its step per direction) behave like smarter balls. Three balls race from the same start with the same η.
//
// Scene (volume metres): a tilted "board" in the middle carries the landscape (height-field mesh coloured by height,
// baked lighting + iso-lines; a contour map on its floor), the balls with glowing trails and their −∇f arrows and a
// gold start pin (pinch-drag it). Right: the landscape shelf (5 mini landscapes). Left: the loss curves (log scale)
// and the dock where switched-off balls wait. All maths lives in core/descent (pure Kotlin, JVM-tested against
// torch.optim); this file only draws what the race records.

package com.bytedance.ailab.demos.descent

import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.width
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.bytedance.ailab.core.P3
import com.bytedance.ailab.core.descent.LR_STEPS
import com.bytedance.ailab.core.descent.Logistic
import com.bytedance.ailab.core.descent.LossSurface
import com.bytedance.ailab.core.descent.Race
import com.bytedance.ailab.core.descent.Surfaces
import com.bytedance.ailab.core.descent.lrIndexOf
import com.bytedance.ailab.lab.BakeCache
import com.bytedance.ailab.lab.About
import com.bytedance.ailab.lab.Bi
import com.bytedance.ailab.lab.Lang
import com.bytedance.ailab.lab.Challenge
import com.bytedance.ailab.lab.Lesson
import com.bytedance.ailab.lab.LessonStep
import com.bytedance.ailab.lab.Q
import com.bytedance.ailab.lab.SpecDemo
import com.bytedance.ailab.lab.Sfx
import com.bytedance.ailab.lab.TapTarget
import com.bytedance.ailab.lab.spec.Blend
import com.bytedance.ailab.lab.spec.LabelSpec
import com.bytedance.ailab.lab.spec.Mat
import com.bytedance.ailab.lab.spec.Node
import com.bytedance.ailab.lab.tr
import com.bytedance.ailab.lab.ui.Button
import com.bytedance.ailab.lab.ui.LabPanel
import com.bytedance.ailab.lab.ui.LabSlider
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
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.log10
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

object DescentDemo : SpecDemo() {
    override val id = "descent"
    override val title = "梯度下降 · 损失曲面"  // i18n-ok (zh half of DemoRegistry.titleOf; English in TITLE_EN)
    override val credit = "Cauchy · 1847"

    // ================================================================ layout
    private val TILT = Math.toRadians(34.0).toFloat()          // the board leans toward the viewer (eye ≈ table height)
    private val CT = cos(TILT); private val ST = sin(TILT)
    private val BC = P3(0.0f, 0.025f, -0.02f)                // centre of the board's floor plane
    private val BOARD = DescentGfx.Board(0.80f, 0.025f, 0.20f)
    private const val GRID_N = 61                              // 3721 vertices, uploaded once per landscape
    private const val TEX = 256
    private const val FLOOR_TEX = 192
    private const val BALL_R = 0.020f
    private const val SLOTS = 64                                // trail dots per ball
    private const val RECENT = 40                               // ... of which the latest RECENT steps at full resolution
    /** Shelf (right): 2 columns × 3 rows, the fifth landscape centred in the bottom row. */
    private fun shelfPos(k: Int) = if (k < 4) P3(0.585f + 0.155f * (k % 2), 0.215f - 0.145f * (k / 2), 0.04f) else P3(0.6625f, -0.075f, 0.04f)
    private val MINI = DescentGfx.Board(0.115f, 0.005f, 0.045f)
    private val CHART_C = P3(-0.69f, 0.165f, 0.0f); private val CHART_W = 0.36f; private val CHART_H = 0.28f
    private val CHART_YAW = Math.toRadians(16.0).toFloat()
    private const val CHART_K = 64                              // points per loss curve
    private const val LOG_LO = -4f; private const val LOG_HI = 3.5f
    private fun dockPos(k: Int) = P3(-0.80f + 0.11f * k, -0.135f, 0.17f)

    /** Board-local → volume. */
    private fun toWorld(p: P3) = P3(BC.x + p.x, BC.y + p.y * CT - p.z * ST, BC.z + p.y * ST + p.z * CT)
    /** Volume direction → board-local. */
    private fun toLocalDir(dx: Float, dy: Float, dz: Float) = P3(dx, dy * CT + dz * ST, -dy * ST + dz * CT)
    private fun tiltQ() = Q.axisAngle(1f, 0f, 0f, TILT)

    // ================================================================ colours (sRGB)
    private val BALL_C = listOf(Rgb(0.95f, 0.97f, 1.0f), Rgb(1.0f, 0.56f, 0.14f), Rgb(0.25f, 0.86f, 1.0f))
    private val GOLD = Rgb(1.0f, 0.82f, 0.38f)
    private val RED = Rgb(1.0f, 0.24f, 0.18f)
    private fun scaled(c: Rgb, k: Float) = Rgb(c.r * k, c.g * k, c.b * k)
    private fun add(c: Rgb, a: Float = 1f, tex: Image? = null) = Mat.Unlit(c, alpha = a, tex = tex, blend = Blend.ADD)
    private val alu = Mat.Pbr(Rgb(0.80f, 0.82f, 0.85f), metallic = 1f, roughness = 0.32f)
    private val slabTex by lazy { ProcTex.bakePlanet(256, 128, Rgb(0.06f, 0.065f, 0.075f), Rgb(0.10f, 0.105f, 0.12f), Rgb(0.16f, 0.165f, 0.18f), 0f, 88) }
    private val glow by lazy { DescentGfx.glowTex() }
    private val ballOn = List(3) { Mat.Pbr(BALL_C[it], metallic = 0.1f, roughness = 0.25f, emissive = scaled(BALL_C[it], 0.55f)) }
    private val ballOff = List(3) { Mat.Pbr(scaled(BALL_C[it], 0.45f), metallic = 0.1f, roughness = 0.45f, emissive = scaled(BALL_C[it], 0.08f)) }
    private val ballRed = Mat.Pbr(RED, metallic = 0.1f, roughness = 0.3f, emissive = scaled(RED, 0.8f))
    private val glowOn = List(3) { add(BALL_C[it], 0.55f, glow) }
    private val glowRed by lazy { add(RED, 0.75f, glow) }
    private val dotMat = List(3) { k -> listOf(1.0f, 0.62f, 0.32f).map { a -> add(BALL_C[k], a) } }
    private val segMat = List(3) { k -> listOf(0.75f, 0.45f, 0.22f).map { a -> add(BALL_C[k], a) } }
    private val arrowMat = List(3) { add(BALL_C[it], 0.95f) }

    // ================================================================ state (public: panel + lesson)
    val race = Race()
    var surfIdx = 0; private set
    var speedIdx = 1; private set
    var playing = true; private set
    private var stepAcc = 0f
    private val hop = FloatArray(3) { 1f }
    private var dragging = false
    private var dragX = 0.0; private var dragY = 0.0
    private var shownSurf = -1
    private val trailKey = IntArray(3) { -1 }
    private var chartT = 0f
    private var chartKey = -1
    private var readT = 0f
    private var readEn: Boolean? = null
    private var redPulse = 0f
    private val SPS = floatArrayOf(3f, 12f, 45f)          // optimiser steps per second

    private val surface: LossSurface get() = race.surface

    // ================================================================ nodes
    private val terrains = ArrayList<DescentGfx.Terrain>()
    private val minis = ArrayList<DescentGfx.Terrain>()
    private lateinit var surfNode: Node
    private lateinit var curtainNode: Node
    private lateinit var floorNode: Node
    private val minRings = ArrayList<Node>()
    private lateinit var startPin: Node; private lateinit var startPost: Node; private lateinit var startRing: Node
    private val balls = ArrayList<Node>(); private val glows = ArrayList<Node>()
    private val shafts = ArrayList<Node>(); private val heads = ArrayList<Node>()
    private val dots = Array(3) { ArrayList<Node>() }; private val segs = Array(3) { ArrayList<Node>() }
    private val shelf = ArrayList<Node>(); private val shelfBase = ArrayList<Node>()
    private val curves = ArrayList<Node>(); private val curveMesh = ArrayList<MeshData>()
    private lateinit var chartMark: Node
    private val chartXY = FloatArray(CHART_K * 2)
    private lateinit var unitCyl: MeshData

    private val shelfMatOn = Mat.Pbr(Rgb(0.30f, 0.26f, 0.16f), metallic = 0.6f, roughness = 0.35f, emissive = scaled(GOLD, 0.35f))
    private val shelfMatOff = Mat.Pbr(Rgb(0.10f, 0.11f, 0.13f), metallic = 0.3f, roughness = 0.5f)

    // ================================================================ build
    override fun buildSpec(cacheDir: File): Node {
        // REBUILD IDEMPOTENCE: every field back to its start value
        race.active.fill(true); race.noise = 0.0; race.setSurface(Surfaces.all[0])
        surfIdx = 0; speedIdx = 1; playing = true; stepAcc = 0f; hop.fill(1f); dragging = false
        shownSurf = -1; trailKey.fill(-1); chartT = 0f; chartKey = -1; readT = 0f; readEn = null; redPulse = 0f
        listOf(minRings, balls, glows, shafts, heads, shelf, shelfBase, curves).forEach { it.clear() }
        dots.forEach { it.clear() }; segs.forEach { it.clear() }; curveMesh.clear(); terrains.clear(); minis.clear(); labels.clear()

        // ---- landscapes: heights + meshes now, textures from the bake cache (≈ 0.1 s on the Mac, ~1.5 s cold on the headset)
        for (s in Surfaces.all) terrains += DescentGfx.Terrain(s, BOARD, GRID_N).also { it.mesh = DescentGfx.gridMesh(it, tw = TEX); it.curtain = DescentGfx.curtainMesh(it) }
        for (s in Surfaces.all) minis += DescentGfx.Terrain(s, MINI, 17).also { it.mesh = DescentGfx.gridMesh(it, tw = TEX) }
        // light (board-local) from the upper left and the viewer's side: valley walls facing the viewer are lit, the others shaded
        val local = P3(-0.45f, 0.70f, 0.55f).let { val l = it.length; floatArrayOf(it.x / l, it.y / l, it.z / l) }
        val baked = BakeCache.load(cacheDir, "descent", BAKE_VERSION) {
            terrains.map { DescentGfx.bakeSurface(it, TEX, local, gridStep(it.s)) } + terrains.map { DescentGfx.bakeFloor(it, FLOOR_TEX) }
        }
        for ((k, t) in terrains.withIndex()) { t.tex = baked[k]; t.floor = baked[terrains.size + k]; minis[k].tex = t.tex }

        val root = Node("DescentRoot")
        unitCyl = MeshGen.cylinder(1f, 1f, 8)

        // bench: dark granite slab with an aluminium edge, like the physics benches
        root.add("Plinth", MeshGen.box(1.78f, 0.03f, 0.86f), Mat.Pbr(Rgb(1f, 1f, 1f), roughness = 0.55f, tex = slabTex), P3(0f, -0.235f, 0.0f))
        root.add("PlinthBand", MeshGen.box(1.79f, 0.007f, 0.87f), alu, P3(0f, -0.235f, 0.0f))

        // ---- the board (tilted): base slab, aluminium rim, floor contour map, the landscape
        val board = root.add(Node("Board")).also { it.pos = BC; it.rot = tiltQ() }
        val S = BOARD.size
        board.add("BoardBase", MeshGen.box(S + 0.03f, 0.014f, S + 0.03f), Mat.Pbr(Rgb(0.07f, 0.075f, 0.09f), metallic = 0.2f, roughness = 0.5f), P3(0f, -0.009f, 0f))
        board.add("BoardRim", MeshGen.box(S + 0.036f, 0.006f, S + 0.036f), alu, P3(0f, -0.014f, 0f))
        floorNode = board.add("Floor", OptGeo.quad(S, S), null, P3(0f, 0.0015f, 0f)).also { it.rot = Q.axisAngle(1f, 0f, 0f, -Math.PI.toFloat() / 2) }
        surfNode = board.add("Surface", terrains[0].mesh, Mat.Unlit(Rgb(1f, 1f, 1f), tex = terrains[0].tex))
        curtainNode = board.add("Curtain", terrains[0].curtain, add(Rgb(0.50f, 0.72f, 1.0f), 0.55f, DescentGfx.curtainTex()))
        val ringMesh = MeshGen.annulus(0.0095f, 0.0135f, 28)
        repeat(4) { k -> minRings += board.add("MinRing_$k", ringMesh, add(Rgb(0.55f, 1.0f, 0.75f), 0.9f)) }
        // easel posts from the plinth to the board's corners
        for ((k, c) in listOf(-1f to -1f, 1f to -1f, -1f to 1f, 1f to 1f).withIndex()) {
            val top = toWorld(P3(c.first * (S / 2 - 0.02f), -0.016f, c.second * (S / 2 - 0.02f)))
            val bot = P3(top.x, -0.22f, top.z)
            val post = root.add("Post_$k", unitCyl, alu)
            DescentGfx.placeLine(post, bot, top, 0.007f)
        }

        // ---- start pin (gold): pinch = restart from the default start, pinch-drag = move the start
        startPost = root.add("StartPost", unitCyl, Mat.Pbr(GOLD, metallic = 1f, roughness = 0.3f))
        startPin = root.add("StartPin", MeshGen.sphere(0.012f, 12, 16, MeshGen.SphereUv.EQUIRECT), Mat.Pbr(GOLD, metallic = 0.6f, roughness = 0.25f, emissive = scaled(GOLD, 0.45f)))
        startPin.tapRadius = 0.05f
        startRing = root.add("StartRing", MeshGen.annulus(0.010f, 0.016f, 28), add(GOLD, 0.9f)).also { it.rot = tiltQ() }

        // ---- balls, glows, gradient arrows
        val ballMesh = MeshGen.sphere(BALL_R, 14, 20, MeshGen.SphereUv.EQUIRECT)
        val quad = OptGeo.quad(1f, 1f)
        val cone = DescentGfx.cone(0.011f, 0.028f)
        for (k in 0 until 3) {
            balls += root.add("Ball_$k", ballMesh, ballOn[k]).also { it.tapRadius = 0.045f }
            glows += root.add("Glow_$k", quad, glowOn[k]).also { it.billboard = true; it.uniformScale(0.095f) }
            shafts += root.add("Shaft_$k", unitCyl, arrowMat[k])
            heads += root.add("Head_$k", cone, arrowMat[k])
        }
        // ---- trails: pooled dots + segments (transforms only, never a mesh edit)
        val dotMesh = MeshGen.sphere(0.0062f, 5, 8, MeshGen.SphereUv.EQUIRECT)
        for (k in 0 until 3) for (i in 0 until SLOTS) {
            dots[k] += root.add("Dot_${k}_$i", dotMesh, dotMat[k][0]).also { it.visible = false }
            if (i < RECENT) segs[k] += root.add("Seg_${k}_$i", unitCyl, segMat[k][0]).also { it.visible = false }
        }

        // ---- landscape shelf (right): five mini landscapes, pinch one to load it
        for (k in Surfaces.all.indices) {
            val c = shelfPos(k)
            shelfBase += root.add("ShelfBase_$k", MeshGen.box(MINI.size + 0.016f, 0.010f, MINI.size + 0.016f), shelfMatOff, c).also { it.rot = tiltQ() }
            shelf += root.add("Shelf_$k", minis[k].mesh, Mat.Unlit(Rgb(1f, 1f, 1f), tex = minis[k].tex), c).also { it.rot = tiltQ(); it.tapRadius = 0.06f }
            // a short aluminium stem under each tile, down to the shelf's rail
            root.add("ShelfStem_$k", unitCyl, alu).also { DescentGfx.placeLine(it, P3(c.x, -0.22f, c.z - 0.03f), P3(c.x, c.y - 0.03f, c.z - 0.03f), 0.0045f) }
        }

        // ---- dock (left, under the chart): where switched-off balls wait
        val padMesh = MeshGen.cylinder(0.030f, 0.020f, 24)
        val padRing = MeshGen.annulus(0.020f, 0.027f, 28)
        for (k in 0 until 3) {
            root.add("Dock_$k", padMesh, Mat.Pbr(Rgb(0.12f, 0.13f, 0.15f), metallic = 0.5f, roughness = 0.4f), dockPos(k) + P3(0f, -0.028f, 0f))
            root.add("DockRing_$k", padRing, add(BALL_C[k], 0.8f), dockPos(k) + P3(0f, -0.0175f, 0f))
        }
        root.add("DockPost", unitCyl, alu).also { DescentGfx.placeLine(it, P3(-0.69f, -0.22f, 0.15f), P3(-0.69f, -0.165f, 0.15f), 0.012f) }
        root.add("DockBar", MeshGen.box(0.34f, 0.008f, 0.07f), Mat.Pbr(Rgb(0.10f, 0.11f, 0.13f), metallic = 0.4f, roughness = 0.45f), P3(-0.69f, -0.168f, 0.15f))

        // ---- loss chart (left): log10(loss − min) against step, one curve per ball, gold line = 1e-3
        val chart = root.add(Node("Chart")).also { it.pos = CHART_C; it.rot = Q.axisAngle(0f, 1f, 0f, CHART_YAW) }
        val tw = 288; val th = 216
        val (px0, px1, py0, py1) = intArrayOf(26, tw - 10, 10, th - 22).let { listOf(it[0], it[1], it[2], it[3]) }
        val tRow = py0 + (LOG_HI - (-3f)) / (LOG_HI - LOG_LO) * (py1 - py0)
        chart.add("ChartPlate", OptGeo.quad(CHART_W, CHART_H), Mat.Unlit(Rgb(1f, 1f, 1f),
            tex = DescentGfx.chartTex(tw, th, (LOG_HI - LOG_LO).roundToInt(), tRow, px0, px1, py0, py1)), P3(0f, 0f, 0f))
        chart.add("ChartBack", MeshGen.box(CHART_W + 0.016f, CHART_H + 0.016f, 0.008f), alu, P3(0f, 0f, -0.006f))
        chart.add("ChartPost", unitCyl, alu).also { DescentGfx.placeLine(it, P3(0f, -0.22f - CHART_C.y, -0.012f), P3(0f, -CHART_H / 2, -0.012f), 0.007f) }
        plotX0 = (px0.toFloat() / tw - 0.5f) * CHART_W; plotX1 = (px1.toFloat() / tw - 0.5f) * CHART_W
        plotY1 = (0.5f - py0.toFloat() / th) * CHART_H; plotY0 = (0.5f - py1.toFloat() / th) * CHART_H
        for (k in 0 until 3) {
            val md = OptGeo.grid(2, CHART_K)
            DescentGfx.polylineInto(md, chartXY, 1, 0.005f, 0.003f + 0.001f * k)
            curveMesh += md
            curves += chart.add("Curve_$k", md, add(BALL_C[k], 0.95f))
        }
        chartMark = chart.add("ChartMark", OptGeo.quad(0.0025f, plotY1 - plotY0), add(GOLD, 0.55f), P3(0f, (plotY0 + plotY1) / 2, 0.002f))
        return root
    }

    private var plotX0 = 0f; private var plotX1 = 0f; private var plotY0 = 0f; private var plotY1 = 0f
    private const val BAKE_VERSION = 4

    /** Coordinate-grid spacing (landscape units) drawn faintly on the surface. */
    private fun gridStep(s: LossSurface) = if (s.x1 - s.x0 > 6) 2.0 else 1.0

    // ================================================================ per frame
    override fun update(dt: Float) {
        if (shownSurf != surfIdx) applySurface()
        // optimiser steps (lockstep), at the chosen speed
        if (playing && !dragging && !race.allDone) {
            stepAcc += dt * SPS[speedIdx]
            var n = 0
            while (stepAcc >= 1f && !race.allDone && n < 8) { stepOnce(); stepAcc -= 1f; n++ }
            if (n == 8) stepAcc = 0f
        } else stepAcc = 0f
        // a hop lasts 2/3 of a step at the slow and medium speeds; at the fast one it is a jump
        val hopRate = if (dt == 0f) 1e9f else SPS[speedIdx] * 1.5f
        for (k in 0 until 3) hop[k] = (hop[k] + dt * hopRate).coerceAtMost(1f)
        redPulse += dt

        for (k in 0 until 3) placeBall(k)
        placeStart()
        for (k in 0 until 3) if (trailKey[k] != trailCode(k)) { trailKey[k] = trailCode(k); placeTrail(k) }
        chartT += dt
        if (chartT >= 0.2f || dt == 0f) { chartT = 0f; updateChart() }
        readT += dt
        if (readT >= 0.1f || dt == 0f || readoutText.isEmpty() || readEn != Lang.en) { readT = 0f; readEn = Lang.en; updateReadout() }
        updateLabels()
    }

    private fun stepOnce() {
        val hit0 = BooleanArray(3) { race.racers[it].firstHit >= 0 }
        val div0 = BooleanArray(3) { race.racers[it].diverged }
        race.step()
        for (k in 0 until 3) if (race.active[k]) hop[k] = 0f
        // one optimiser step (all racers in lockstep): a quiet tick, softer at the fast speed
        Sfx.play(Sfx.TICK, gain = if (speedIdx >= 2) 0.10f else 0.16f)
        for (k in 0 until 3) {
            if (!race.active[k] || k >= balls.size) continue
            val r = race.racers[k]; val pan = (balls[k].pos.x / 0.9f).coerceIn(-1f, 1f)
            // this ball has just reached the minimum: a chime, pitched per optimiser (SGD low … Adam high)
            if (!hit0[k] && r.firstHit >= 0) Sfx.play(Sfx.CHIME, gain = 0.5f, pitch = 0.9f + 0.12f * k, pan = pan)
            // this ball has just diverged (flew off the surface): a low thud
            if (!div0[k] && r.diverged) Sfx.play(Sfx.THUD, gain = 0.6f, pitch = 0.8f, pan = pan)
        }
    }

    private fun trailCode(k: Int) = if (!race.active[k]) -2 else race.racers[k].steps * 4 + (if (race.racers[k].diverged) 1 else 0) + shownSurf * 100_000

    private fun applySurface() {
        shownSurf = surfIdx
        val t = terrains[surfIdx]
        surfNode.setMesh(t.mesh)
        curtainNode.setMesh(t.curtain)
        surfNode.setMat(Mat.Unlit(Rgb(1f, 1f, 1f), tex = t.tex))
        floorNode.setMat(Mat.Unlit(Rgb(1f, 1f, 1f), tex = t.floor))
        val mins = t.s.minima
        for ((k, r) in minRings.withIndex()) {
            r.visible = k < mins.size
            if (k < mins.size) r.pos = t.local(mins[k][0], mins[k][1], 0.0015f)
        }
        for ((k, n) in shelfBase.withIndex()) n.setMat(if (k == surfIdx) shelfMatOn else shelfMatOff)
        for ((k, n) in shelf.withIndex()) {
            val lift = if (k == surfIdx) 0.024f else 0.006f                 // the loaded one stands out toward the viewer
            n.pos = shelfPos(k) + P3(0f, lift * CT, lift * ST)
        }
        trailKey.fill(-1); chartKey = -1
    }

    // ---- balls
    private fun ballLocal(k: Int): P3 {
        val r = race.racers[k]; val t = terrains[surfIdx]
        val i = r.steps
        val p1 = t.local(r.hx[i], r.hy[i], BALL_R)
        if (i == 0 || hop[k] >= 1f) return p1
        val p0 = t.local(r.hx[i - 1], r.hy[i - 1], BALL_R)
        val s = ProcTex.smooth01(hop[k])
        val d = (p1 - p0).length
        val arc = 4f * s * (1 - s) * minOf(0.05f, 0.35f * d)
        return P3(p0.x + (p1.x - p0.x) * s, p0.y + (p1.y - p0.y) * s + arc, p0.z + (p1.z - p0.z) * s)
    }

    private val gBuf = DoubleArray(2)

    private fun placeBall(k: Int) {
        val r = race.racers[k]; val ball = balls[k]; val gl = glows[k]
        if (!race.active[k]) {
            setPos(ball, dockPos(k)); ball.setMat(ballOff[k]); gl.visible = false
            shafts[k].visible = false; heads[k].visible = false
            return
        }
        val lp = ballLocal(k)
        val wp = toWorld(lp)
        setPos(ball, wp); setPos(gl, wp); gl.visible = true
        if (r.diverged && hop[k] >= 1f) {
            val on = ((redPulse * 3f).toInt() % 2) == 0
            ball.setMat(if (on) ballRed else ballOn[k]); gl.setMat(if (on) glowRed else glowOn[k])
            shafts[k].visible = false; heads[k].visible = false
            return
        }
        ball.setMat(ballOn[k]); gl.setMat(glowOn[k])
        // −∇f arrow, tangent to the drawn surface, length ∝ √(|∇f| / |∇f at the start|)
        val t = terrains[surfIdx]; val s = t.s
        r.gradHere(s, gBuf)
        val gn = hypot(gBuf[0], gBuf[1])
        val g0 = startGrad()
        if (gn < 1e-9 || gn < 1e-4 * g0) { shafts[k].visible = false; heads[k].visible = false; return }
        val x = r.theta[0]; val y = r.theta[1]
        val dl = 0.004 * (s.x1 - s.x0)
        val a = t.local(x, y); val b = t.local(x - dl * gBuf[0] / gn, y - dl * gBuf[1] / gn)
        var tx = b.x - a.x; var ty = b.y - a.y; var tz = b.z - a.z
        val tl = sqrt(tx * tx + ty * ty + tz * tz).coerceAtLeast(1e-6f); tx /= tl; ty /= tl; tz /= tl
        val len = (0.032f + 0.075f * sqrt((gn / g0).toFloat().coerceAtMost(1.6f))).coerceAtMost(0.125f)
        val base = P3(lp.x + tx * BALL_R, lp.y + ty * BALL_R, lp.z + tz * BALL_R)
        val neck = P3(base.x + tx * (len - 0.026f), base.y + ty * (len - 0.026f), base.z + tz * (len - 0.026f))
        val wNeck = toWorld(neck)
        line(shafts[k], toWorld(base), wNeck, 0.0042f)
        val wd = P3(tx, ty * CT - tz * ST, ty * ST + tz * CT)
        if (heads[k].pos != wNeck) { heads[k].pos = wNeck; heads[k].rot = Q.fromTo(0f, 1f, 0f, wd.x, wd.y, wd.z) }
        shafts[k].visible = true; heads[k].visible = true
    }

    private var g0Key = ""; private var g0Val = 1.0
    /** |∇f| at the current start (the arrow's reference length). */
    private fun startGrad(): Double {
        val key = "${surface.key}:${race.startX}:${race.startY}"
        if (key != g0Key) {
            g0Key = key; surface.grad(race.startX, race.startY, gBuf)
            g0Val = hypot(gBuf[0], gBuf[1]).coerceAtLeast(1e-3 * (1 + surface.noiseScale))
        }
        return g0Val
    }

    private fun placeStart() {
        val t = terrains[surfIdx]
        val foot = t.local(race.startX, race.startY, 0.002f)
        val top = P3(foot.x, foot.y + 0.062f, foot.z)
        line(startPost, toWorld(foot), toWorld(top), 0.0028f)
        setPos(startPin, toWorld(P3(top.x, top.y + 0.010f, top.z)))
        setPos(startRing, toWorld(foot))
    }

    // ---- trails
    // Ring-buffer slots, so a step moves ONE dot and ONE segment per ball (the adapter re-sends only changed nodes):
    // the latest RECENT points sit in slot h mod RECENT, older history in the remaining OLD slots at a power-of-two
    // stride (changes only when the stride doubles).
    private const val OLD = SLOTS - RECENT

    private fun placeTrail(k: Int) {
        val ds = dots[k]; val ss = segs[k]
        if (!race.active[k]) { ds.forEach { it.visible = false }; ss.forEach { it.visible = false }; return }
        val r = race.racers[k]; val t = terrains[surfIdx]
        val cnt = r.count
        val lo = maxOf(0, cnt - RECENT)
        fun at(h: Int) = toWorld(t.local(r.hx[h], r.hy[h], 0.006f))
        for (slot in 0 until RECENT) {
            val h = lo + ((slot - lo % RECENT) % RECENT + RECENT) % RECENT
            val d = ds[slot]; val sg = ss[slot]
            if (h >= cnt) { d.visible = false; sg.visible = false; continue }
            val w = at(h)
            d.visible = true; setPos(d, w)
            val age = cnt - 1 - h
            val lvl = if (age < 10) 0 else if (age < 25) 1 else 2
            d.setMat(dotMat[k][lvl])
            sg.visible = h >= 1 && h - 1 >= lo
            if (sg.visible) { line(sg, at(h - 1), w, 0.0028f); sg.setMat(segMat[k][lvl]) }
        }
        // older history: every s-th point, s = the smallest power of two that fits OLD slots
        var s = 1; while ((lo + s - 1) / s > OLD) s *= 2
        val m = (lo + s - 1) / s
        for (j in 0 until OLD) {
            val d = ds[RECENT + j]
            if (j >= m) { d.visible = false; continue }
            d.visible = true; setPos(d, at(j * s)); d.setMat(dotMat[k][2])
        }
    }

    /** Assign only real changes: every assignment bumps the node's version and costs a device transform update. */
    private fun setPos(n: Node, p: P3) { if (n.pos != p) n.pos = p }

    /** [DescentGfx.placeLine], change-aware. */
    private fun line(n: Node, a: P3, b: P3, r: Float) {
        val dx = b.x - a.x; val dy = b.y - a.y; val dz = b.z - a.z
        val len = sqrt(dx * dx + dy * dy + dz * dz).coerceAtLeast(1e-4f)
        val p = P3((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2); val sc = P3(r, len, r)
        if (n.pos == p && n.scale == sc) return
        n.pos = p; n.rot = Q.fromTo(0f, 1f, 0f, dx, dy, dz); n.scale = sc
    }

    // ---- loss chart
    private fun updateChart() {
        val key = race.steps * 8 + race.active.withIndex().sumOf { (i, a) -> if (a) 1 shl i else 0 } + shownSurf * 1_000_000
        if (key == chartKey) return
        chartKey = key
        val xMax = chartRange()
        val fm = surface.fMin
        for (k in 0 until 3) {
            val r = race.racers[k]
            curves[k].visible = race.active[k] && r.count >= 1
            if (!curves[k].visible) continue
            val cnt = r.count
            val m = minOf(cnt, CHART_K)
            for (j in 0 until m) {
                val h = if (cnt <= CHART_K) j else (j.toLong() * (cnt - 1) / (CHART_K - 1)).toInt()
                val lv = if (r.diverged && h == cnt - 1) LOG_HI else log10((r.hf[h] - fm).coerceAtLeast(1e-6)).toFloat()
                chartXY[j * 2] = plotX0 + (plotX1 - plotX0) * (h.toFloat() / xMax).coerceAtMost(1f)
                chartXY[j * 2 + 1] = plotY0 + (plotY1 - plotY0) * ((lv - LOG_LO) / (LOG_HI - LOG_LO)).coerceIn(0f, 1f)
            }
            DescentGfx.polylineInto(curveMesh[k], chartXY, m, 0.005f, 0.003f + 0.001f * k)
            meshEdited(curveMesh[k])
        }
        // gold tick at step 200 (the challenge's budget) once the axis is wider than that
        chartMark.visible = xMax > 200
        if (chartMark.visible) chartMark.pos = P3(plotX0 + (plotX1 - plotX0) * 200f / xMax, (plotY0 + plotY1) / 2, 0.002f)
    }

    /** Steps shown on the chart's x axis: 50, then doubling (100, 200, 400 …). */
    fun chartRange(): Int { var x = 50; while (x < race.steps) x *= 2; return x }

    // ================================================================ text
    private val NAMES = listOf("SGD" to "SGD", "动量" to "Momentum", "Adam" to "Adam")  // i18n-ok (zh/en pairs)
    private fun ballName(k: Int) = tr(NAMES[k].first, NAMES[k].second)
    private val SURF_NAMES = listOf("碗形峡谷（二次型）" to "Quadratic ravine", "Rosenbrock 香蕉谷" to "Rosenbrock banana valley",  // i18n-ok
        "Himmelblau 四谷" to "Himmelblau, 4 valleys", "鞍点山口" to "Saddle pass", "真实模型：逻辑回归" to "Real model: logistic regression")  // i18n-ok
    private val SHELF_NAMES = listOf("碗形" to "Bowl", "香蕉谷" to "Banana", "四谷" to "4 valleys", "鞍点" to "Saddle", "逻辑回归" to "Logistic")  // i18n-ok
    fun surfName(k: Int = surfIdx) = tr(SURF_NAMES[k].first, SURF_NAMES[k].second)

    private fun fmtLr(lr: Double) = if (lr >= 1.0) "1" else lr.toString()
    private fun fmtLoss(v: Double): String = when {
        !v.isFinite() -> "—"
        v >= 1000 -> "%.0f".format(v)
        v >= 0.01 -> "%.3g".format(v)
        v <= 0.0 -> "0"
        else -> { val e = kotlin.math.floor(log10(v)).toInt(); "%.1fe%d".format(v / Math.pow(10.0, e.toDouble()), e) }   // 4.7e-4
    }
    private fun noiseText() = if (surface === Surfaces.LOGISTIC) tr("小批量 %d/40", "minibatch %d/40").format(Logistic.batchSize(race.noise))
                              else if (race.noise == 0.0) "0" else tr("%d%%（高斯）", "%d%% (Gaussian)").format((race.noise * 100).roundToInt())   // a stand-in for minibatch noise

    /** 2/λmax at the minimum nearest the SGD ball: plain GD is stable near there iff η is below it. */
    private fun stableLr(): Double {
        val r = race.racers[0]
        val m = surface.nearestMin(r.theta[0], r.theta[1])
        return 2.0 / surface.lambdaMax(m[0], m[1])
    }

    private fun updateReadout() {
        val l1 = tr("%s · 学习率 η = %s · 噪声 %s · 谷底处 SGD 稳定需 η < 2/λmax = %s", "%s · learning rate η = %s · noise %s · SGD stable at the bottom if η < 2/λmax = %s")
            .format(surfName(), fmtLr(race.lr), noiseText(), "%.2g".format(stableLr()))
        val l2 = (0 until 3).joinToString("   ·   ") { k ->
            val r = race.racers[k]
            when {
                !race.active[k] -> tr("%s（未参赛）", "%s (off)").format(ballName(k))
                r.diverged -> tr("%s 第 %d 步发散！", "%s diverged at step %d!").format(ballName(k), r.steps)
                else -> tr("%s 第 %d 步 损失 %s%s", "%s step %d loss %s%s").format(ballName(k), r.steps, fmtLoss(r.loss), if (r.firstHit >= 0) " ✓" else "")
            }
        }
        readoutText = "$l1\n$l2\n${insight()}"
    }

    /** One line saying what the user is looking at (state-dependent). */
    fun insight(): String {
        val rs = race.racers
        val sgd = rs[0]; val mom = rs[1]; val adam = rs[2]
        val a = race.active
        val lam = surface.nearestMin(sgd.theta[0], sgd.theta[1]).let { surface.lambdaMax(it[0], it[1]) }
        // curvature where SGD took off (its last point on the landscape): the local η·λmax decides
        val lamOff = if (sgd.diverged && sgd.steps >= 1) surface.lambdaMax(sgd.hx[sgd.steps - 1], sgd.hy[sgd.steps - 1]) else lam
        return when {
            a[0] && sgd.diverged && surface === Surfaces.BOWL && race.lr * lam > 2 ->
                tr("η·λmax = %.1f > 2：每步都冲过谷底、越弹越远（误差每步 ×|1 − ηλ| = %.1f）→ 发散", "η·λmax = %.1f > 2: each step overshoots the valley by more (error ×|1 − ηλ| = %.1f per step) → diverges")
                    .format(race.lr * lam, race.lr * lam - 1)
            a[0] && sgd.diverged && race.lr * lamOff > 2 ->
                tr("起跳处 η·λmax = %.1f > 2：一步就冲过谷底、飞出曲面 → 发散。试试把 η 调小", "Where it jumped, η·λmax = %.1f > 2: one step overshoots the valley and leaves the surface → diverged. Try a smaller η")
                    .format(race.lr * lamOff)
            a[0] && sgd.diverged -> tr("SGD 在陡坡上一步迈得太大（|∇f| 很大时 η·|∇f| 也很大），直接飞出了曲面", "SGD took one huge step on a steep wall (η·|∇f| is big where |∇f| is big) and left the surface")
            a[1] && mom.diverged -> tr("动量把速度也带过了头：重球的稳定上限是 η < 2(1+β)/λmax，比 SGD 宽，但不是无限", "Momentum overshoots with its velocity: a heavy ball is stable for η < 2(1+β)/λmax — wider than SGD, not unlimited")
            surface === Surfaces.SADDLE && a[0] && hypot(sgd.theta[0], sgd.theta[1]) < 0.2 && sgd.steps > 0 ->
                tr("鞍点：梯度≈0 却不是最低点。SGD 在这里磨蹭；动量和 Adam（或一点噪声）把球推了下去", "A saddle: ∇f ≈ 0 but not a minimum. SGD dawdles here; momentum, Adam or a little noise push the ball off")
            surface === Surfaces.HIMMELBLAU -> tr("四个谷底：从哪出发、怎么走，决定落进哪个谷 —— 非凸地形没有唯一答案", "Four valleys: where you start and how you move decides which one you end in — non-convex, no single answer")
            surface === Surfaces.LOGISTIC -> tr("真实模型：逻辑回归的 2 个权重、40 个样本；特征 2 没有归一化，所以是峡谷。噪声 = 真的小批量", "A real model: logistic regression, 2 weights, 40 samples; feature 2 is not normalised, hence the ravine. Noise = real minibatches")
            surface === Surfaces.ROSENBROCK && a[2] && !adam.diverged && a[0] && sgd.diverged.not() && race.lr < 0.004 ->
                tr("η 小到 SGD 不发散时，它在平坦的弯谷里几乎挪不动；动量会攒速度", "With η small enough for SGD, it barely moves along the flat curved valley; momentum builds up speed")
            surface === Surfaces.ROSENBROCK -> tr("香蕉谷：先掉进谷里，再沿几乎平坦的弯谷慢慢挪向 (1, 1)；Adam 按每个方向的坡度自动缩放步长", "Banana valley: drop into the valley, then crawl along its flat curve to (1, 1); Adam rescales its step per direction")
            a[0] && race.lr * lam > 1 -> tr("η > 1/λmax：SGD 在陡的方向来回横跳（之字形），在平缓的方向慢慢爬", "η > 1/λmax: SGD zig-zags across the steep direction while it creeps along the gentle one")
            a[1] && mom.firstHit > 0 && (!a[0] || sgd.firstHit < 0 || mom.firstHit < sgd.firstHit) ->
                tr("动量先到：它把过去的梯度累加成速度，沿平缓的谷底越滚越快 —— 像更重的球", "Momentum arrives first: it adds past gradients into a velocity and speeds up along the gentle floor — a heavier ball")
            race.lr * lam < 0.05 -> tr("η 太小：每步只挪一点点，要很多步才到谷底", "η too small: every step moves a tiny bit — it takes ages to reach the bottom")
            else -> tr("小球 = 模型参数，高度 = 损失；每一步沿 −∇f 走 η·|∇f| —— 这就是训练", "Ball = the model's parameters, height = loss; each step goes η·|∇f| along −∇f — that is training")
        }
    }

    private fun updateLabels() {
        labels.clear()
        val corner = toWorld(P3(-BOARD.size / 2 + 0.17f, BOARD.gap + BOARD.height + 0.03f, -BOARD.size / 2))
        labels += LabelSpec(tr("%s · 高度 ∝ log(1+损失)", "%s · height ∝ log(1 + loss)").format(surfName()), P3(corner.x, minOf(corner.y + 0.04f, 0.44f), corner.z))
        labels += LabelSpec(tr("曲面架 · 捏一个换地形", "Landscape shelf · pinch one"), P3(0.6625f, 0.335f, 0.04f))
        labels += LabelSpec(tr("损失曲线（对数）· 金线 = 1e-3", "Loss curves (log) · gold = 1e-3"), P3(CHART_C.x, CHART_C.y + CHART_H / 2 + 0.045f, CHART_C.z))
        for (k in 0 until 3) {
            val r = race.racers[k]
            val p = balls[k].pos
            val txt = when {
                !race.active[k] -> tr("%s（关）", "%s (off)").format(ballName(k))
                r.diverged -> tr("%s 发散！", "%s diverged!").format(ballName(k))
                r.firstHit >= 0 -> "${ballName(k)} ✓"
                else -> ballName(k)
            }
            val dx = if (race.active[k]) (k - 1) * 0.075f else 0f
            labels += LabelSpec(txt, P3(p.x + dx, p.y + 0.055f, p.z + 0.01f))
        }
    }

    // ================================================================ actions (panel, pinch, lesson)
    fun selectSurface(k: Int) {
        surfIdx = k.coerceIn(0, Surfaces.all.size - 1)
        race.setSurface(Surfaces.all[surfIdx]); restartRun()
    }
    fun setLr(lr: Double) { val v = LR_STEPS[lrIndexOf(lr)]; if (v != race.lr) { race.lr = v; restartRun() } }
    fun setLrIndex(i: Int) = setLr(LR_STEPS[i.coerceIn(0, LR_STEPS.size - 1)])
    val lrIndex: Int get() = lrIndexOf(race.lr)
    fun setNoise(v: Double) { val q = (v * 20).roundToInt() / 20.0; if (q != race.noise) { race.noise = q; restartRun() } }
    fun setSpeed(k: Int) { speedIdx = k.coerceIn(0, 2) }
    fun setBall(k: Int, on: Boolean) { if (race.active[k] == on) return; race.active[k] = on; if (on) restartRun() else trailKey[k] = -1 }
    /** Back to the start (same settings) and run. */
    fun restartRun() { race.reset(); hop.fill(1f); stepAcc = 0f; playing = true; trailKey.fill(-1); chartKey = -1 }
    fun resetStart() { race.setStart(surface.startX, surface.startY); restartRun() }
    fun togglePlay() { if (race.allDone) restartRun() else playing = !playing }
    fun stepOnceManual() { playing = false; if (!race.allDone) { stepOnce(); hop.fill(1f); for (k in 0 until 3) if (race.active[k]) hop[k] = 0f } }

    override val tapTargets get() = (0 until 3).map { k -> TapTarget("Ball_$k", NAMES[k].first + " → 开/关", NAMES[k].second + " → on/off") { race.active[k] } } +  // i18n-ok
        TapTarget("StartPin", "起点 → 重来", "Start → restart") +
        SHELF_NAMES.mapIndexed { k, n -> TapTarget("Shelf_$k", n.first, n.second) { surfIdx == k } }

    override fun isTarget(name: String) = name.startsWith("Ball_") || name == "StartPin" || name.startsWith("Shelf_")

    override fun onTap(name: String) {
        when {
            name.startsWith("Ball_") -> name.removePrefix("Ball_").toIntOrNull()?.let { setBall(it, !race.active[it]) }
            name == "StartPin" -> resetStart()
            name.startsWith("Shelf_") -> name.removePrefix("Shelf_").toIntOrNull()?.let { selectSurface(it) }
        }
    }

    override fun tapMessage(name: String): String? = when {
        name.startsWith("Ball_") -> name.removePrefix("Ball_").toIntOrNull()?.let { k ->
            if (race.active[k]) tr("%s 上场 —— 全体从起点重新出发", "%s joins — everyone restarts from the start").format(ballName(k))
            else tr("%s 退场，回到左边的球座", "%s leaves the race, back to its dock").format(ballName(k))
        }
        name == "StartPin" -> tr("回到默认起点，重新出发 —— 捏住它拖动可以换起点", "Back to the default start — pinch and drag it to move the start")
        name.startsWith("Shelf_") -> tr("换成「%s」· η = %s", "Loaded “%s” · η = %s").format(surfName(), fmtLr(race.lr))
        else -> null
    }

    // ---- drag: move the start over the landscape (hand motion projected onto the tilted board)
    private const val DRAG_GAIN = 1.25f
    override fun isDragTarget(name: String) = name == "StartPin"
    override fun onDragStart(name: String) { dragging = true; dragX = race.startX; dragY = race.startY }
    override fun onDrag(name: String, dx: Float, dy: Float, dz: Float) {
        val l = toLocalDir(dx, dy, dz)
        val s = surface; val k = (s.x1 - s.x0) / BOARD.size
        val m = 0.02 * (s.x1 - s.x0)
        dragX = (dragX + l.x * k * DRAG_GAIN).coerceIn(s.x0 + m, s.x1 - m)
        dragY = (dragY - l.z * k * DRAG_GAIN).coerceIn(s.y0 + m, s.y1 - m)
        race.setStart(dragX, dragY); hop.fill(1f); trailKey.fill(-1); chartKey = -1
    }
    override fun onDragEnd(name: String) { dragging = false; restartRun() }

    override fun reviewVariants(): Map<String, () -> Unit> = mapOf(
        "rosen" to { selectSurface(1); setSpeed(2); repeat(240) { update(1f / 60f) } },
        "himmel" to { selectSurface(2); setSpeed(2); repeat(600) { update(1f / 60f) } },
        "diverge" to { selectSurface(0); setLr(0.3); setSpeed(1); repeat(40) { update(1f / 60f) } },
        "saddle" to { selectSurface(3); setSpeed(1); repeat(270) { update(1f / 60f) } },
        "logistic" to { selectSurface(4); setNoise(0.5); setSpeed(2); repeat(300) { update(1f / 60f) } },
        "noisy" to { selectSurface(1); setLr(0.0015); setNoise(1.0); setSpeed(2); repeat(300) { update(1f / 60f) } },   // longest readout
    )

    // ================================================================ panel
    @Composable
    override fun Panel() {
        LabPanel(readout) {
            LabSlider(tr("学习率 η", "Learning rate η"), lrIndex.toFloat(), 0f..(LR_STEPS.size - 1).toFloat(), "η = " + fmtLr(race.lr), width = 380) {
                setLrIndex(it.roundToInt())
            }
            LabSlider(tr("梯度噪声", "Gradient noise"), race.noise.toFloat(), 0f..1f, noiseText(), width = 300) { setNoise(it.toDouble()) }
            Button(onClick = { togglePlay() }) {
                Text(if (race.allDone) tr("↻ 再跑一次", "↻ Run again") else if (playing) tr("⏸ 暂停", "⏸ Pause") else tr("▶ 继续", "▶ Run"))
            }
            Button(onClick = { stepOnceManual() }, enabled = !race.allDone) { Text(tr("单步", "Step")) }
            Button(onClick = { restartRun() }) { Text(tr("↺ 重来", "↺ Restart")) }
            Spacer(Modifier.width(6.dp))
            SegmentControl {
                listOf(tr("慢", "Slow"), tr("中", "Med"), tr("快", "Fast")).forEachIndexed { k, l ->
                    SegmentItem(selected = speedIdx == k, onClick = { setSpeed(k) }, title = { Text(l) })
                }
            }
        }
    }

    // ================================================================ guided track (导学)
    // ---- background card (📖): context for a viewer who has never met this topic
    override val about = About(listOf(
        About.Page(
            Bi("这个实验在做什么？", "What is this experiment?"),
            Bi("中间是一块起伏的「地形」，三个发光小球从金色起点往下滚：白球 SGD、橙球「动量」、青球 Adam。面板上的「学习率 η」决定每步迈多大，「梯度噪声」让每步带点随机。右边架子可换 5 种地形，左边是三个球的损失曲线；捏住金色起点针拖动可以换出发点。", "In the middle is a hilly “landscape”; three glowing balls roll down from a gold start pin: white SGD, orange Momentum, cyan Adam. The panel's learning rate η sets how big each step is; gradient noise adds randomness. The shelf on the right swaps between 5 landscapes, the left shows each ball's loss curve, and you can pinch-drag the gold pin to move the start."),
        ),
        About.Page(
            Bi("先认识几个词", "A few words first"),
            Bi("模型里有许多可调的数，叫「参数」。「损失」是一个分数，衡量模型错得有多厉害：越低越好。给参数的每一种取值都算出损失，就得到一张「损失地形」；「训练」就是在这张地形上找最低点。真实网络有上百万个参数，地形就有上百万维；这里只取 2 个参数，才画得成你看得见的山谷。", "A model contains many adjustable numbers called “parameters”. The “loss” is a score for how wrong the model is: lower is better. Give every possible setting of the parameters its loss and you get a “loss landscape”; “training” means finding its lowest point. Real networks have millions of parameters, so millions of dimensions; here we keep just 2, so the valley is something you can see."),
        ),
        About.Page(
            Bi("背景：下山的老办法", "Background: an old way down"),
            Bi("1847 年，法国数学家柯西为了解天文学里的方程组，提出「梯度下降」：站在任意一点，算出哪个方向最陡，往下迈一小步，再重复。1964 年波利亚克加入「动量」，像一个更重的球；2014 年 Kingma 和 Ba 提出 Adam，给每个方向自动调步长。今天几乎所有神经网络都是这样训练的。", "In 1847 the French mathematician Cauchy, solving systems of equations from astronomy, proposed “gradient descent”: wherever you stand, find the steepest downhill direction, take a small step, repeat. In 1964 Polyak added “momentum”, like a heavier ball; in 2014 Kingma and Ba proposed Adam, which tunes the step for each direction. Almost every neural network today is trained this way."),
        ),
        About.Page(
            Bi("核心想法：蒙眼下山", "Key idea: downhill, blindfolded"),
            Bi("想象你蒙着眼站在山坡上，只能用脚感觉哪边往下最陡（这就是「梯度」），然后朝那边迈一步，步长就是学习率 η。步子太小，天黑也走不到谷底；步子太大，一脚跨过谷底冲上对面山坡，越弹越高（发散）。动量像一辆有惯性的小车，Adam 像在陡处自动收小步子。", "Imagine standing blindfolded on a hillside, feeling with your feet which way is steepest down (that is the “gradient”), then stepping that way; the step length is the learning rate η. Steps too small and you never reach the bottom; too big and you stride over the valley and up the other side, bouncing higher each time (divergence). Momentum is a cart that keeps rolling; Adam shortens its stride on steep ground."),
        ),
        About.Page(
            Bi("怎么看这个 3D 场景", "How to read the 3-D scene"),
            Bi("板上的两个水平方向 = 两个参数；高度和颜色 = 损失；绿色小环 = 真正的最低点。小球 = 模型此刻的参数，身后的光迹 = 走过的路，前面的箭头 = 下坡方向 −∇f；底下的等高线是从上往下看的地图。左边曲线的纵轴是对数刻度，金线 = 损失 0.001。", "The two horizontal directions on the board = the two parameters; height and colour = the loss; green rings = the true minima. A ball = the model's current parameters, its glowing trail = the path so far, its arrow = the downhill direction −∇f; the contour lines underneath are the map seen from above. On the left the loss curves use a log scale; the gold line = loss 0.001."),
        ),
        About.Page(
            Bi("关键数字 · 今天", "Key numbers · today"),
            Bi("在碗形峡谷上，陡方向的弯曲度 λ = 10，所以 η 必须小于 2/λ = 0.2，否则小球必然飞出去 —— 这不靠运气，是算得出来的。同样 η = 0.02，动量约 37 步到谷底，SGD 要 180 多步。今天训练聊天机器人、图像识别的大模型，用的多是 Adam 这一族方法，只是参数有数十亿个，而不是 2 个。", "On the quadratic ravine the steep direction has curvature λ = 10, so η must stay below 2/λ = 0.2 or a ball is bound to fly off — not luck, arithmetic. At the same η = 0.02 Momentum reaches the bottom in about 37 steps; SGD needs 180+. Today's chatbots and image recognisers are mostly trained with optimisers of the Adam family — just with billions of parameters instead of 2."),
        ),
    ))

    override val lesson: Lesson by lazy {
        Lesson(
            listOf(
                LessonStep(
                    say = Bi("学习率 η 就是每一步的步长。如果把它调得很大，小球会怎样？", "The learning rate η is the step length. What happens if you make it very large?"),
                    options = listOf(
                        Bi("更快到达谷底", "It reaches the bottom faster"),
                        Bi("来回震荡，甚至飞出去", "Bounces around, may fly off"),
                        Bi("没什么影响", "No real effect"),
                    ),
                    answer = 1,
                ),
                LessonStep(
                    say = Bi("在碗形峡谷上，把「学习率 η」滑块一格一格往右拉，直到有小球飞出曲面（发散）。", "On the quadratic ravine, push the “learning rate η” slider right, notch by notch, until a ball flies off the surface."),
                    target = "Ball_0",
                    onEnter = { if (surfIdx != 0) selectSurface(0) },
                    done = { surfIdx == 0 && race.racers.indices.any { race.active[it] && race.racers[it].diverged } },
                    autoplay = { selectSurface(0); setBall(0, true); setLr(0.3) },
                    reveal = true,
                    insight = Bi("误差每步乘以 |1 − ηλ|。这里陡方向 λ = 10：η < 0.2 时越来越小，η = 0.2 时原地来回弹，η > 0.2 每步都冲过谷底并且越冲越远。η 太小则 |1 − ηλ| ≈ 1，几乎不动。", "The error is multiplied by |1 − ηλ| each step. Here the steep direction has λ = 10: below η = 0.2 it shrinks, at 0.2 it bounces in place, above it every step overshoots further. A tiny η gives |1 − ηλ| ≈ 1: barely moving."),
                ),
                LessonStep(
                    say = Bi("同一个 η = 0.02，白球 SGD 和橙球「动量」从同一点出发。谁先到谷底？", "Same η = 0.02: the white SGD ball and the orange momentum ball start from the same point. Which reaches the bottom first?"),
                    options = listOf(
                        Bi("白球 SGD", "White: SGD"),
                        Bi("橙球 动量", "Orange: momentum"),
                        Bi("同时到", "A tie"),
                    ),
                    answer = 1,
                ),
                LessonStep(
                    say = Bi("看比赛（η = 0.02）：留意橙球怎样越滚越快，再看左边的损失曲线。", "Watch the race (η = 0.02): see how the orange ball speeds up, then look at the loss curves on the left."),
                    target = "Ball_1",
                    onEnter = { selectSurface(0); setNoise(0.0); setBall(0, true); setBall(1, true); setLr(0.02); restartRun() },
                    done = { surfIdx == 0 && race.active[1] && race.racers[1].firstHit > 0 },
                    autoplay = { selectSurface(0); setNoise(0.0); setBall(1, true); setLr(0.02); setSpeed(2) },
                    reveal = true,
                    insight = Bi("橙球约第 37 步就到了，SGD 要 180 多步。动量把过去的梯度累加成「速度」：在平缓的谷底方向越滚越快，在陡的方向来回抵消 —— 像一个更重的球。", "Orange arrives at about step 37, SGD needs 180+. Momentum adds past gradients into a velocity: it speeds up along the gentle floor while the side-to-side kicks cancel — a heavier ball."),
                ),
                LessonStep(
                    say = Bi("捏右边架子上的「香蕉谷」（Rosenbrock，η = 0.05）。三个球里谁能走下去？", "Pinch “Banana” on the shelf (Rosenbrock, η = 0.05). Which of the three balls makes it down?"),
                    target = "Shelf_1",
                    done = { surfIdx == 1 && race.lr == 0.05 && (race.steps >= 60 || race.allDone) },
                    autoplay = { onTap("Shelf_1"); setSpeed(2) },
                    insight = Bi("同一个 η：SGD 和动量一碰陡壁就被弹飞；Adam 用每个方向梯度的均方根给步长归一化，步子 ≈ η 与坡度无关，所以稳稳走进弯谷。", "Same η: SGD and momentum are flung off the steep wall; Adam divides each step by the running RMS of that direction's gradient, so its step is ≈ η whatever the slope — it walks into the curved valley."),
                ),
            ),
            challenge = Challenge(
                say = Bi("挑战：在香蕉谷上，200 步内把损失降到 1e-3 以下 —— 自选优化器（捏小球让它退场）和学习率。", "Challenge: on the banana valley, get the loss below 1e-3 within 200 steps — pick the optimiser (pinch balls out) and the learning rate."),
                progress = { challengeProgress() },
                done = { challengeSolved() },
                autoplay = { selectSurface(1); resetStart(); setNoise(0.0); setBall(1, true); setLr(0.003); setSpeed(2) },
            ),
        )
    }

    private fun challengeSolved() = surfIdx == 1 && race.startToMin >= 1.5 &&
        race.racers.indices.any { race.active[it] && race.racers[it].firstHit in 1..200 }

    private fun challengeProgress(): String {
        if (surfIdx != 1) return tr("先捏右边架子上的「香蕉谷」", "First pinch “Banana” on the shelf")
        if (race.startToMin < 1.5) return tr("起点离谷底太近 —— 捏金色起点针恢复默认", "The start is too close to the bottom — pinch the gold pin to reset it")
        val win = race.racers.indices.filter { race.active[it] && race.racers[it].firstHit in 1..200 }.minByOrNull { race.racers[it].firstHit }
        if (win != null) return tr("✓ %s，η = %s：第 %d 步降到 1e-3 以下", "✓ %s, η = %s: below 1e-3 at step %d").format(ballName(win), fmtLr(race.lr), race.racers[win].firstHit)
        val act = race.racers.indices.filter { race.active[it] }
        if (act.isEmpty()) return tr("至少让一个小球上场（捏左边球座上的球）", "Put at least one ball in the race (pinch it on the dock)")
        val best = act.minBy { race.racers[it].bestWithin(200) }
        val r = race.racers[best]
        return tr("200 步内最低：%s %s（已走 %d/200 步）· 目标 < 1e-3", "Lowest within 200 steps: %s %s (%d/200 steps run) · target < 1e-3")
            .format(ballName(best), fmtLoss(r.bestWithin(200)), minOf(r.steps, 200))
    }

}
