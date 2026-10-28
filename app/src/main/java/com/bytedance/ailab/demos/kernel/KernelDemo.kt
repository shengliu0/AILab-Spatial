// 核技巧 · 升维分类 (Boser, Guyon & Vapnik, 1992) — data a straight line cannot separate in 2-D becomes separable by a
// FLAT plane once it is lifted into 3-D, and that plane, seen from above, is a curve.
//
// LESSON: two classes of glowing beads lie on a floor (inner disc vs outer ring). The best straight line — a real linear
// SVM on the raw 2-D points, drawn as a vertical glass wall — gets about 60 %. Pinch 升维: every bead rises to
// z = φ(x, y) = x² + y² (a translucent paraboloid grows with them). A second linear SVM, trained on the lifted 3-D
// points, finds the widest-margin PLANE (with its margin slab); its cut through the paraboloid drops back to the floor
// as the decision CIRCLE. Support vectors carry white rings — only they define the plane: drag (or pinch) a far bead
// and nothing moves; drag a support vector and the plane follows. Other data sets: XOR (needs z = x·y), two moons
// (needs a learned RBF coordinate), a linearly separable baseline. Noise and the soft-margin C show the trade-off.
//
// Scene (volume metres): a tilted board in the middle (the floor = data plane, the lifted world above it); left: four
// data-set tiles; right: three lift tiles (mini surfaces) and the 升维 button. All maths lives in core/kernel (pure
// Kotlin SMO, JVM-tested against scipy); this file draws it. Every frame changes only transforms/materials; the
// plane polygons (7 vertices) are the only meshes edited live; the RBF surface mesh only on data change.

package com.bytedance.ailab.demos.kernel

import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.width
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.bytedance.ailab.core.P3
import com.bytedance.ailab.core.kernel.DataSet
import com.bytedance.ailab.core.kernel.KernelData
import com.bytedance.ailab.core.kernel.KernelGeom
import com.bytedance.ailab.core.kernel.KernelLab
import com.bytedance.ailab.core.kernel.Lift
import com.bytedance.ailab.core.kernel.RbfFeature
import com.bytedance.ailab.lab.About
import com.bytedance.ailab.lab.Bi
import com.bytedance.ailab.lab.Challenge
import com.bytedance.ailab.lab.Lang
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
import com.bytedance.ailab.lab.ui.LabPanel
import com.bytedance.ailab.lab.ui.LabSlider
import com.bytedance.ailab.lab.ui.SegmentControl
import com.bytedance.ailab.lab.ui.SegmentItem
import com.bytedance.ailab.render.MeshData
import com.bytedance.ailab.render.MeshGen
import com.bytedance.ailab.render.OptGeo
import com.bytedance.ailab.render.ProcTex
import com.bytedance.ailab.render.Rgb
import com.pico.spatial.ui.design.Text
import java.io.File
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sin

object KernelDemo : SpecDemo() {
    override val id = "kernel"
    override val title = "核技巧 · 升维分类"  // i18n-ok (zh half of DemoRegistry.titleOf; English in TITLE_EN)
    override val credit = "Boser, Guyon & Vapnik · 1992"

    // ================================================================ layout
    private val TILT = Math.toRadians(33.0).toFloat()
    private val CT = cos(TILT); private val ST = sin(TILT)
    private val BC = P3(0f, -0.012f, -0.03f)                 // centre of the board's floor
    private const val K = 0.30f                              // metres per data unit
    private const val CELLS = 40                             // contour grid
    private const val POOL = 130                             // contour segments per curve
    private const val RINGS_N = 18; private const val SEG_N = 48
    private const val BEAD_R = 0.0125f
    private val L get() = lab.L
    private fun dataTile(k: Int) = P3(-0.765f + 0.135f * (k % 2), 0.215f - 0.135f * (k / 2), 0.03f)
    private fun liftTile(k: Int) = P3(0.715f, 0.285f - 0.125f * k, 0.03f)
    private val LEVER = P3(0.715f, -0.105f, 0.15f)
    /** Stage-local → volume. */
    private fun toWorld(p: P3) = P3(BC.x + p.x, BC.y + p.y * CT - p.z * ST, BC.z + p.y * ST + p.z * CT)
    private fun tiltQ() = Q.axisAngle(1f, 0f, 0f, TILT)

    // ================================================================ colours
    private val COL_A = Rgb(1.0f, 0.58f, 0.16f)              // class +1 (orange)
    private val COL_B = Rgb(0.22f, 0.84f, 1.0f)              // class −1 (cyan)
    private val GOLD = Rgb(1.0f, 0.74f, 0.22f)
    private val CUT = Rgb(1.0f, 0.97f, 0.88f)
    private val PLANE = Rgb(0.74f, 0.66f, 1.0f)
    private val WALL = Rgb(1.0f, 0.42f, 0.38f)
    private fun sc(c: Rgb, k: Float) = Rgb(c.r * k, c.g * k, c.b * k)
    private fun add(c: Rgb, a: Float = 1f, tex: com.bytedance.ailab.render.Image? = null) = Mat.Unlit(c, alpha = (a * 20f).roundToInt() / 20f, tex = tex, blend = Blend.ADD)
    private val alu = Mat.Pbr(Rgb(0.80f, 0.82f, 0.85f), metallic = 1f, roughness = 0.32f)
    private val darkSteel = Mat.Pbr(Rgb(0.10f, 0.11f, 0.13f), metallic = 0.5f, roughness = 0.45f)
    private val slabTex by lazy { ProcTex.bakePlanet(256, 128, Rgb(0.06f, 0.065f, 0.075f), Rgb(0.10f, 0.105f, 0.12f), Rgb(0.16f, 0.165f, 0.18f), 0f, 88) }
    private val glow by lazy { KernelGfx.glowTex() }
    private val surfTex by lazy { KernelGfx.surfTex() }
    private val beadMat = listOf(Mat.Pbr(COL_A, metallic = 0.1f, roughness = 0.3f, emissive = sc(COL_A, 0.6f)), Mat.Pbr(COL_B, metallic = 0.1f, roughness = 0.3f, emissive = sc(COL_B, 0.6f)))
    private val glowMat by lazy { listOf(add(COL_A, 0.55f, glow), add(COL_B, 0.55f, glow)) }
    private val stemMat = listOf(add(COL_A, 0.35f), add(COL_B, 0.35f))
    private val footMat = listOf(add(COL_A, 0.55f), add(COL_B, 0.55f))
    private val svMat = add(Rgb(1f, 0.97f, 0.85f), 0.95f)
    private val badMat = add(Rgb(1f, 0.22f, 0.20f), 0.95f)

    // ================================================================ state
    val lab = KernelLab()
    var goalLifted = false; private set
    var liftT = 0f; private set
    var pendingLift: Lift? = null; private set
    var planeFade = 0f; private set
    var dropT = 0f; private set
    var speedIdx = 1; private set
    private val RISE_S = floatArrayOf(4.0f, 2.2f, 1.1f)
    /** Lesson bookkeeping: a non-SV moved with the plane unchanged / an SV moved and the plane moved. */
    var farOk = false; private set
    var svOk = false; private set
    private var lastMove = 0                                   // 0 none, 1 non-SV (plane kept), 2 SV (plane moved)
    private var lastMoveT = 0f
    private var lastAlpha = 0.0
    // drag / nudge
    private var dragIdx = -1
    private var dragX = 0.0; private var dragY = 0.0
    private var snapW = DoubleArray(3); private var snapB = 0.0; private var snapSv = false; private var snapX = 0.0; private var snapY = 0.0
    private var snapLifted = false
    private var nudgeIdx = -1; private var nudgeT = 0f
    private var nudgeFrom = DoubleArray(2); private var nudgeTo = DoubleArray(2)
    private val queue = ArrayDeque<() -> Unit>()
    // redraw keys
    private var drawnVersion = -1; private var drawnMode = -1; private var drawnDrop = -1f; private var drawnLiftT = -1f
    private var shownLift: Lift? = null; private var shownRbf: RbfFeature? = null
    private var readT = 0f; private var readEn: Boolean? = null

    /** The 3-D model is what's shown (fully lifted, no lift switch pending). */
    val showLifted get() = goalLifted && pendingLift == null && liftT >= 1f

    // ================================================================ nodes
    private lateinit var stage: Node
    private lateinit var surfNode: Node
    private val surfMesh = HashMap<Lift, MeshData>()
    private val beads = ArrayList<Node>(); private val marks = ArrayList<Node>(); private val stems = ArrayList<Node>(); private val feet = ArrayList<Node>()
    private val cutSegs = ArrayList<Node>(); private val floorSegs = ArrayList<Node>()
    private val planes = ArrayList<Node>(); private val planeMesh = ArrayList<MeshData>(); private val edges = ArrayList<Node>()
    private val dataTiles = ArrayList<Node>(); private val dataBases = ArrayList<Node>()
    private val liftTiles = ArrayList<Node>(); private val liftBases = ArrayList<Node>()
    private lateinit var lever: Node; private lateinit var arrow: Node
    private lateinit var unitCyl: MeshData
    private val tileOn = Mat.Pbr(Rgb(0.30f, 0.26f, 0.16f), metallic = 0.6f, roughness = 0.35f, emissive = sc(GOLD, 0.35f))
    private val tileOff = Mat.Pbr(Rgb(0.10f, 0.11f, 0.13f), metallic = 0.3f, roughness = 0.5f)
    private val segA = ArrayList<DoubleArray>()

    // ================================================================ build
    override fun buildSpec(cacheDir: File): Node {
        // REBUILD IDEMPOTENCE: every field back to its start value
        lab.configure(DataSet.RINGS, 0, Lift.RADIAL, 1000.0); lab.resetPoints()
        goalLifted = false; liftT = 0f; pendingLift = null; planeFade = 0f; dropT = 0f; speedIdx = 1
        farOk = false; svOk = false; lastMove = 0; lastMoveT = 0f; dragIdx = -1; nudgeIdx = -1; nudgeT = 0f; queue.clear()
        drawnVersion = -1; drawnMode = -1; drawnDrop = -1f; drawnLiftT = -1f; shownLift = null; shownRbf = null; readT = 0f; readEn = null
        listOf(beads, marks, stems, feet, cutSegs, floorSegs, planes, edges, dataTiles, dataBases, liftTiles, liftBases).forEach { it.clear() }
        planeMesh.clear(); surfMesh.clear(); labels.clear(); polyN.fill(0); drawnFloorMat = null

        val root = Node("KernelRoot")
        unitCyl = MeshGen.cylinder(1f, 1f, 8)
        root.add("Plinth", MeshGen.box(1.78f, 0.03f, 0.86f), Mat.Pbr(Rgb(1f, 1f, 1f), roughness = 0.55f, tex = slabTex), P3(0f, -0.235f, 0f))
        root.add("PlinthBand", MeshGen.box(1.79f, 0.007f, 0.87f), alu, P3(0f, -0.235f, 0f))

        // ---- the board: base, rim, floor (= the 2-D data plane)
        stage = root.add(Node("Stage")).also { it.pos = BC; it.rot = tiltQ() }
        val S = (2 * L).toFloat() * K
        stage.add("BoardBase", MeshGen.box(S + 0.03f, 0.014f, S + 0.03f), Mat.Pbr(Rgb(0.07f, 0.075f, 0.09f), metallic = 0.2f, roughness = 0.5f), P3(0f, -0.009f, 0f))
        stage.add("BoardRim", MeshGen.box(S + 0.036f, 0.006f, S + 0.036f), alu, P3(0f, -0.014f, 0f))
        stage.add("Floor", OptGeo.quad(S, S), Mat.Unlit(Rgb(1f, 1f, 1f), tex = KernelGfx.floorTex(256, L)), P3(0f, 0.0012f, 0f)).also { it.rot = Q.axisAngle(1f, 0f, 0f, -Math.PI.toFloat() / 2) }
        for ((k, c) in listOf(-1f to -1f, 1f to -1f, -1f to 1f, 1f to 1f).withIndex()) {
            val top = toWorld(P3(c.first * (S / 2 - 0.02f), -0.016f, c.second * (S / 2 - 0.02f)))
            KernelGfx.line(root.add("Post_$k", unitCyl, alu), P3(top.x, -0.22f, top.z), top, 0.007f)
        }

        // ---- lifted surfaces (one mesh per lift; the RBF one is refilled when its feature is refitted)
        for (lf in Lift.values()) surfMesh[lf] = KernelGfx.polarMesh(RINGS_N, SEG_N)
        fillSurf(Lift.RADIAL); fillSurf(Lift.PRODUCT)
        surfNode = stage.add("Surface", surfMesh[Lift.RADIAL], add(Rgb(1f, 1f, 1f), 0.6f, surfTex)).also { it.visible = false }

        // ---- beads: sphere + glow + support-vector ring (children), stem to the floor, footprint
        val beadMesh = MeshGen.sphere(BEAD_R, 12, 18, MeshGen.SphereUv.EQUIRECT)
        val quad = OptGeo.quad(1f, 1f)
        val ring = KernelGfx.ringXY(0.020f, 0.0255f)
        val foot = MeshGen.cylinder(0.0075f, 0.001f, 14)
        for (i in 0 until lab.n) {
            val c = if (lab.pts.label[i] > 0) 0 else 1
            val b = stage.add("Bead_$i", beadMesh, beadMat[c]).also { it.tapRadius = 0.045f }
            b.add("Glow_$i", quad, glowMat[c]).also { it.billboard = true; it.uniformScale(0.062f) }
            marks += b.add("Mark_$i", ring, svMat).also { it.billboard = true; it.visible = false }
            beads += b
            stems += stage.add("Stem_$i", unitCyl, stemMat[c]).also { it.visible = false }
            feet += stage.add("Foot_$i", foot, footMat[c])
        }
        // ---- curves: the cut on the surface, and its shadow on the floor (pooled segments)
        repeat(POOL) { k ->
            cutSegs += stage.add("Cut_$k", unitCyl, add(CUT, 0.95f)).also { it.visible = false }
            floorSegs += stage.add("Flr_$k", unitCyl, add(GOLD, 0.95f)).also { it.visible = false }
        }
        // ---- the separating plane and its two margin planes (clipped to the lifted box), with outlines
        for (k in 0 until 3) {
            val md = KernelGfx.fanMesh(); planeMesh += md
            planes += stage.add(listOf("Plane", "MarginP", "MarginN")[k], md, add(PLANE, 0.2f)).also { it.visible = false }
            repeat(6) { e -> edges += stage.add("Edge_${k}_$e", unitCyl, add(PLANE, 0.8f)).also { it.visible = false } }
        }

        // ---- left: data-set tiles
        val dTex = DataSet.values().map { KernelGfx.dataTex(KernelData.make(it, 0), 96, COL_A, COL_B) }
        for (k in 0 until 4) {
            val c = dataTile(k)
            dataBases += root.add("DataBase_$k", MeshGen.box(0.122f, 0.008f, 0.122f), tileOff, c + P3(0f, -0.004f * CT, -0.004f * ST)).also { it.rot = tiltQ() }
            dataTiles += root.add("Data_$k", OptGeo.quad(0.108f, 0.108f), Mat.Unlit(Rgb(1f, 1f, 1f), tex = dTex[k]), c + P3(0f, 0.001f * CT, 0.001f * ST)).also {
                it.rot = tiltQ() * Q.axisAngle(1f, 0f, 0f, -Math.PI.toFloat() / 2); it.tapRadius = 0.06f
            }
            KernelGfx.line(root.add("DataStem_$k", unitCyl, alu), P3(c.x, -0.22f, c.z - 0.03f), P3(c.x, c.y - 0.03f, c.z - 0.03f), 0.0045f)
        }
        // ---- right: lift tiles (mini surfaces) and the 升维 button
        val moonsRbf = KernelLab().apply { configure(DataSet.MOONS, 0, Lift.RBF, 1000.0) }
        for (lf in Lift.values()) {
            val k = lf.ordinal; val c = liftTile(k)
            val md = KernelGfx.polarMesh(8, 24)
            KernelGfx.fillSurface(md, 8, 24, 1.0, 0.055f, 0.7f) { x, y -> lf.h(when (lf) { Lift.RADIAL -> x * x + y * y; Lift.PRODUCT -> x * y; Lift.RBF -> moonsRbf.phi(x, y) }) }
            liftBases += root.add("LiftBase_$k", MeshGen.cylinder(0.061f, 0.008f, 28), tileOff, c + P3(0f, -0.006f, 0f)).also { it.rot = tiltQ() }
            liftTiles += root.add("Lift_$k", md, add(Rgb(1f, 1f, 1f), 0.95f, surfTex), c).also { it.rot = tiltQ(); it.tapRadius = 0.06f }
            KernelGfx.line(root.add("LiftStem_$k", unitCyl, alu), P3(c.x, -0.22f, c.z - 0.03f), P3(c.x, c.y - 0.03f, c.z - 0.03f), 0.0045f)
        }
        root.add("LeverPost", unitCyl, alu).also { KernelGfx.line(it, P3(LEVER.x, -0.22f, LEVER.z), P3(LEVER.x, LEVER.y - 0.02f, LEVER.z), 0.010f) }
        lever = root.add("LiftLever", MeshGen.cylinder(0.040f, 0.016f, 32), Mat.Pbr(Rgb(0.16f, 0.15f, 0.22f), metallic = 0.6f, roughness = 0.3f, emissive = sc(PLANE, 0.25f)), LEVER).also { it.tapRadius = 0.065f }
        arrow = lever.add(Node("Arrow")).also { it.pos = P3(0f, 0.02f, 0f) }
        arrow.add("ArrowShaft", unitCyl, add(GOLD, 1f), P3(0f, 0.018f, 0f)).also { it.scale = P3(0.007f, 0.036f, 0.007f) }
        arrow.add("ArrowHead", KernelGfx.cone(0.018f, 0.026f), add(GOLD, 1f), P3(0f, 0.036f, 0f))
        root.add("LeverRing", MeshGen.annulus(0.042f, 0.049f, 40), add(PLANE, 0.8f), LEVER + P3(0f, 0.0085f, 0f))

        applyTiles()
        return root
    }

    private fun fillSurf(lf: Lift) {
        val md = surfMesh[lf] ?: return
        KernelGfx.fillSurface(md, RINGS_N, SEG_N, 1.0, K) { x, y -> lf.h(when (lf) { Lift.RADIAL -> x * x + y * y; Lift.PRODUCT -> x * y; Lift.RBF -> lab.phi(x, y) }) }
    }

    // ================================================================ per frame
    override fun update(dt: Float) {
        // ---- queued autoplay actions (lesson): one at a time, when nothing is animating
        if (queue.isNotEmpty() && nudgeIdx < 0 && dragIdx < 0 && settled()) queue.removeFirst().invoke()
        // ---- the lift animation (down first when a different lift is pending)
        val target = if (pendingLift != null) 0f else if (goalLifted) 1f else 0f
        val rate = 1f / RISE_S[speedIdx]
        liftT = if (liftT < target) minOf(target, liftT + dt * rate) else maxOf(target, liftT - dt * rate * 2f)
        if (liftT == 0f && pendingLift != null) { lab.configure(lift = pendingLift!!); pendingLift = null; dropT = 0f }
        if (showLifted) {
            planeFade = minOf(1f, planeFade + dt / 0.6f)
            if (planeFade >= 1f) dropT = minOf(1f, dropT + dt / (1.3f * RISE_S[speedIdx] / 2.2f))
        } else { planeFade = 0f; dropT = 0f }
        // ---- nudge animation
        if (nudgeIdx >= 0) {
            nudgeT = minOf(1f, nudgeT + dt / 0.45f)
            val s = ProcTex.smooth01(nudgeT).toDouble()
            lab.move(nudgeIdx, nudgeFrom[0] + (nudgeTo[0] - nudgeFrom[0]) * s, nudgeFrom[1] + (nudgeTo[1] - nudgeFrom[1]) * s, refitFeature = false)
            if (nudgeT >= 1f) { val i = nudgeIdx; nudgeIdx = -1; endMove(i) }
        }
        lastMoveT = maxOf(0f, lastMoveT - dt)

        // ---- surface mesh for the active lift
        if (shownLift != lab.lift || (lab.lift == Lift.RBF && shownRbf !== lab.rbf)) {
            if (lab.lift == Lift.RBF) { fillSurf(Lift.RBF); meshEdited(surfMesh[Lift.RBF]!!) }
            shownLift = lab.lift; shownRbf = lab.rbf
            surfNode.setMesh(surfMesh[lab.lift]); drawnVersion = -1
        }
        val t = ProcTex.smooth01(liftT)
        surfNode.visible = t > 0.01f
        if (surfNode.visible) { val s = P3(1f, t, 1f); if (surfNode.scale != s) surfNode.scale = s; surfNode.setMat(add(Rgb(1f, 1f, 1f), 0.30f + 0.30f * t, surfTex)) }
        val ar = if (goalLifted) Q.axisAngle(1f, 0f, 0f, Math.PI.toFloat()) else Q.identity()
        if (arrow.rot != ar) { arrow.rot = ar; arrow.pos = if (goalLifted) P3(0f, 0.062f, 0f) else P3(0f, 0.02f, 0f) }

        // ---- beads (transforms only when something moved)
        val mode = if (showLifted) 1 else 0
        val changed = drawnVersion != lab.version || drawnMode != mode
        if (changed || drawnLiftT != t) { placeBeads(t); drawnLiftT = t }
        if (changed) { drawnVersion = lab.version; drawnMode = mode; drawnDrop = -1f; markBeads(); drawPlanes() }
        planeAlpha(t)
        if (changed || drawnDrop != dropT) { drawCurves(); drawnDrop = dropT }

        readT += dt
        if (readT >= 0.1f || dt == 0f || readoutText.isEmpty() || readEn != Lang.en) { readT = 0f; readEn = Lang.en; updateReadout() }
        updateLabels()
    }

    private fun settled() = pendingLift == null && (liftT == 0f || liftT == 1f) && (!showLifted || dropT >= 1f)

    private fun h(lf: Lift, phi: Double) = lf.h(phi).coerceIn(0.0, 1.0).toFloat()
    private fun local(x: Double, y: Double, hUnits: Float, lift: Float = 0f) = P3((x * K).toFloat(), hUnits * K + lift, (-y * K).toFloat())

    private fun placeBeads(t: Float) {
        val p = lab.pts
        for (i in 0 until lab.n) {
            val hh = h(lab.lift, lab.phi(p.x[i], p.y[i])) * t
            val c = local(p.x[i], p.y[i], hh, BEAD_R * 0.7f)
            if (beads[i].pos != c) beads[i].pos = c
            val f = local(p.x[i], p.y[i], 0f, 0.0015f)
            if (feet[i].pos != f) feet[i].pos = f
            stems[i].visible = hh * K > 0.02f
            if (stems[i].visible) KernelGfx.line(stems[i], f, c, 0.0016f)
        }
    }

    private fun markBeads() {
        val lifted = showLifted
        for (i in 0 until lab.n) {
            val bad = lab.marginOf(i, lifted) < 0
            marks[i].visible = bad || (lifted && lab.isSupport(i, lifted))      // 2-D soft-margin SVs are nearly all points: rings only where they teach
            marks[i].setMat(if (bad) badMat else svMat)
        }
    }

    /** Vertex count of each clipped polygon (plane, +margin, −margin); 0 = misses the box. */
    private val polyN = IntArray(3)

    /** The plane (lifted) or the vertical wall (2-D), clipped to the box above the floor, with the margin planes. */
    private fun drawPlanes() {
        val lifted = showLifted
        val m = lab.model(lifted)
        val lf = lab.lift
        val w = if (lifted) m.w else doubleArrayOf(m.w[0], m.w[1], 0.0)
        val lo = doubleArrayOf(-L, -L, lf.zf); val hi = doubleArrayOf(L, L, if (lifted) lf.zt else lf.zf + 0.42)
        for (k in 0 until 3) {
            val d = m.b + when (k) { 0 -> 0.0; 1 -> -1.0; else -> 1.0 }
            val poly = if (m.wNorm < 1e-9) emptyList() else KernelGeom.planeBox(w, d, lo, hi)
            val pts = poly.map { local(it[0], it[1], h(lf, it[2])) }
            polyN[k] = if (pts.size >= 3) pts.size else 0
            KernelGfx.fanInto(planeMesh[k], pts); meshEdited(planeMesh[k])
            for (e in 0 until polyN[k]) KernelGfx.line(edges[k * 6 + e], pts[e], pts[(e + 1) % pts.size], if (k == 0) 0.0022f else 0.0014f)
        }
    }

    private fun planeAlpha(t: Float) {
        val lifted = showLifted
        val a = if (lifted) planeFade else if (pendingLift == null) (1f - t) else 0f
        val col = if (lifted) PLANE else WALL
        val on = a >= 0.03f
        for (k in 0 until 3) {
            val show = on && polyN[k] > 0 && (k == 0 || lifted)               // the 2-D wall alone: its margin walls were clutter
            planes[k].visible = show
            if (show) planes[k].setMat(add(col, (if (k == 0) 0.13f else 0.035f) * a))
            val em = add(col, (if (k == 0) 0.95f else 0.30f) * a)
            for (e in 0 until 6) { val n = edges[k * 6 + e]; n.visible = show && e < polyN[k]; if (n.visible) n.setMat(em) }
        }
    }

    /** Lifted: the plane ∩ surface (white) and its shadow dropping to the floor (gold). 2-D: the straight line. */
    private fun drawCurves() {
        val lifted = showLifted
        val lf = lab.lift
        lab.contour(lifted, CELLS, segA)
        val s = ProcTex.smooth01(dropT)
        val show2d = !lifted && pendingLift == null && liftT < 1f
        var nc = 0; var nf = 0
        for (seg in segA) {
            val ax = seg[0]; val ay = seg[1]; val bx = seg[2]; val by = seg[3]
            if (lifted) {
                val inDisc = hypot(ax, ay) <= 1.0 && hypot(bx, by) <= 1.0
                val ha = h(lf, lab.phi(ax, ay)); val hb = h(lf, lab.phi(bx, by))
                if (inDisc && nc < POOL) { val n = cutSegs[nc++]; n.visible = true; KernelGfx.line(n, local(ax, ay, ha, 0.003f), local(bx, by, hb, 0.003f), 0.0026f) }
                if (planeFade >= 1f && nf < POOL && (inDisc || s >= 1f)) {
                    val n = floorSegs[nf++]; n.visible = true
                    KernelGfx.line(n, local(ax, ay, ha * (1 - s), 0.003f), local(bx, by, hb * (1 - s), 0.003f), 0.0030f)
                }
            } else if (show2d && nf < POOL) {
                val n = floorSegs[nf++]; n.visible = true
                KernelGfx.line(n, local(ax, ay, 0f, 0.003f), local(bx, by, 0f, 0.003f), 0.0030f)
            }
        }
        for (k in nc until POOL) cutSegs[k].visible = false
        for (k in nf until POOL) floorSegs[k].visible = false
        val fm = if (lifted) add(GOLD, 0.95f) else add(WALL, 0.95f * (1f - ProcTex.smooth01(liftT)))
        if (drawnFloorMat != fm) { drawnFloorMat = fm; floorSegs.forEach { it.setMat(fm) } }
    }
    private var drawnFloorMat: Mat? = null

    private fun applyTiles() {
        for (k in 0 until 4) {
            val on = lab.set.ordinal == k
            dataBases[k].setMat(if (on) tileOn else tileOff)
            val c = dataTile(k); val lift = if (on) 0.022f else 0.004f
            dataTiles[k].pos = c + P3(0f, lift * CT, lift * ST)
        }
        val want = pendingLift ?: lab.lift
        for (k in 0 until 3) {
            val on = want.ordinal == k
            liftBases[k].setMat(if (on) tileOn else tileOff)
            liftTiles[k].pos = liftTile(k) + P3(0f, (if (on) 0.018f else 0f) * CT, (if (on) 0.018f else 0f) * ST)
        }
    }

    // ================================================================ text
    private val SET_NAMES = listOf("同心圆环" to "Rings", "XOR 四象限" to "XOR quadrants", "双月牙" to "Two moons", "线性可分" to "Linearly separable")  // i18n-ok
    private val SET_SHORT = listOf("圆环" to "Rings", "XOR" to "XOR", "双月" to "Moons", "线性" to "Linear")  // i18n-ok
    fun setName(k: Int = lab.set.ordinal) = tr(SET_NAMES[k].first, SET_NAMES[k].second)
    fun liftName(lf: Lift = lab.lift) = when (lf) { Lift.RADIAL -> "z = x² + y²"; Lift.PRODUCT -> "z = x·y"; Lift.RBF -> tr("z = ½tanh(RBF 得分)", "z = ½tanh(RBF score)") }
    private fun noiseName() = listOf(tr("无噪声", "no noise"), tr("噪声低", "low noise"), tr("噪声高", "high noise"))[lab.noise]
    private fun fmtC() = if (lab.C >= 999.0) tr("C = 1000（≈硬间隔）", "C = 1000 (≈ hard)") else "C = " + (if (lab.C >= 10) "%.0f" else if (lab.C >= 1) "%.1f" else "%.2f").format(lab.C)
    private fun pct(a: Double) = (a * 100).roundToInt()

    private fun term(v: Double, sym: String, first: Boolean): String {
        val s = if (v < 0) "−" else if (first) "" else "+"
        val body = "%.2f".format(abs(v)) + sym
        return if (first) s + body else " $s $body"
    }

    private fun planeText(): String {
        val lifted = showLifted
        val p = lab.unitPlane(lifted)
        val m = lab.model(lifted)
        val eq = if (lifted) term(p[0], "x", true) + term(p[1], "y", false) + term(p[2], "z", false) + term(p[3], "", false) + " = 0"
                 else term(p[0], "x", true) + term(p[1], "y", false) + term(p[3], "", false) + " = 0"
        val nsv = m.sol.supportVectors().size; val nb = m.sol.boundSupportVectors().size
        val mg = if (m.margin.isInfinite()) "∞" else "%.2f".format(m.margin)
        val head = if (lifted) tr("平面", "Plane") else tr("直线", "Line")
        return tr("%s %s · 间隔 2/|w| = %s · 支持向量 %d 个%s", "%s %s · margin 2/|w| = %s · %d support vectors%s")
            .format(head, eq, mg, nsv, if (nb > 0) tr("（%d 个越界）", " (%d inside the margin)").format(nb) else "")
    }

    private fun updateReadout() {
        val lifted = showLifted
        val dim = if (lifted) tr("3-D，%s", "3-D, %s").format(liftName()) else if (liftT > 0f || pendingLift != null) tr("升维中…", "lifting…") else "2-D"
        val l1 = tr("数据：%s（40 点，%s）· 维度：%s · 训练准确率 %d%% · %s", "Data: %s (40 pts, %s) · dimension: %s · training accuracy %d%% · %s")
            .format(setName(), noiseName(), dim, pct(lab.accuracy(lifted)), fmtC())
        readoutText = "$l1\n${planeText()}\n${insight()}"
    }

    fun insight(): String {
        val lifted = showLifted
        val a2 = lab.accuracy(false); val a3 = lab.accuracy(true)
        return when {
            lastMoveT > 0f && lastMove == 1 -> tr("这个点不是支持向量（α = 0）：挪动它，w 和 b 一点都不变", "Not a support vector (α = 0): moving it leaves w and b exactly unchanged")
            lastMoveT > 0f && lastMove == 2 -> tr("这是支持向量（α = %.2f）：它一动，最宽间隔的平面就跟着动", "A support vector (α = %.2f): move it and the widest-margin plane moves too").format(lastAlpha)
            pendingLift != null || (liftT > 0f && liftT < 1f) -> tr("升维：每个点 (x, y) 升到高度 z = φ(x, y)，在地面上的位置不变", "Lifting: each point (x, y) rises to height z = φ(x, y); its floor position stays")
            !lifted && a2 >= 0.999 -> tr("这组数据一条直线就分开了（线性可分）—— 升维也只是把直线变成竖直的平面", "One straight line already separates this set (linearly separable) — no lift needed")
            !lifted -> tr("2-D 里最好的直线也只分对 %d%% —— 捏「升维」，把点抬到第三维", "The best straight line in 2-D gets only %d%% — pinch “Lift” to raise the points into 3-D").format(pct(a2))
            a3 >= 0.999 && lab.circle() != null && planeFade >= 1f -> tr("3-D 里一个平面就分开了；它切抛物面得到圆（r = %.2f），投回地面就是弯曲的边界", "In 3-D one flat plane separates them; it cuts the paraboloid in a circle (r = %.2f) — the curved 2-D boundary").format(lab.circle()!![2])
            a3 >= 0.999 -> tr("3-D 里一个平面就全分对了；它和曲面的交线投回地面 = 2-D 里的弯曲边界", "In 3-D one flat plane gets them all; its cut through the surface, dropped to the floor, is the curved boundary")
            lab.noise > 0 -> tr("有噪声时没有平面能全分对：C 小 → 间隔宽、容忍越界；C 大 → 间隔窄、尽量分对", "With noise no plane is perfect: small C → wide margin, tolerates violations; large C → narrow, fits harder")
            else -> tr("这种升维不够：两类点在 3-D 里仍然交错 —— 换一种 z = φ(x, y) 试试", "This lift is not enough: the classes still interleave in 3-D — try another z = φ(x, y)")
        }
    }

    private fun updateLabels() {
        labels.clear()
        labels += LabelSpec(tr("地面 = 原始 2-D 数据", "Floor = the raw 2-D data"), toWorld(P3(0f, 0.01f, (L * K).toFloat() + 0.05f)))
        if (liftT > 0.3f) labels += LabelSpec(liftName(), toWorld(P3(-(L * K).toFloat() + 0.03f, K * 1.05f, -(L * K).toFloat() * 0.6f)))
        if (showLifted && planeFade > 0.5f) labels += LabelSpec(tr("分隔平面 · 间隔 %.2f", "Separating plane · margin %.2f").format(lab.svm3.margin), toWorld(P3((L * K).toFloat() - 0.02f, K * 0.95f, -(L * K).toFloat() * 0.5f)))
        else if (!goalLifted && liftT == 0f) labels += LabelSpec(tr("最好的直线 · %d%%", "Best straight line · %d%%").format(pct(lab.accuracy(false))), toWorld(P3((L * K).toFloat() - 0.02f, 0.11f, -(L * K).toFloat() * 0.5f)))
        labels += LabelSpec(tr("数据集 · 捏一个换", "Data sets · pinch one"), P3(-0.698f, 0.335f, 0.03f))
        labels += LabelSpec(tr("升维方式 z = φ(x, y)", "Lift z = φ(x, y)"), P3(0.715f, 0.395f, 0.03f))
        labels += LabelSpec(if (goalLifted) tr("降回 2-D", "Back to 2-D") else tr("升维 ⇧", "Lift ⇧"), LEVER + P3(0f, -0.075f, 0.03f))
    }

    // ================================================================ actions (panel, pinch, lesson)
    fun selectData(k: Int) {
        val s = DataSet.values()[k.coerceIn(0, 3)]
        lab.configure(set = s); dropT = 0f; lastMoveT = 0f; applyTiles()
        farOk = false; svOk = false
    }
    /** Pick a lift and raise the points into it (via the floor if another lift is up). */
    fun chooseLift(lf: Lift) {
        if (lab.lift == lf && pendingLift == null) { goalLifted = true }
        else if (liftT == 0f) { lab.configure(lift = lf); pendingLift = null; goalLifted = true; dropT = 0f }
        else { pendingLift = lf; goalLifted = true }
        applyTiles()
    }
    fun setLifted(on: Boolean) { goalLifted = on; if (!on) pendingLift?.let { lab.configure(lift = it); pendingLift = null } ; applyTiles() }
    fun setNoise(k: Int) { if (k != lab.noise) { lab.configure(noise = k); dropT = 0f; lastMoveT = 0f } }
    fun setC(c: Double) { val q = 10.0.pow((log10(c) * 10).roundToInt() / 10.0).coerceIn(0.1, 1000.0); if (q != lab.C) lab.configure(C = q) }
    fun setSpeed(k: Int) { speedIdx = k.coerceIn(0, 2) }
    fun resetPoints() { lab.resetPoints(); lastMoveT = 0f; nudgeIdx = -1 }

    /** Push bead [i] toward the decision boundary (animated). A far bead goes only as far as stays outside the margin. */
    fun nudge(i: Int) {
        if (i !in 0 until lab.n || nudgeIdx >= 0 || dragIdx >= 0) return
        val lifted = showLifted
        val sv = lab.isSupport(i, lifted)
        val step = if (sv) 0.10 else lab.safeStepTowardBoundary(i, lifted, 0.22).coerceAtLeast(0.0)
        val d = lab.towardBoundary(i, lifted)
        beginMove(i)
        nudgeIdx = i; nudgeT = 0f
        nudgeFrom = doubleArrayOf(lab.pts.x[i], lab.pts.y[i])
        nudgeTo = doubleArrayOf(lab.pts.x[i] + d[0] * step, lab.pts.y[i] + d[1] * step)
    }
    fun farBead() = lab.farPoint(showLifted)
    fun svBead() = lab.freeSupport(showLifted)
    fun nudgeFar() = nudge(farBead())
    fun nudgeSv() = nudge(svBead())

    private fun beginMove(i: Int) {
        snapLifted = showLifted
        val m = lab.model(snapLifted)
        snapW = m.w.copyOf(); snapB = m.b; snapSv = lab.isSupport(i, snapLifted); snapX = lab.pts.x[i]; snapY = lab.pts.y[i]
        lastAlpha = m.sol.alpha[i]
    }
    private fun endMove(i: Int) {
        if (lab.lift == Lift.RBF) lab.retrain(true)
        val m = lab.model(snapLifted)
        val dPlane = KernelLab.planeDelta(snapW, snapB, m.w, m.b)
        val moved = hypot(lab.pts.x[i] - snapX, lab.pts.y[i] - snapY)
        if (moved >= 0.05) {
            if (!snapSv && dPlane < 1e-5) { farOk = true; lastMove = 1; lastMoveT = 4f }
            else if (snapSv && dPlane > 1e-3) { svOk = true; lastMove = 2; lastMoveT = 4f }
            else lastMoveT = 0f
        }
    }

    override val tapTargets: List<TapTarget> get() {
        val l = ArrayList<TapTarget>()
        l += TapTarget("LiftLever", "升维 → 抬起/放下", "Lift → raise/lower") { goalLifted }
        SET_SHORT.forEachIndexed { k, n -> l += TapTarget("Data_$k", n.first, n.second) { lab.set.ordinal == k } }
        l += TapTarget("Lift_0", "z = x²+y²", "z = x²+y²") { (pendingLift ?: lab.lift) == Lift.RADIAL }
        l += TapTarget("Lift_1", "z = x·y", "z = x·y") { (pendingLift ?: lab.lift) == Lift.PRODUCT }
        l += TapTarget("Lift_2", "z = RBF", "z = RBF") { (pendingLift ?: lab.lift) == Lift.RBF }
        val far = farBead(); if (far >= 0) l += TapTarget("Bead_$far", "远点 → 推一下", "Far point → nudge")
        val sv = svBead(); if (sv >= 0 && sv != far) l += TapTarget("Bead_$sv", "支持向量 → 推一下", "Support vector → nudge")
        return l
    }

    override fun isTarget(name: String) = name == "LiftLever" || name.startsWith("Data_") || name.startsWith("Lift_") || name.startsWith("Bead_")

    override fun onTap(name: String) {
        when {
            name == "LiftLever" -> setLifted(!goalLifted)
            name.startsWith("Data_") -> name.removePrefix("Data_").toIntOrNull()?.let { selectData(it) }
            name.startsWith("Lift_") -> name.removePrefix("Lift_").toIntOrNull()?.let { chooseLift(Lift.values()[it.coerceIn(0, 2)]) }
            name.startsWith("Bead_") -> name.removePrefix("Bead_").toIntOrNull()?.let { tapSv = lab.isSupport(it, showLifted); nudge(it) }
        }
    }
    private var tapSv = false

    override fun tapMessage(name: String): String? = when {
        name == "LiftLever" -> if (goalLifted) tr("升维：每个点升到 z = φ(x, y) —— 看平面在 3-D 里把它们分开", "Lifting: every point rises to z = φ(x, y) — watch a plane separate them in 3-D")
                               else tr("降回 2-D：平面变回一面竖直的墙 = 一条直线", "Back to 2-D: the plane becomes a vertical wall = a straight line")
        name.startsWith("Data_") -> tr("换成「%s」—— 先看 2-D 里最好的直线能分对多少", "Loaded “%s” — first see how well a straight line does").format(setName())
        name.startsWith("Lift_") -> tr("升维方式 %s：点会升到这张曲面上", "Lift %s: the points rise onto this surface").format(liftName(pendingLift ?: lab.lift))
        name.startsWith("Bead_") -> if (tapSv) tr("支持向量：推它一下 —— 平面跟着动", "Support vector: nudged — the plane follows")
                                   else tr("不是支持向量：推它一下 —— 平面不动", "Not a support vector: nudged — the plane stays put")
        else -> null
    }

    // ---- drag a bead over the floor (hand motion projected onto the tilted board); retrains live
    private const val DRAG_GAIN = 1.2f
    override fun isDragTarget(name: String) = name.startsWith("Bead_")
    override fun onDragStart(name: String) {
        val i = name.removePrefix("Bead_").toIntOrNull() ?: return
        if (nudgeIdx >= 0) return
        dragIdx = i; dragX = lab.pts.x[i]; dragY = lab.pts.y[i]; beginMove(i)
    }
    override fun onDrag(name: String, dx: Float, dy: Float, dz: Float) {
        val i = dragIdx; if (i < 0) return
        val lx = dx; val lz = -dy * ST + dz * CT
        dragX += lx / K * DRAG_GAIN; dragY -= lz / K * DRAG_GAIN
        lab.move(i, dragX, dragY, refitFeature = false)
        val r = hypot(dragX, dragY); if (r > KernelData.R_MAX) { dragX *= KernelData.R_MAX / r; dragY *= KernelData.R_MAX / r }
    }
    override fun onDragEnd(name: String) { val i = dragIdx; if (i >= 0) { dragIdx = -1; endMove(i) } }

    override fun reviewVariants(): Map<String, () -> Unit> = mapOf(
        "lifted" to { selectData(0); setNoise(0); setSpeed(2); chooseLift(Lift.RADIAL); repeat(300) { update(1f / 60f) } },
        "xor_radial" to { selectData(1); setSpeed(2); chooseLift(Lift.RADIAL); repeat(300) { update(1f / 60f) } },
        "xor_product" to { selectData(1); setSpeed(2); chooseLift(Lift.PRODUCT); repeat(400) { update(1f / 60f) } },
        "moons_rbf" to { selectData(2); setSpeed(2); chooseLift(Lift.RBF); repeat(400) { update(1f / 60f) } },
        "noisy_softC" to { selectData(0); setNoise(2); setC(1.0); setSpeed(2); chooseLift(Lift.RADIAL); repeat(400) { update(1f / 60f) } },
        "linear_2d" to { setLifted(false); setNoise(0); setC(1000.0); selectData(3); repeat(300) { update(1f / 60f) } },
    )

    // ================================================================ panel
    @Composable
    override fun Panel() {
        LabPanel(readout) {
            LabSlider(tr("软间隔 C", "Soft margin C"), log10(lab.C).toFloat(), -1f..3f, fmtC(), width = 360) { setC(10.0.pow(it.toDouble())) }
            SegmentControl {
                listOf(tr("无噪声", "No noise"), tr("噪声低", "Low noise"), tr("噪声高", "High noise")).forEachIndexed { k, l ->
                    SegmentItem(selected = lab.noise == k, onClick = { setNoise(k) }, title = { Text(l) })
                }
            }
            Button(onClick = { resetPoints() }) { Text(tr("↺ 复位点", "↺ Reset points")) }
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
            Bi("地面上有两类发光的点：橙色在内圈，青色围成外环。红色竖墙是 2-D 里最好的直线，只分对约六成。捏右前方的「升维」，每个点升到高度 z = x² + y²：外圈升得更高，一块平面就能从两层之间切过。左边可换数据集，右边可换升维方式；面板调噪声和软间隔 C。", "Two classes of glowing points lie on the floor: orange inside, cyan in a ring around them. The red wall is the best straight line in 2-D — about 60 % right. Pinch “Lift” (front right) and each point rises to height z = x² + y²: the ring rises higher, so one flat plane slips between the two layers. Swap data sets on the left and lifts on the right; the panel sets noise and the soft margin C."),
        ),
        About.Page(
            Bi("先认识几个词", "A few words first"),
            Bi("「分类器」是一种模型：给它一个点，它说出属于哪一类。最简单的分类器是一条直线（在 3-D 里是一块平面）：这边一类，那边另一类。「训练」就是用已知类别的点去摆这条线。「特征」是描述一个点的数，这里每个点本来只有 x、y 两个；「升维」就是再给它算出第三个特征。", "A “classifier” is a model that takes a point and says which class it belongs to. The simplest is a straight line (in 3-D, a flat plane): one class on this side, the other on that. “Training” means placing the line using points whose classes are known. “Features” are the numbers describing a point; here each starts with just two, x and y. “Lifting” computes a third one."),
        ),
        About.Page(
            Bi("背景：最宽的间隔", "Background: the widest margin"),
            Bi("1960 年代，瓦普尼克等人提出：在所有能分开两类的直线里，选离两边最近的点都最远的那条 ——「最宽间隔」，对新数据最稳。1992 年，博泽、居永和瓦普尼克加上「核技巧」，让这把直尺能在更高维的空间里用，于是有了支持向量机（SVM）。在深度学习兴起之前，它是许多识别任务的首选。", "In the 1960s Vapnik and colleagues proposed: of all the lines that separate two classes, pick the one that stays farthest from the nearest points on both sides — the “widest margin”, the safest bet for new data. In 1992 Boser, Guyon and Vapnik added the “kernel trick”, letting this ruler work in higher dimensions: the support vector machine (SVM). Before deep learning it was the first choice for many recognition tasks."),
        ),
        About.Page(
            Bi("核心想法：抬高了就直了", "Key idea: lift it and cut straight"),
            Bi("桌上撒着黑豆和红豆，红豆围在中间，一根直尺怎么也隔不开。但把离中心越远的豆子抬得越高，一张平平的纸板就能从两层之间插进去。纸板在高处是「直」的，从上往下看，它的切口却是一个圆。核技巧更进一步：不必真算出新坐标，只要会算两个点在高维里有多「像」就够了。", "Black and red beans are scattered on a table, the red ones in the middle; no ruler can separate them. But lift each bean higher the farther it is from the centre, and a flat card slides between the two layers. Up there the card is “straight”; seen from above, its cut is a circle. The kernel trick goes further: you never compute the new coordinates — you only need how “alike” two points are up there."),
        ),
        About.Page(
            Bi("怎么看这个 3D 场景", "How to read the 3-D scene"),
            Bi("地面 = 原始的 2-D 数据；珠子的高度 = 新算出的第三个特征 z；半透明曲面 = 升维的规则（抛物面、马鞍面或 RBF 曲面）。玻璃平面 = SVM 找到的分界，两侧的薄板 = 间隔；平面切曲面的交线落回地面，就是 2-D 里弯曲的边界。带白环的珠子 = 支持向量：只有它们决定平面，挪动远处的点，平面纹丝不动。", "The floor = the raw 2-D data; a bead's height = the new third feature z; the translucent surface = the lift rule (paraboloid, saddle or RBF surface). The glass plane = the boundary the SVM found; the slab around it = the margin; the plane's cut through the surface, dropped to the floor, is the curved 2-D boundary. Beads with white rings = support vectors: only they fix the plane — move a far bead and nothing changes."),
        ),
        About.Page(
            Bi("有多大？今天在哪里？", "How big? Where is it today?"),
            Bi("这里每组只有 40 个点，SVM 在你眼前精确求解。真实常用的高斯（RBF）核相当于升到无穷多维，却只需算点与点之间的相似度。如今图像和语言多交给深度网络，但数据少的场合（如医学检测、垃圾邮件过滤）仍常用 SVM；而「先把数据变到好分的空间，再切直线」正是深度网络每一层在做的事。", "Each set here has just 40 points, and the SVM is solved exactly in front of you. The widely used Gaussian (RBF) kernel is like lifting into infinitely many dimensions, yet needs only point-to-point similarities. Deep networks now handle images and language, but SVMs still serve where data is scarce, such as medical tests and spam filters — and “transform the data until a straight cut works” is what every layer of a deep network does."),
        ),
    ))

    override val lesson: Lesson by lazy {
        Lesson(
            listOf(
                LessonStep(
                    say = Bi("地面上有两类点：橙色在内圈，青色在外圈。能用一条直线把它们分开吗？", "Two classes on the floor: orange inside, cyan in a ring around them. Can one straight line separate them?"),
                    options = listOf(
                        Bi("能", "Yes"),
                        Bi("不能", "No"),
                        Bi("加点噪声就能", "Only with some noise"),
                    ),
                    answer = 1,
                    onEnter = { if (lab.set != DataSet.RINGS) selectData(0) },
                ),
                LessonStep(
                    say = Bi("红色竖墙 = 2-D 里最好的直线（线性 SVM），只分对约六成。捏右前方的「升维」按钮，把每个点抬到 z = x² + y²。", "The red wall = the best straight line in 2-D (a linear SVM): about 60 % right. Pinch the “Lift” button (front right) to raise every point to z = x² + y²."),
                    target = "LiftLever",
                    done = { goalLifted && liftT >= 1f && lab.lift == Lift.RADIAL },
                    autoplay = { selectData(0); setNoise(0); chooseLift(Lift.RADIAL) },
                ),
                LessonStep(
                    say = Bi("看：外圈离中心远，升得更高。一个平面从两层之间切过 —— 再看它和抛物面的交线落回地面。", "Look: the outer ring is farther out, so it rises higher. A plane slips between the two layers — watch its cut through the paraboloid drop to the floor."),
                    done = { showLifted && dropT >= 1f && lab.lift == Lift.RADIAL },
                    autoplay = { if (lab.set != DataSet.RINGS) selectData(0); chooseLift(Lift.RADIAL) },
                    reveal = true,
                    insight = Bi("在 3-D 里一个平面就分开了。平面切抛物面 z = x²+y² 得到一个圆，投回地面就是 2-D 里弯曲的边界。这就是核技巧：在高维画「直」的，回到低维就是「弯」的。", "In 3-D a single flat plane separates them. It cuts the paraboloid z = x²+y² in a circle, which dropped to the floor is the curved 2-D boundary. That is the kernel trick: straight in high dimensions, curved back down."),
                ),
                LessonStep(
                    say = Bi("带白环的点是「支持向量」。把一个离边界很远、没有白环的点推近一些，分界会变吗？", "Beads with white rings are the support vectors. If you push a bead that is far from the boundary (no ring) closer, does the boundary change?"),
                    options = listOf(
                        Bi("会变", "Yes, it moves"),
                        Bi("不会变", "No, it stays"),
                        Bi("只变一点点", "Only a little"),
                    ),
                    answer = 1,
                ),
                LessonStep(
                    say = Bi("试试：拖动（或在面板点「远点 → 推一下」）一个远处的点，再拖动一个带白环的支持向量，看平面怎么动。", "Try it: drag (or tap “Far point → nudge”) a far bead, then drag a ringed support vector and watch the plane."),
                    onEnter = { farOk = false; svOk = false },
                    done = { farOk && svOk },
                    autoplay = {
                        if (lab.set != DataSet.RINGS) selectData(0)
                        chooseLift(Lift.RADIAL)
                        queue.addLast { nudgeFar() }; queue.addLast { nudgeSv() }
                    },
                    reveal = true,
                    insight = Bi("只有支持向量（αᵢ > 0）出现在 w = Σ αᵢyᵢφ(xᵢ) 里：远处的点 α = 0，挪动它 w、b 纹丝不动；支持向量一动，最宽间隔的平面就得重新找。", "Only support vectors (αᵢ > 0) appear in w = Σ αᵢyᵢφ(xᵢ): a far bead has α = 0, so moving it leaves w and b untouched; move a support vector and the widest-margin plane must be found again."),
                ),
            ),
            challenge = Challenge(
                say = Bi("挑战：换到左边的「XOR」数据（四个象限），选对升维方式，让平面在 3-D 里 100% 分对。", "Challenge: load the “XOR” set (four quadrants, left shelf) and pick the lift that lets a plane get 100 % in 3-D."),
                progress = { challengeProgress() },
                done = { lab.set == DataSet.XOR && showLifted && lab.accuracy(true) >= 0.999 },
                autoplay = { setNoise(0); selectData(1); chooseLift(Lift.PRODUCT) },
            ),
        )
    }

    private fun challengeProgress(): String = when {
        lab.set != DataSet.XOR -> tr("先捏左边的「XOR」数据", "First pinch “XOR” on the left shelf")
        !showLifted -> tr("2-D 直线：%d%% —— 捏右边一种升维方式", "2-D line: %d%% — pinch a lift on the right").format(pct(lab.accuracy(false)))
        lab.accuracy(true) >= 0.999 -> tr("✓ %s：3-D 平面 100%% 分对", "✓ %s: the 3-D plane gets 100%%").format(liftName())
        lab.noise > 0 -> tr("%s：%d%% · 有噪声时很难 100%%，先选「无噪声」", "%s: %d%% · hard with noise — choose “No noise”").format(liftName(), pct(lab.accuracy(true)))
        else -> tr("%s：%d%% —— 这种升维没把四个象限分成上下两层，换一种", "%s: %d%% — this lift does not split the quadrants into two layers; try another").format(liftName(), pct(lab.accuracy(true)))
    }
}
