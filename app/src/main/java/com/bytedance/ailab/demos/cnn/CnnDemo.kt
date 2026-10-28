// 卷积神经网络 · CNN (LeCun, 1989) — a tensor C×H×W IS a 3-D block; the demo draws every one of them.
//
// LESSON: a k×k×C kernel is a small CUBE that pierces ALL input channels at once. At each position its
// k·k·C products are summed into ONE number = one pixel of one feature map; the same weights slide over every
// position (weight sharing). Several kernels → several maps → the next block; ReLU + 2×2 max-pool shrink it;
// a second conv, a second pool, then a dense layer votes. Deeper cells see more: 3 → 4 → 8 → 10 input pixels.
// Multispectral mode adds a near-infrared plane: the NDVI-like (NIR − R) kernel lights real vegetation and
// ignores artificial turf that is exactly as green in RGB — information an RGB camera simply does not have.
//
// Models (core/cnn, tools/train_cnn.py): six REAL little CNNs trained with PyTorch on synthetic data (digits
// rendered from the Mac's system fonts; procedurally generated 4-band tiles), run here in pure Kotlin.
// Scene (left → right): picture → input block → conv-1 block → pool-1 → conv-2 → pool-2 → class bars.
// Front shelf: the 8 first-layer kernels of the live network as voxel cubes (warm = +, cool = −).

package com.bytedance.ailab.demos.cnn

import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.width
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.bytedance.ailab.core.P3
import com.bytedance.ailab.core.cnn.CnnLab
import com.bytedance.ailab.core.cnn.CnnModel
import com.bytedance.ailab.core.cnn.CnnOps
import com.bytedance.ailab.core.cnn.CnnScale
import com.bytedance.ailab.core.cnn.Glyphs
import com.bytedance.ailab.core.cnn.ReceptiveField
import com.bytedance.ailab.core.cnn.SlantProbe
import com.bytedance.ailab.lab.Bi
import com.bytedance.ailab.lab.Challenge
import com.bytedance.ailab.lab.LabAssets
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
import com.bytedance.ailab.lab.ui.LabPanel
import com.bytedance.ailab.lab.ui.LabRow
import com.bytedance.ailab.lab.ui.LabSwitch
import com.bytedance.ailab.lab.ui.SegmentControl
import com.bytedance.ailab.lab.ui.SegmentItem
import com.bytedance.ailab.render.Image
import com.bytedance.ailab.render.MeshData
import com.bytedance.ailab.render.MeshGen
import com.bytedance.ailab.render.ProcTex
import com.bytedance.ailab.render.Rgb
import com.pico.spatial.ui.design.Text
import java.io.File
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

object CnnDemo : SpecDemo() {
    override val id = "cnn"
    override val title = "卷积神经网络"  // i18n-ok (zh half of DemoRegistry.titleOf; English in TITLE_EN)
    override val credit = "LeCun · 1989"

    // ================================================================ data
    private val lab by lazy { CnnLab(LabAssets::bytes) }
    private val glyphs by lazy { Glyphs.parse(LabAssets.bytes("cnn/glyphs.bin")) }
    private val glyphImg = HashMap<String, Image>()
    private fun gimg(key: String): Image = glyphImg.getOrPut(key) { CnnGfx.glyphImage(glyphs.getValue(key)) }
    private fun gaspect(key: String): Float = glyphs.getValue(key).let { it.w.toFloat() / it.h }

    // hand-made kernel names (bank order of tools/train_cnn.py)
    private val HAND_DIGIT = listOf("竖边" to "V-edge", "横边" to "H-edge", "斜边↗" to "Diag ↗", "斜边↘" to "Diag ↘",  // i18n-ok (zh/en pairs)
        "模糊" to "Blur", "锐化" to "Sharpen", "亮点" to "Spot", "红−蓝" to "R − B")  // i18n-ok
    private val HAND_SCENE = listOf("NIR−R" to "NIR − R", "绿色" to "Green", "水体" to "Water", "近红外" to "NIR",  // i18n-ok
        "竖边" to "V-edge", "横边" to "H-edge", "模糊" to "Blur", "锐化" to "Sharpen")  // i18n-ok
    private val CLASS = listOf("植被" to "vegetation", "水体" to "water", "人造草坪" to "turf", "屋顶" to "roof", "裸土" to "soil")  // i18n-ok
    private val CH_NAME = listOf("R", "G", "B", "NIR")

    // ================================================================ state (public: panel + lesson)
    var scene = false; private set
    var learned = true; private set
    var nir = true; private set
    var speedIdx = 1; private set
    var playing = true; private set
    var sel = 0; private set
    var rfOn = false; private set
    private var digitPos = 0
    private var scenePos = 0
    private var rfI = 2; private var rfJ = 2
    private var rfTaps = 0
    private var scanK = 0                 // 0 .. 26·26-1, row-major
    private var scanAcc = 0f
    private var holdT = 0f
    private var manual = false            // kernel dragged by hand: whole map shown + marker
    private var dragI = 0f; private var dragJ = 0f
    private var stepsTotal = 0            // kernel positions visited since the scene was built
    private var stepsAtSel = 0            // ... when the current kernel was picked
    private var rfDragI = 0f; private var rfDragJ = 0f
    private var dirty = true
    private var shownEn: Boolean? = null
    private var prevI = 0; private var prevJ = 0
    private var lastRevealK = -1

    private val SPEEDS = floatArrayOf(4f, 24f, 160f)       // kernel positions per second
    private val N_POS = 26 * 26

    // ================================================================ layout
    private val YAW = Math.toRadians(36.0).toFloat()   // planes face front-right; stacks recede back-left
    private val CA = cos(YAW); private val SA = sin(YAW)

    /** One tensor drawn as a stack of [n] planes (local XY), plane 0 in front, the whole stack yawed. */
    private class Block(val n: Int, val side: Int, val size: Float, val gap: Float, val c: P3) {
        val px = size / side
        var count = n
        lateinit var g: Node
        fun z(k: Int) = ((count - 1) / 2f - k) * gap
        fun lx(j: Float) = (j + 0.5f - side / 2f) * px
        fun ly(i: Float) = (side / 2f - i - 0.5f) * px
        val depth get() = (count - 1) * gap
        fun world(x: Float, y: Float, z: Float) = P3(c.x + x * CA + z * SA, c.y + y, c.z - x * SA + z * CA)
    }

    private const val BY = 0.190f
    private val IN = Block(4, 28, 0.29f, 0.095f, P3(-0.565f, BY, 0.02f))
    private val C1 = Block(8, 26, 0.24f, 0.030f, P3(-0.201f, BY, -0.03f))
    private val P1 = Block(8, 13, 0.13f, 0.024f, P3(0.087f, BY, -0.03f))
    private val C2 = Block(16, 11, 0.11f, 0.009f, P3(0.294f, BY, -0.03f))
    private val P2 = Block(16, 5, 0.06f, 0.009f, P3(0.462f, BY, -0.03f))
    private val BLOCKS get() = listOf(IN, C1, P1, C2, P2)

    // atlases (texels per pixel): input with pixel gaps, the rest compact
    private val A_IN = CnnGfx.Atlas(4, 28, 4)
    private val A_C1 = CnnGfx.Atlas(8, 26, 3)
    private val A_P1 = CnnGfx.Atlas(8, 13, 5)
    private val A_C2 = CnnGfx.Atlas(16, 11, 5)
    private val A_P2 = CnnGfx.Atlas(16, 5, 6)

    private val SHELF_X0 = -0.80f; private val SHELF_DX = 0.096f
    private val SHELF_Y = -0.105f; private val SHELF_Z = 0.30f
    private val SV = 0.0115f           // shelf voxel size
    private val SP = 0.0165f           // shelf voxel pitch
    private val SZ = 0.019f            // shelf slice pitch

    private val BAR_X0 = 0.565f; private val BAR_L = 0.17f
    private val THUMB = P3(-0.835f, BY, 0.08f); private const val THUMB_S = 0.080f

    // ================================================================ colours (sRGB)
    private val CH = listOf(Rgb(1.0f, 0.30f, 0.26f), Rgb(0.32f, 1.0f, 0.40f), Rgb(0.34f, 0.55f, 1.0f), Rgb(0.90f, 0.40f, 1.0f))
    private val GOLD = Rgb(1.0f, 0.84f, 0.42f)
    private val RFC = Rgb(1.0f, 0.38f, 0.80f)
    private val CYAN = Rgb(0.30f, 0.80f, 1.0f)
    private fun scaled(c: Rgb, k: Float) = Rgb(c.r * k, c.g * k, c.b * k)
    private fun add(c: Rgb, a: Float = 1f, tex: Image? = null) = Mat.Unlit(c, alpha = a, tex = tex, blend = Blend.ADD)
    /** Feature maps: alpha-blended (opacity ∝ |activation|) so 8-16 stacked maps occlude instead of summing to mush. */
    private fun map(a: Float, tex: Image) = Mat.Unlit(Rgb(1f, 1f, 1f), alpha = a, tex = tex, blend = Blend.TRANSPARENT)

    private val alu = Mat.Pbr(Rgb(0.80f, 0.82f, 0.85f), metallic = 1f, roughness = 0.32f)
    private val slabTex by lazy {
        ProcTex.bakePlanet(256, 128, Rgb(0.06f, 0.065f, 0.075f), Rgb(0.10f, 0.105f, 0.12f), Rgb(0.16f, 0.165f, 0.18f), 0f, 88)
    }
    private val frameTex by lazy { CnnGfx.frameTex() }

    /** Voxel palette: weight / max|w| of its kernel → 15 levels, warm = +, cool = −, a faint core for ~0. */
    private val VOX_L = 7
    private val voxMats by lazy {
        List(2 * VOX_L + 1) { k ->
            val v = (k - VOX_L).toFloat() / VOX_L
            val t = kotlin.math.abs(v)
            val c = if (v >= 0f) CnnGfx.WARM else CnnGfx.COOL
            add(Rgb(0.10f + c.r * 0.95f * t, 0.10f + c.g * 0.95f * t, 0.11f + c.b * 0.95f * t))
        }
    }
    private fun voxMat(v: Float) = voxMats[(v.coerceIn(-1f, 1f) * VOX_L).roundToInt() + VOX_L]

    // ================================================================ nodes
    private val inPlanes = ArrayList<Node>(); private val c1Planes = ArrayList<Node>(); private val p1Planes = ArrayList<Node>()
    private val c2Planes = ArrayList<Node>(); private val p2Planes = ArrayList<Node>()
    private val wires = ArrayList<Node>()
    private val wireMeshIn = HashMap<Int, MeshData>()
    private lateinit var thumb: Node
    private val tags = ArrayList<Node>()
    private lateinit var kRig: Node
    private lateinit var kBox: Node
    private lateinit var kFill: Node
    private val kBoxMesh = HashMap<Int, MeshData>(); private val kFillMesh = HashMap<Int, MeshData>()
    private val kVox = Array(4) { arrayOfNulls<Node>(9) }
    private lateinit var outPix: Node
    private lateinit var selFrame: Node
    private lateinit var selBack: Node
    private val rfFills = ArrayList<Node>()
    private val rfFillMeshIn = HashMap<Int, MeshData>()
    private val rays = ArrayList<Node>()
    private lateinit var revA: Node; private lateinit var revB: Node
    private lateinit var revAm: MeshData; private lateinit var revBm: MeshData
    private val shelfRigs = ArrayList<Node>(); private val shelfBoxes = ArrayList<Node>(); private val shelfNames = ArrayList<Node>()
    private val shelfVox = Array(8) { Array(4) { arrayOfNulls<Node>(9) } }
    private val shelfBoxMesh = HashMap<Int, MeshData>()
    private val rfBoxes = ArrayList<Node>(); private val rfLines = ArrayList<Node>()
    private val rfBoxMeshIn = HashMap<Int, MeshData>()
    private val bars = ArrayList<Node>(); private val tracks = ArrayList<Node>(); private val barNames = ArrayList<Node>(); private val fans = ArrayList<Node>()
    private val barLen = FloatArray(10)
    private lateinit var unitCyl: MeshData
    private lateinit var unitBox: MeshData

    // per-state textures, cached so revisiting a state re-uses the same Image (the adapter caches by identity)
    private class Tex(val c1: Image, val p1: Image, val c2: Image, val p2: Image)
    private val texCache = HashMap<String, Tex>()
    private val inCache = HashMap<String, Pair<Image, Image>>()   // input atlas + thumbnail
    private var acts: CnnModel.Acts? = null
    private var model: CnnModel? = null
    private var tex: Tex? = null
    private var matSel: Mat? = null; private var matDim: Mat? = null; private var matGhost: Mat? = null

    // ================================================================ build
    override fun buildSpec(cacheDir: File): Node {
        // REBUILD IDEMPOTENCE: every field back to its start value
        scene = false; learned = true; nir = true; speedIdx = 1; playing = true; sel = 0; rfOn = false
        digitPos = 0; scenePos = 0; rfI = 2; rfJ = 2; rfTaps = 0; scanK = 0; scanAcc = 0f; holdT = 0f; manual = false
        stepsTotal = 0; stepsAtSel = 0; dirty = true; shownEn = null; prevI = 0; prevJ = 0; lastRevealK = -1
        rfDragI = 0f; rfDragJ = 0f; dragI = 0f; dragJ = 0f; barLen.fill(0f); acts = null; model = null; tex = null
        listOf(inPlanes, c1Planes, p1Planes, c2Planes, p2Planes, wires, tags, rays, shelfRigs, shelfBoxes, shelfNames,
            rfBoxes, rfLines, rfFills, bars, tracks, barNames, fans).forEach { it.clear() }
        wireMeshIn.clear(); kBoxMesh.clear(); kFillMesh.clear(); shelfBoxMesh.clear(); rfBoxMeshIn.clear(); rfFillMeshIn.clear()
        IN.count = 3

        val root = Node("CnnRoot")
        unitCyl = MeshGen.cylinder(1f, 1f, 8)
        unitBox = MeshGen.box(1f, 1f, 1f)
        val voxMesh = MeshGen.box(1f, 1f, 1f)

        // bench: dark granite slab with an aluminium edge, like the physics benches
        root.add("Plinth", MeshGen.box(1.78f, 0.03f, 0.58f), Mat.Pbr(Rgb(1f, 1f, 1f), roughness = 0.55f, tex = slabTex), P3(0f, -0.235f, 0.07f))
        root.add("PlinthBand", MeshGen.box(1.79f, 0.007f, 0.59f), alu, P3(0f, -0.235f, 0.07f))

        // ---- tensor blocks
        fun block(b: Block, tag: String, a: CnnGfx.Atlas, list: ArrayList<Node>) {
            b.g = root.add(Node("${tag}Block")).also { it.pos = b.c; it.rot = Q.axisAngle(0f, 1f, 0f, YAW) }
            val cs = a.cellSize(b.size)
            for (k in 0 until b.n) {
                val uv = a.uvCell(k)
                list += b.g.add("${tag}Plane$k", CnnGfx.quadUv(cs, cs, uv[0], uv[1], uv[2], uv[3]), null, P3(0f, 0f, b.z(k)))
            }
        }
        IN.count = 4
        block(IN, "In", A_IN, inPlanes); block(C1, "C1", A_C1, c1Planes); block(P1, "P1", A_P1, p1Planes)
        block(C2, "C2", A_C2, c2Planes); block(P2, "P2", A_P2, p2Planes)
        IN.count = 3
        val wireMat = add(Rgb(0.42f, 0.55f, 0.72f), 0.55f)
        for ((t, b) in listOf("In", "C1", "P1", "C2", "P2").zip(BLOCKS)) {
            val m = CnnGfx.wireBox(b.size + 0.012f, b.size + 0.012f, b.depth + 0.02f, 0.0016f)
            if (b === IN) { wireMeshIn[3] = m; wireMeshIn[4] = CnnGfx.wireBox(b.size + 0.012f, b.size + 0.012f, 3 * b.gap + 0.02f, 0.0016f) }
            wires += b.g.add(if (t == "P2") "Pool2" else "${t}Wire", m, wireMat)
        }
        wires[4].tapRadius = 0.052f        // the last pooled block: pinch → receptive field

        // channel tags on the input planes
        for (k in 0 until 4) {
            val key = "ch_${CH_NAME[k]}"; val h = 0.026f
            tags += IN.g.add("Tag$k", CnnGfx.quadUv(h * gaspect(key), h, 0f, 0f, 1f, 1f),
                Mat.Unlit(CH[k], tex = gimg(key), blend = Blend.TRANSPARENT), P3(IN.size / 2 - h * gaspect(key) / 2, IN.size / 2 + 0.020f + k * 0.024f, IN.z(k)))
        }

        // ---- the picture (what an eye sees) → its channel planes (what the network sees)
        thumb = root.add("Sample", CnnGfx.quadUv(THUMB_S, THUMB_S, 0f, 0f, 1f, 1f), null, THUMB)
        thumb.tapRadius = 0.06f
        root.add("SampleFrame", CnnGfx.wireBox(THUMB_S + 0.008f, THUMB_S + 0.008f, 0.004f, 0.0018f), add(Rgb(0.55f, 0.70f, 0.90f), 0.8f), THUMB)
        val arrow = root.add("SampleArrow", unitCyl, add(Rgb(0.55f, 0.70f, 0.90f), 0.6f))
        CnnGfx.placeLine(arrow, P3(THUMB.x + THUMB_S / 2 + 0.01f, BY, THUMB.z), IN.world(-IN.size / 2 - 0.01f, 0f, IN.z(if (SA > 0f) IN.count - 1 else 0)), 0.0016f)

        // ---- the sliding kernel: a k×k×C voxel cube riding on the input block
        kRig = IN.g.add(Node("KernelRig"))
        for (c in listOf(3, 4)) {
            val d = (c - 1) * IN.gap + 0.026f; val s = 3 * IN.px + 0.006f
            kBoxMesh[c] = CnnGfx.wireBox(s, s, d, 0.0024f); kFillMesh[c] = MeshGen.box(s, s, d)
        }
        kFill = kRig.add("KernelFill", kFillMesh[3], add(GOLD, 0.07f))
        kBox = kRig.add("KernelBox", kBoxMesh[3], add(GOLD, 1f))
        kBox.tapRadius = 0.05f
        for (c in 0 until 4) for (t in 0 until 9) {
            kVox[c][t] = kRig.add("KVox_${c}_$t", voxMesh, voxMats[VOX_L]).also { it.uniformScale(IN.px * 0.56f) }
        }
        val cs1 = A_C1.cellSize(C1.size)
        // a dark card behind the selected map: its zeros read as dark instead of showing the maps behind it
        selBack = C1.g.add("SelBack", CnnGfx.quadUv(cs1, cs1, 0f, 0f, 1f, 1f), Mat.Unlit(Rgb(0.02f, 0.025f, 0.035f), alpha = 0.88f, blend = Blend.TRANSPARENT))
        selFrame = C1.g.add("SelFrame", CnnGfx.wireBox(cs1 + 0.008f, cs1 + 0.008f, 0.002f, 0.0022f), add(GOLD, 0.9f))
        outPix = C1.g.add("OutPix", CnnGfx.quadUv(C1.px * 1.6f, C1.px * 1.6f, 0f, 0f, 1f, 1f), add(GOLD, 1f, frameTex))
        for (r in 0 until 4) rays += root.add("Ray$r", unitCyl, add(GOLD, 0.55f))
        revAm = CnnGfx.quadUv(0.01f, 0.01f, 0f, 0f, 1f, 1f); revBm = CnnGfx.quadUv(0.01f, 0.01f, 0f, 0f, 1f, 1f)
        revA = C1.g.add("RevealA", revAm, null); revB = C1.g.add("RevealB", revBm, null)

        // ---- the kernel shelf (front): the 8 first-layer kernels of the live network
        root.add("Rail", MeshGen.box(8 * SHELF_DX + 0.02f, 0.006f, 0.05f), add(Rgb(0.30f, 0.45f, 0.65f), 0.5f),
            P3(SHELF_X0 + 3.5f * SHELF_DX, SHELF_Y - 0.052f, SHELF_Z))
        for (c in listOf(3, 4)) shelfBoxMesh[c] = CnnGfx.wireBox(3 * SP + 0.006f, 3 * SP + 0.006f, (c - 1) * SZ + SV + 0.008f, 0.0016f)
        for (k in 0 until 8) {
            val rig = root.add(Node("ShelfRig$k")).also { it.pos = P3(SHELF_X0 + k * SHELF_DX, SHELF_Y, SHELF_Z); it.rot = Q.axisAngle(0f, 1f, 0f, YAW) }
            shelfRigs += rig
            shelfBoxes += rig.add("Shelf_$k", shelfBoxMesh[3], add(Rgb(0.45f, 0.62f, 0.85f), 0.6f)).also { it.tapRadius = 0.046f }
            for (c in 0 until 4) for (t in 0 until 9)
                shelfVox[k][c][t] = rig.add("SVox_${k}_${c}_$t", voxMesh, voxMats[VOX_L]).also { it.uniformScale(SV) }
            shelfNames += root.add("ShelfName$k", CnnGfx.quadUv(1f, 1f, 0f, 0f, 1f, 1f), null, P3(SHELF_X0 + k * SHELF_DX, SHELF_Y - 0.078f, SHELF_Z + 0.01f))
        }

        // ---- receptive field: one wire box per block + the frustum edges between them
        val rfMat = add(RFC, 1f)
        for ((l, b) in BLOCKS.withIndex()) {
            val w = (ReceptiveField.window(4, 0, l).let { it[1] - it[0] + 1 }).toFloat()
            val s = w * b.px + 0.004f
            val m = CnnGfx.wireBox(s, s, b.depth + 0.024f, 0.0022f)
            if (b === IN) { rfBoxMeshIn[3] = m; rfBoxMeshIn[4] = CnnGfx.wireBox(s, s, 3 * b.gap + 0.024f, 0.0022f) }
            rfBoxes += b.g.add("RFBox$l", m, rfMat).also { it.visible = false }
            val fm = MeshGen.box(s, s, b.depth + 0.024f)
            if (b === IN) { rfFillMeshIn[3] = fm; rfFillMeshIn[4] = MeshGen.box(s, s, 3 * b.gap + 0.024f) }
            rfFills += b.g.add("RFFill$l", fm, add(RFC, 0.10f)).also { it.visible = false }
        }
        for (e in 0 until 16) rfLines += root.add("RFLine$e", unitCyl, add(RFC, 0.45f)).also { it.visible = false }

        // ---- dense layer → class bars
        for (k in 0 until 10) {
            tracks += root.add("Track$k", unitBox, add(Rgb(0.20f, 0.30f, 0.42f), 0.35f))
            bars += root.add("Bar$k", unitBox, add(CYAN))
            barNames += root.add("BarName$k", CnnGfx.quadUv(1f, 1f, 0f, 0f, 1f, 1f), null)
            fans += root.add("Fan$k", unitCyl, add(CYAN, 0.2f))
        }

        applyState()
        return root
    }

    // ================================================================ state → scene
    private val chans get() = if (!scene) 3 else if (nir) 4 else 3
    private val sampleIdx get() = if (!scene) lab.digitShow[digitPos] else lab.sceneShow[scenePos]
    private val curModel: CnnModel get() = lab.model(scene, learned, nir)

    /** Re-run inference and swap textures / colours. Only on a user change, never per frame. */
    private fun applyState() {
        dirty = false
        val m = curModel; model = m
        val a = lab.acts(scene, learned, nir, sampleIdx); acts = a
        val c = chans
        IN.count = c
        // input atlas + thumbnail (per sample and sensor)
        val (inImg, thImg) = inCache.getOrPut("${if (scene) "s" else "d"}$c|$sampleIdx") {
            val x = a.input
            // scenes are dense: each band is contrast-stretched to its own range for display (digits shown as is)
            val lo = FloatArray(x.c) { k -> if (scene) (0 until 784).minOf { x.d[k * 784 + it] } else 0f }
            val hi = FloatArray(x.c) { k -> if (scene) (0 until 784).maxOf { x.d[k * 784 + it] }.coerceAtLeast(lo[k] + 0.05f) else 1f }
            val atlas = A_IN.bake(CnnGfx.grey(0.85f)) { k, i, j -> if (k < x.c) CnnGfx.grey((x[k, i, j] - lo[k]) / (hi[k] - lo[k])) else CnnGfx.grey(0f) }
            val th = Image(84, 84)
            for (y in 0 until 84) for (xx in 0 until 84) th.px[y * 84 + xx] = ProcTex.argb(1f, x[0, y / 3, xx / 3], x[1, y / 3, xx / 3], x[2, y / 3, xx / 3])
            atlas to th
        }
        for (k in 0 until 4) {
            inPlanes[k].setMat(add(scaled(CH[k], if (scene) 0.62f else 0.92f), 1f, inImg)); inPlanes[k].visible = k < c
            inPlanes[k].pos = P3(0f, 0f, IN.z(k)); tags[k].visible = k < c
            tags[k].pos = P3(tags[k].pos.x, tags[k].pos.y, IN.z(k))
        }
        thumb.setMat(Mat.Unlit(Rgb(1f, 1f, 1f), tex = thImg))
        wires[0].setMesh(wireMeshIn[c]); kBox.setMesh(kBoxMesh[c]); kFill.setMesh(kFillMesh[c]); rfBoxes[0].setMesh(rfBoxMeshIn[c]); rfFills[0].setMesh(rfFillMeshIn[c])
        // feature-map atlases (per model and sample)
        val t = texCache.getOrPut("${m.name}|$sampleIdx") {
            // every map scaled to its own 98th-percentile |value| (a colour scale only; the readout gives the numbers)
            fun per(t: com.bytedance.ailab.core.cnn.Tensor) = FloatArray(t.c) { k -> CnnScale.robust(t.d.copyOfRange(k * t.h * t.w, (k + 1) * t.h * t.w)) }
            val q1 = per(a.conv1); val q2 = per(a.conv2); val qp1 = per(a.pool1); val qp2 = per(a.pool2)
            val fr = ProcTex.argb(0.65f, 0.50f, 0.62f, 0.80f)
            Tex(A_C1.bake(fr) { k, i, j -> CnnGfx.divergingA(a.conv1[k, i, j] / q1[k]) },
                A_P1.bake(fr) { k, i, j -> CnnGfx.divergingA(a.pool1[k, i, j] / qp1[k]) },
                A_C2.bake(fr) { k, i, j -> CnnGfx.divergingA(a.conv2[k, i, j] / q2[k]) },
                A_P2.bake(fr) { k, i, j -> CnnGfx.divergingA(a.pool2[k, i, j] / qp2[k]) })
        }
        tex = t
        matSel = map(1f, t.c1); matDim = map(0.20f, t.c1); matGhost = map(0.10f, t.c1)
        revA.setMat(matSel); revB.setMat(matSel)
        val mid = map(0.85f, t.p1); p1Planes.forEach { it.setMat(mid) }
        val m2 = map(0.70f, t.c2); c2Planes.forEach { it.setMat(m2) }
        val m3 = map(0.95f, t.p2); p2Planes.forEach { it.setMat(m3) }
        applyKernels()
        applyBars()
        applyLang()
        lastRevealK = -1
    }

    /** Voxel colours of the shelf and of the sliding kernel (weights / max |w| of that kernel). */
    private fun applyKernels() {
        val m = model ?: return
        if (dirty) return                 // a rebuild is pending: applyState() will call us with the new model
        val c = m.cIn
        for (k in 0 until 8) {
            val w = m.kernel(k); val mx = w.maxOf { kotlin.math.abs(it) }.coerceAtLeast(1e-6f)
            for (ch in 0 until 4) for (t in 0 until 9) {
                val n = shelfVox[k][ch][t]!!
                n.visible = ch < c
                if (ch < c) {
                    n.setMat(voxMat(w[ch * 9 + t] / mx))
                    n.pos = P3((t % 3 - 1) * SP, (1 - t / 3) * SP, ((c - 1) / 2f - ch) * SZ)
                }
            }
            shelfBoxes[k].setMesh(shelfBoxMesh[c])
            shelfRigs[k].pos = P3(SHELF_X0 + k * SHELF_DX, SHELF_Y + if (k == sel) 0.018f else 0f, SHELF_Z)
            shelfBoxes[k].setMat(if (k == sel) add(GOLD, 1f) else add(Rgb(0.45f, 0.62f, 0.85f), 0.6f))
        }
        val w = m.kernel(sel); val mx = w.maxOf { kotlin.math.abs(it) }.coerceAtLeast(1e-6f)
        for (ch in 0 until 4) for (t in 0 until 9) {
            val n = kVox[ch][t]!!
            n.visible = ch < c
            if (ch < c) { n.setMat(voxMat(w[ch * 9 + t] / mx)); n.pos = P3((t % 3 - 1) * IN.px, (1 - t / 3) * IN.px, IN.z(ch) + 0.012f) }
        }
        for (k in 0 until 8) c1Planes[k].setMat(if (k == sel) matSel else matDim)
    }

    private val nOut get() = if (scene) 5 else 10
    private fun barY(k: Int): Float = if (!scene) BY + 0.150f - k * 0.0333f else BY + 0.115f - k * 0.058f

    private fun applyBars() {
        for (k in 0 until 10) {
            val on = k < nOut
            bars[k].visible = on; tracks[k].visible = on; barNames[k].visible = on; fans[k].visible = on
            if (!on) continue
            val y = barY(k)
            tracks[k].pos = P3(BAR_X0 + BAR_L / 2, y, 0f); tracks[k].scale = P3(BAR_L, 0.0045f, 0.004f)
        }
    }

    private fun applyLang() {
        val en = Lang.en
        shownEn = en
        val names = if (learned) null else if (scene) HAND_SCENE else HAND_DIGIT
        for (k in 0 until 8) {
            val key = if (names == null) "learned_$k" else "${if (scene) "hs" else "hd"}_${if (en) "en" else "zh"}_$k"
            val h = 0.024f
            shelfNames[k].scale = P3(h * gaspect(key), h, 1f)
            shelfNames[k].setMat(Mat.Unlit(if (k == sel) GOLD else Rgb(0.80f, 0.88f, 1.0f), tex = gimg(key), blend = Blend.TRANSPARENT))
        }
        for (k in 0 until nOut) {
            val key = if (!scene) "digit_$k" else "cls_${if (en) "en" else "zh"}_$k"
            val h = if (scene) 0.024f else 0.026f
            val w = h * gaspect(key)
            barNames[k].scale = P3(w, h, 1f)
            barNames[k].pos = P3(BAR_X0 + BAR_L + 0.012f + w / 2, barY(k), 0.002f)
            barNames[k].setMat(Mat.Unlit(Rgb(0.88f, 0.94f, 1.0f), tex = gimg(key), blend = Blend.TRANSPARENT))
        }
    }

    // ================================================================ per frame
    override fun update(dt: Float) {
        if (dirty) applyState()
        if (shownEn != Lang.en) applyLang()
        val a = acts ?: return

        // ---- the scan
        if (playing && !manual) {
            if (scanK >= N_POS - 1) {
                holdT += dt
                if (holdT > 1.5f) { holdT = 0f; prevI = 25; prevJ = 25; scanK = 0; scanAcc = 0f; stepsTotal++ }
            } else {
                scanAcc += dt * SPEEDS[speedIdx]
                while (scanAcc >= 1f && scanK < N_POS - 1) { prevI = scanK / 26; prevJ = scanK % 26; scanK++; scanAcc -= 1f; stepsTotal++ }
                if (scanK >= N_POS - 1) { scanAcc = 0f; prevI = 25; prevJ = 25 }
            }
        }
        val ci = scanK / 26; val cj = scanK % 26
        // glide between positions at the slow speed (purely visual; the computation is per integer position)
        val f = if (playing && !manual && speedIdx == 0) ProcTex.smooth01(scanAcc / 0.55f) else 1f
        val gi = prevI + (ci - prevI) * f; val gj = prevJ + (cj - prevJ) * f
        kRig.pos = P3(IN.lx(gj + 1f), IN.ly(gi + 1f), 0f)
        // the selected kernel's map slides to the front of the conv-1 stack (the others keep their order behind it)
        val ease = if (dt == 0f) 1f else 1f - kotlin.math.exp(-dt * 6f)
        for (k in 0 until 8) {
            val want = C1.z(if (k == sel) 0 else if (k < sel) k + 1 else k)
            val z = c1Planes[k].pos.z
            if (kotlin.math.abs(want - z) > 1e-5f) c1Planes[k].pos = P3(0f, 0f, z + (want - z) * ease)
        }
        val zs = c1Planes[sel].pos.z
        outPix.pos = P3(C1.lx(cj.toFloat()), C1.ly(ci.toFloat()), zs + 0.004f)
        selFrame.pos = P3(0f, 0f, zs); selBack.pos = P3(0f, 0f, zs - 0.003f)
        // the 4 rays: corners of the kernel window on the back input plane → the output pixel
        val zb = if (SA > 0f) IN.z(0) + 0.012f else IN.z(IN.count - 1) - 0.012f   // the plane facing the conv-1 block
        val x0 = IN.lx(gj) - IN.px / 2; val x1 = IN.lx(gj + 2f) + IN.px / 2
        val y0 = IN.ly(gi) + IN.px / 2; val y1 = IN.ly(gi + 2f) - IN.px / 2
        val o = C1.world(C1.lx(cj.toFloat()), C1.ly(ci.toFloat()), zs)
        val corners = listOf(IN.world(x0, y0, zb), IN.world(x1, y0, zb), IN.world(x0, y1, zb), IN.world(x1, y1, zb))
        for (r in 0 until 4) CnnGfx.placeLine(rays[r], corners[r], o, 0.0011f)

        // ---- progressive reveal of the selected map while scanning (2 tiny quads edited in place)
        val scanning = playing && !manual && scanK < N_POS - 1
        for (k in 0 until 8) c1Planes[k].setMat(if (k != sel) matDim else if (scanning) matGhost else matSel)
        revA.visible = scanning && ci > 0; revB.visible = scanning
        revA.pos = P3(0f, 0f, zs + 0.0012f); revB.pos = revA.pos
        if (scanning && scanK != lastRevealK) {
            lastRevealK = scanK
            val z = 0f; val h = C1.size / 2; val p = C1.px
            var uv = A_C1.uv(sel, 0f, 0f, ci.toFloat(), 26f)
            CnnGfx.setQuad(revAm, -h, h, h, h - ci * p, z, uv[0], uv[1], uv[2], uv[3])
            uv = A_C1.uv(sel, ci.toFloat(), 0f, ci + 1f, cj + 1f)
            CnnGfx.setQuad(revBm, -h, h - ci * p, -h + (cj + 1) * p, h - (ci + 1) * p, z, uv[0], uv[1], uv[2], uv[3])
            meshEdited(revAm); meshEdited(revBm)
        }

        // ---- class bars (eased) + dense fan
        val pFrom = P2.world(0f, 0f, 0f)
        for (k in 0 until nOut) {
            val p = a.probs[k]
            barLen[k] += (p - barLen[k]) * (1f - kotlin.math.exp(-dt * 8f))
            if (dt == 0f) barLen[k] = p
            val L = (barLen[k] * BAR_L).coerceAtLeast(0.002f)
            val y = barY(k)
            bars[k].pos = P3(BAR_X0 + L / 2, y, 0.002f); bars[k].scale = P3(L, if (scene) 0.022f else 0.017f, 0.012f)
            val hot = if (scene) p > 0.5f else k == a.top()
            bars[k].setMat(add(if (hot) GOLD else CYAN, if (hot) 1f else 0.75f))
            CnnGfx.placeLine(fans[k], P3(pFrom.x + 0.07f, pFrom.y, pFrom.z), P3(BAR_X0 - 0.004f, y, 0f), 0.0009f)
            fans[k].setMat(add(if (hot) GOLD else CYAN, ((0.06f + 0.5f * p) * 10).roundToInt() / 10f))
        }

        updateRf()
        updateReadout(a, ci, cj)
        updateLabels()
    }

    private fun updateRf() {
        for (b in rfBoxes) b.visible = rfOn
        for (b in rfFills) b.visible = rfOn
        for (l in rfLines) l.visible = rfOn
        if (!rfOn) return
        // window of the chosen pool-2 cell in every block (rows and columns separately)
        val corners = ArrayList<List<P3>>()
        for ((l, b) in BLOCKS.withIndex()) {
            val wr = ReceptiveField.window(4, rfI, l); val wc = ReceptiveField.window(4, rfJ, l)
            val cx = (b.lx(wc[0].toFloat()) + b.lx(wc[1].toFloat())) / 2; val cy = (b.ly(wr[0].toFloat()) + b.ly(wr[1].toFloat())) / 2
            rfBoxes[l].pos = P3(cx, cy, 0f); rfFills[l].pos = P3(cx, cy, 0f)
            val hw = (wc[1] - wc[0] + 1) * b.px / 2; val zf = (b.depth / 2 + 0.012f) * if (SA > 0f) 1f else -1f
            // [right face corners (toward the next block), left face corners (toward the previous one)]
            corners += listOf(b.world(cx - hw, cy + hw, zf), b.world(cx + hw, cy + hw, zf), b.world(cx - hw, cy - hw, zf), b.world(cx + hw, cy - hw, zf),
                b.world(cx - hw, cy + hw, -zf), b.world(cx + hw, cy + hw, -zf), b.world(cx - hw, cy - hw, -zf), b.world(cx + hw, cy - hw, -zf))
        }
        for (g in 0 until 4) for (c in 0 until 4) CnnGfx.placeLine(rfLines[g * 4 + c], corners[g][c], corners[g + 1][4 + c], 0.0011f)
    }

    // ================================================================ text
    private fun kernelName(k: Int): String =
        if (learned) "#${k + 1}" else (if (scene) HAND_SCENE[k] else HAND_DIGIT[k]).let { tr(it.first, it.second) }

    private fun updateReadout(a: CnnModel.Acts, i: Int, j: Int) {
        val m = model ?: return
        val c = chans
        val l1 = tr("张量  %s → 卷积 8×26×26 → ReLU+池化 8×13×13 → 卷积 16×11×11 → 池化 16×5×5 → 全连接 %d",
            "Tensors  %s → conv 8×26×26 → ReLU+pool 8×13×13 → conv 16×11×11 → pool 16×5×5 → dense %d").format(a.input.shape, nOut)
        val p = CnnOps.partials(a.input, m.w1, sel, i, j)
        fun nz(v: Float) = if (kotlin.math.abs(v) < 0.005f) 0f else v      // no "-0.00"
        val parts = (0 until c).joinToString("  ") { "${CH_NAME[it]} %+.2f".format(nz(p[it])) }
        val y = a.conv1[sel, i, j]
        val l2 = tr("核「%s」%d×3×3 在 (%d,%d)：%s  + b %+.2f = %+.2f → ReLU %.2f",
            "Kernel “%s” %d×3×3 at (%d,%d): %s  + b %+.2f = %+.2f → ReLU %.2f").format(kernelName(sel), c, i, j, parts, nz(m.b1[sel]), nz(y), maxOf(0f, y))
        val l3 = if (rfOn) tr("感受野：末层格 (%d,%d) ← 卷积② 2×2 ← 池化① 4×4 ← 卷积① 8×8 ← 输入 10×10 像素（行 %d–%d，列 %d–%d）",
            "Receptive field: last cell (%d,%d) ← conv② 2×2 ← pool① 4×4 ← conv① 8×8 ← input 10×10 px (rows %d–%d, cols %d–%d)")
            .format(rfI, rfJ, rfI * 4, rfI * 4 + 9, rfJ * 4, rfJ * 4 + 9)
        else if (!scene) {
            val t = a.top()
            tr("预测 %d（%.0f%%）· %.1fk 参数小模型，没见过的字体上准确率 %.1f%% · 同一组权重扫遍每个位置（权重共享）",
                "Prediction %d (%.0f%%) · toy CNN, %.1fk params, %.1f%% on unseen fonts · one set of weights at every position")
                .format(t, a.probs[t] * 100, m.params / 1000f, lab.heldOut(m.name) * 100)
        } else {
            val present = (0 until 5).filter { a.probs[it] > 0.5f }.joinToString(" ") { tr(CLASS[it].first, CLASS[it].second) + " %.0f%%".format(a.probs[it] * 100) }
            val vt = tr("植被/草坪判对率 %.0f%%/%.0f%%", "veg/turf acc %.0f%%/%.0f%%").format(lab.labelAcc(m.name, "veg") * 100, lab.labelAcc(m.name, "turf") * 100)
            tr("检出：%s · %s · %s", "Present: %s · %s · %s").format(present.ifEmpty { "—" }, vt,
                if (nir) tr("通道在核里相加", "channels sum in the kernel") else tr("没有 NIR：草坪像草", "no NIR: turf looks like grass"))
        }
        readoutText = "$l1\n$l2\n$l3"
    }

    private fun updateLabels() {
        labels.clear()
        val c = chans
        fun top(b: Block, dy: Float = 0.045f) = P3(b.c.x, BY + b.size / 2 + dy, b.c.z)
        labels += LabelSpec(tr("输入 %d×28×28 · 合成数据", "Input %d×28×28 · synthetic").format(c), top(IN))
        labels += LabelSpec(tr("卷积① 8×26×26", "Conv① 8×26×26"), top(C1))
        labels += LabelSpec(tr("ReLU+池化 8×13×13", "ReLU+pool 8×13×13"), top(P1, 0.09f))
        labels += LabelSpec(tr("卷积② 16×11×11", "Conv② 16×11×11"), P3(C2.c.x, BY - C2.size / 2 - 0.07f, C2.c.z))
        labels += LabelSpec(tr("池化 16×5×5 → 全连接", "Pool 16×5×5 → dense"), top(P2, 0.12f))
        labels += LabelSpec(if (learned) tr("卷积核架（学到的）", "Kernel shelf (learned)") else tr("卷积核架（手工）", "Kernel shelf (hand-made)"),
            P3(SHELF_X0 + 3.5f * SHELF_DX, SHELF_Y + 0.075f, SHELF_Z))
    }

    // ================================================================ actions (panel, pinch, lesson)
    fun setScene(on: Boolean) { if (scene == on) return; scene = on; sel = 0; restartScan(); dirty = true; stepsAtSel = stepsTotal }
    fun setLearned(on: Boolean) { if (learned == on) return; learned = on; restartScan(); dirty = true; stepsAtSel = stepsTotal }
    fun setNir(on: Boolean) { if (nir == on) return; nir = on; restartScan(); dirty = true; stepsAtSel = stepsTotal }
    fun setSpeed(k: Int) { speedIdx = k.coerceIn(0, 2) }
    fun setRf(on: Boolean) { rfOn = on }
    fun select(k: Int) { sel = k.coerceIn(0, 7); restartScan(); if (!dirty) { applyKernels(); applyLang() }; stepsAtSel = stepsTotal }
    fun nextSample() { if (!scene) digitPos = (digitPos + 1) % lab.digitShow.size else scenePos = (scenePos + 1) % lab.sceneShow.size; dirty = true; restartScan() }
    /** Show the first display digit with label [d] (lesson / challenge). */
    fun showDigit(d: Int) { val p = lab.digitShow.indexOfFirst { lab.digits.labels[it] == d }; if (p >= 0 && p != digitPos) { digitPos = p; dirty = true; restartScan() } }
    fun play(on: Boolean) { playing = on; if (on) manual = false }
    private fun restartScan() { scanK = 0; scanAcc = 0f; holdT = 0f; prevI = 0; prevJ = 0; manual = false; lastRevealK = -1; if (!playing) playing = true }

    /** Positions the kernel has visited since the current kernel was picked. */
    val scannedSinceSelect: Int get() = stepsTotal - stepsAtSel
    val currentDigit: Int get() = if (scene) -1 else lab.digits.labels[lab.digitShow[digitPos]]
    /** Learned kernel whose map lights the slant of the shown 7 most (vs its top bar) — the challenge answer. */
    private fun slant(): FloatArray = SlantProbe.selectivity(lab.acts(false, true, true, lab.digitShow[digitPos]))
    val bestSlant: Int get() = slant().let { s -> s.indices.maxBy { s[it] } }

    override val tapTargets get() = listOf(
        TapTarget("Sample", "图像 → 下一张", "Image → next"),
        TapTarget("KernelBox", "卷积核 → 暂停/继续", "Kernel → pause/run") { playing },
    ) + (0 until 8).map { k ->
        val n = if (learned) ("#${k + 1}" to "#${k + 1}") else if (scene) HAND_SCENE[k] else HAND_DIGIT[k]
        TapTarget("Shelf_$k", n.first, n.second) { sel == k }
    } + TapTarget("Pool2", "末层 → 感受野", "Last layer → receptive field") { rfOn }

    override fun isTarget(name: String) = name == "Sample" || name == "KernelBox" || name == "Pool2" || name.startsWith("Shelf_")

    override fun onTap(name: String) {
        when {
            name == "Sample" -> nextSample()
            name == "KernelBox" -> { if (manual) { manual = false; playing = true } else playing = !playing }
            name == "Pool2" -> {
                if (!rfOn) rfOn = true
                else { val path = RF_PATH; val k = (path.indexOf(rfI * 5 + rfJ) + 1) % path.size; rfI = path[k] / 5; rfJ = path[k] % 5 }
                rfTaps++
            }
            name.startsWith("Shelf_") -> name.removePrefix("Shelf_").toIntOrNull()?.let { select(it) }
        }
    }
    private val RF_PATH = listOf(12, 6, 8, 18, 16, 0, 24, 2, 22)   // centre first, then around

    override fun tapMessage(name: String): String? = when {
        name == "Sample" -> if (!scene) tr("下一张：数字 %d —— 卷积核重新从左上角扫起", "Next: digit %d — the kernel restarts top-left").format(currentDigit)
                            else tr("下一块多光谱场景（%d/%d）", "Next multispectral tile (%d/%d)").format(scenePos + 1, lab.sceneShow.size)
        name == "KernelBox" -> if (playing) tr("继续扫描", "Scanning again") else tr("暂停 —— 捏住它拖动，可以手动滑过输入", "Paused — pinch and drag it to slide by hand")
        name == "Pool2" -> tr("末层格 (%d,%d) 往回看到输入的 10×10 像素", "Last-layer cell (%d,%d) sees 10×10 input pixels").format(rfI, rfJ)
        name.startsWith("Shelf_") -> tr("用核「%s」扫描 → 点亮特征图 %d", "Kernel “%s” scans → feature map %d").format(kernelName(sel), sel + 1)
        else -> null
    }

    // ---- drag: slide the kernel by hand / move the receptive-field cell
    private const val DRAG_GAIN = 1.4f
    override fun isDragTarget(name: String) = name == "KernelBox" || name == "Pool2"
    override fun onDragStart(name: String) {
        if (name == "KernelBox") { manual = true; dragI = (scanK / 26).toFloat(); dragJ = (scanK % 26).toFloat() }
        else { rfOn = true; rfDragI = rfI.toFloat(); rfDragJ = rfJ.toFloat() }
    }
    override fun onDrag(name: String, dx: Float, dy: Float, dz: Float) {
        // project the hand motion onto the block's own axes (it is yawed): +x along the rows, +y up
        val along = dx * CA - dz * SA
        if (name == "KernelBox") {
            // gain 1.4: one input pixel per ~7 mm of hand travel, the whole 26-position sweep in ~19 cm
            dragJ = (dragJ + along / IN.px * DRAG_GAIN).coerceIn(0f, 25f); dragI = (dragI - dy / IN.px * DRAG_GAIN).coerceIn(0f, 25f)
            scanK = dragI.roundToInt() * 26 + dragJ.roundToInt(); prevI = scanK / 26; prevJ = scanK % 26
        } else {
            rfDragJ = (rfDragJ + along / 0.03f).coerceIn(0f, 4f); rfDragI = (rfDragI - dy / 0.03f).coerceIn(0f, 4f)
            rfI = rfDragI.roundToInt(); rfJ = rfDragJ.roundToInt()
        }
    }
    override fun onDragEnd(name: String) {}
    /** (row, col) of the kernel and the receptive-field cell — for tests. */
    fun debugCursor() = (scanK / 26) to (scanK % 26)
    fun debugRf() = rfI to rfJ

    override fun reviewVariants(): Map<String, () -> Unit> = mapOf(
        "scene" to { setScene(true); setLearned(false); setNir(true); select(0); setSpeed(2); repeat(200) { update(1f / 60f) } },
        "rf" to { setScene(false); setLearned(true); setRf(true); onTap("Pool2") },
    )

    // ================================================================ panel
    @Composable
    override fun Panel() {
        LabPanel(readout) {
            LabRow {
                SegmentControl {
                    SegmentItem(selected = !scene, onClick = { setScene(false) }, title = { Text(tr("RGB 数字", "RGB digits")) })
                    SegmentItem(selected = scene, onClick = { setScene(true) }, title = { Text(tr("多光谱场景", "Multispectral")) })
                }
                Spacer(Modifier.width(14.dp))
                SegmentControl {
                    SegmentItem(selected = learned, onClick = { setLearned(true) }, title = { Text(tr("学到的核", "Learned")) })
                    SegmentItem(selected = !learned, onClick = { setLearned(false) }, title = { Text(tr("手工核", "Hand-made")) })
                }
                Spacer(Modifier.width(14.dp))
                SegmentControl {
                    listOf(tr("慢", "Slow"), tr("中", "Med"), tr("快", "Fast")).forEachIndexed { k, l ->
                        SegmentItem(selected = speedIdx == k, onClick = { setSpeed(k) }, title = { Text(l) })
                    }
                }
                if (scene) {
                    Spacer(Modifier.width(14.dp))
                    SegmentControl {
                        SegmentItem(selected = nir, onClick = { setNir(true) }, title = { Text("RGB+NIR") })
                        SegmentItem(selected = !nir, onClick = { setNir(false) }, title = { Text(tr("仅 RGB", "RGB only")) })
                    }
                }
                Spacer(Modifier.width(14.dp))
                LabSwitch(tr("感受野", "Receptive field"), rfOn) { setRf(it) }
            }
        }
    }

    // ================================================================ guided track (导学)
    private var stepBase = 0
    private var rfBase = 0

    override val lesson: Lesson by lazy {
        Lesson(
            listOf(
                LessonStep(
                    say = Bi("一个 3×3×3 的卷积核扫过一张 RGB 图（3 个通道），会输出几张特征图？", "A 3×3×3 kernel sweeps over an RGB image (3 channels). How many feature maps come out?"),
                    options = listOf(
                        Bi("1 张", "1"),
                        Bi("3 张（每个通道一张）", "3 (one per channel)"),
                        Bi("9 张", "9"),
                        Bi("27 张", "27"),
                    ),
                    answer = 0,
                ),
                LessonStep(
                    say = Bi("看金色的核在输入块上滑动：每停一处，27 个像素各乘一个权重再相加，只点亮右边特征图里的 1 个像素。", "Watch the gold kernel slide over the input block: at each stop 27 pixels × 27 weights are summed into ONE pixel of a feature map."),
                    target = "KernelBox",
                    onEnter = { stepBase = stepsTotal },
                    done = { !scene && stepsTotal - stepBase >= 30 },
                    autoplay = { setScene(false); setSpeed(2); play(true) },
                    reveal = true,
                    insight = Bi("1 张！核在深度上贯穿全部通道，R、G、B 在核里面就加成一个数。8 个核 → 8 张特征图，叠成下一个方块。", "One! The kernel spans all channels in depth, so R, G and B are summed inside it. 8 kernels → 8 maps, stacked into the next block."),
                ),
                LessonStep(
                    say = Bi("多光谱场景有 4 层：R、G、B 和近红外 NIR。哪个手工核能挑出真植被，却不理会一样绿的人造草坪？", "A multispectral tile has 4 planes: R, G, B and near-infrared. Which hand-made kernel finds real plants but ignores equally green artificial turf?"),
                    options = listOf(
                        Bi("绿色核 G − (R+B)/2", "Green: G − (R+B)/2"),
                        Bi("NIR − R 核", "NIR − R"),
                        Bi("竖边核", "V-edge"),
                        Bi("模糊核", "Blur"),
                    ),
                    answer = 1,
                ),
                LessonStep(
                    say = Bi("切到「多光谱场景」+「手工核」，捏架子上的「NIR−R」核，看它的特征图哪里亮。", "Switch to Multispectral + Hand-made and pinch the “NIR − R” kernel on the shelf: where does its map light up?"),
                    target = "Shelf_0",
                    done = { scene && !learned && nir && sel == 0 && scannedSinceSelect >= 300 },
                    autoplay = { setScene(true); setLearned(false); setNir(true); onTap("Shelf_0"); setSpeed(2) },
                    reveal = true,
                    insight = Bi("NIR−R 只点亮真植被：叶子强烈反射近红外、吸收红光（NDVI 的原理）。草坪在 RGB 里和草一样绿，近红外却很暗。试试「仅 RGB」：草坪判对率从 99% 掉到 63%。", "NIR − R lights only real plants: leaves reflect near-infrared and absorb red (the NDVI idea). Turf is as green as grass in RGB but dark in NIR. Try “RGB only”: turf accuracy drops 99% → 63%."),
                ),
                LessonStep(
                    say = Bi("回到 RGB 数字 7。捏最后那个小方块（16×5×5），看它的一个格子往回能看到输入的多大一块。", "Back to an RGB digit 7. Pinch the last small block (16×5×5): how much of the input can one of its cells see?"),
                    target = "Pool2",
                    onEnter = { rfBase = rfTaps; setScene(false); setLearned(true); showDigit(7) },
                    done = { rfOn && rfTaps > rfBase },
                    autoplay = { onTap("Pool2") },
                    insight = Bi("3 → 4 → 8 → 10 像素：每过一层，视野变大。25 个格子合起来覆盖整张图，全连接层再把它们的证据加权投票。", "3 → 4 → 8 → 10 pixels: every layer widens the view. The 25 cells together cover the image; the dense layer weighs their evidence."),
                ),
            ),
            challenge = Challenge(
                say = Bi("挑战：在架子上的 8 个学到的核里，找出最爱 7 那一道「斜线」（而不是上面横杠）的核。", "Challenge: among the 8 learned kernels on the shelf, find the one that loves the 7’s slant (not its top bar)."),
                progress = {
                    if (scene || !learned) tr("先切回「RGB 数字」+「学到的核」", "Switch to RGB digits + Learned first")
                    else if (currentDigit != 7) tr("捏左边的图换到数字 7（现在是 %d）", "Pinch the picture until it shows a 7 (now %d)").format(currentDigit)
                    else slant().let { v -> val rank = 1 + v.count { it > v[sel] }
                        tr("核 #%d：斜线比横杠亮 %+.2f，8 个里排第 %d", "Kernel #%d: slant minus bar %+.2f, rank %d of 8").format(sel + 1, v[sel], rank) }
                },
                done = { !scene && learned && currentDigit == 7 && sel == bestSlant },
                autoplay = { setScene(false); setLearned(true); showDigit(7); select(bestSlant) },
            ),
        )
    }
}
