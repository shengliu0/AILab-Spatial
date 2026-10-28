// 词向量空间 · Word Embeddings (Mikolov et al. 2013) — a REAL mini skip-gram model you can walk around.
//
// LESSON: a model that only learns to predict neighbouring words ends up placing words in a space where
// DIRECTIONS mean something (man→woman ≈ king→queen, country→capital); analogy = vector arithmetic, a
// parallelogram A − B + C ≈ D.
//
// SCENE (volume metres): a turntable plinth; above it a cloud of 96 glowing word dots (colour = category),
// positioned by a 3-D PCA of the 32-D unit word vectors (one PCA per relation view, see core/embedding).
// Word names are baked text tiles floating above the dots. 8 words per view are pinchable (Pin0..7):
//   1st pinch  A          -> its 5 nearest neighbours (cosine in 32-D) light up with threads
//   2nd pinch  B          -> gold arrow B→A = the difference vector A − B
//   3rd pinch  C          -> a second gold arrow from C by the same vector lands at A − B + C; the
//                            parallelogram is drawn and the top-3 candidates are ringed (A, B, C excluded)
// The knob on the plinth rotates the cloud (pinch = 45°, pinch-drag = free).
// The model (core/embedding) is trained by tools/train_embedding.py on a generated corpus.

package com.bytedance.ailab.demos.embedding

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import com.bytedance.ailab.core.P3
import com.bytedance.ailab.core.embedding.WordAtlas
import com.bytedance.ailab.core.embedding.WordSpace
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
import com.pico.spatial.ui.design.Text
import java.io.File
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

object EmbeddingDemo : SpecDemo() {
    override val id = "embedding"
    override val title = "词向量空间"   // i18n-ok (Chinese title; English in DemoRegistry.TITLE_EN)
    override val credit = "Mikolov et al. · 2013"

    // ---------------------------------------------------------------- model (immutable)
    private val ws by lazy { WordSpace(LabAssets.bytes("embedding/emb.bin")) }
    private val atlas by lazy { WordAtlas(LabAssets.bytes("embedding/words.bin")) }
    private val shown by lazy { ws.shownIds }
    private fun w(en: String): Int = ws.en.indexOf(en).also { require(it >= 0) { "no word $en" } }
    private fun name(i: Int): String = if (i < 0) "?" else tr(ws.zh[i], ws.en[i])

    // ---------------------------------------------------------------- layout
    private val C0 = P3(0f, 0.115f, -0.02f)        // cloud centre (volume)
    private const val R_XZ = 0.56f                 // cloud radius in the turntable plane (spins inside |x|,|z| < 0.6)
    private const val H_Y = 0.25f                 // cloud half height
    private const val TILE_UP = 0.034f
    private const val VIEWS = 4
    private val SPEEDS = floatArrayOf(0f, 10f, 20f, 40f)   // deg/s: off / slow / mid / fast
    private val PIN_WORDS = listOf(
        listOf("king", "queen", "man", "woman", "prince", "princess", "boy", "girl"),
        listOf("China", "Beijing", "Japan", "Tokyo", "France", "Paris", "UK", "London"),
        listOf("big", "bigger", "small", "smaller", "fast", "faster", "cold", "colder"),
        listOf("king", "queen", "man", "woman", "Japan", "Tokyo", "France", "Paris"),
    )
    /** Category colours (sRGB): male, female, country, capital, adjective, comparative, animal, fruit, colour. */
    private val CAT = listOf(Rgb(0.35f, 0.66f, 1.0f), Rgb(1.0f, 0.42f, 0.70f), Rgb(0.25f, 0.95f, 0.62f), Rgb(1.0f, 0.78f, 0.25f),
        Rgb(0.70f, 0.55f, 1.0f), Rgb(1.0f, 0.52f, 0.22f), Rgb(0.95f, 0.68f, 0.50f), Rgb(0.72f, 1.0f, 0.32f), Rgb(0.82f, 0.86f, 0.98f))
    private val GOLD = Rgb(1.0f, 0.82f, 0.32f)
    private val CYAN = Rgb(0.45f, 0.90f, 1.0f)

    // ---------------------------------------------------------------- state (reset in buildSpec)
    var preset = 0
        private set
    private var fromView = 0
    private var morph = 1f
    var showAll = false
    var spinIdx = 0
    private var yaw = -0.35f
    private var yawTravel = 0f
    /** Selected words, in pinch order: A, B, C. */
    private val sel = ArrayList<Int>()
    private var grow = 1f                         // analogy / arrow animation 0..1
    private var dirty = true
    private var enShown = -1
    private var lastMsg = Bi("", "")
    private var neighbours: List<Pair<Int, Double>> = emptyList()
    private var cands: List<Pair<Int, Double>> = emptyList()
    private var spinDragging = false

    // ---------------------------------------------------------------- precomputed views
    /** Cloud-local positions per view [view][word] (only shown words are non-null). */
    private val viewPos: Array<Array<P3?>> by lazy {
        Array(VIEWS) { v ->
            val raw = Array<DoubleArray?>(ws.v) { if (ws.shown[it]) ws.project(v, it) else null }
            // fit the view's own words into the turntable cylinder; other words (which project near the centre) are clamped
            val fitIds = if (v < 3) ws.relations[v].flatMap { listOf(it.first, it.second) } else shown.toList()
            var rxz = 1e-9; var ry = 1e-9
            for (i in fitIds) { val p = raw[i]!!; rxz = max(rxz, sqrt(p[0] * p[0] + p[2] * p[2])); ry = max(ry, abs(p[1])) }
            // per-axis scale (a linear map keeps parallelograms parallelograms), anisotropy capped at 2.2x
            var sxz = R_XZ / rxz; var sy = H_Y / ry
            sxz = min(sxz, 2.2 * sy); sy = min(sy, 2.2 * sxz)
            Array(ws.v) { i -> raw[i]?.let {
                var x = it[0] * sxz; var y = it[1] * sy; var z = it[2] * sxz
                val r = sqrt(x * x + z * z); if (r > R_XZ) { x *= R_XZ / r; z *= R_XZ / r }
                y = y.coerceIn(-H_Y.toDouble(), H_Y.toDouble())
                P3(x.toFloat(), y.toFloat(), z.toFloat())
            } }
        }
    }
    /** Words that belong to the relation of each view (bright, labelled); the "all" view: every shown word. */
    private val viewWords: Array<Set<Int>> by lazy {
        Array(VIEWS) { v -> if (v < 3) ws.relations[v].flatMap { listOf(it.first, it.second) }.toSet() else shown.toSet() }
    }
    private val cur = HashMap<Int, P3>()
    /** Tile height above its dot (clears the dot and its rings). */
    private val tileUp = HashMap<Int, Float>()
    private fun posOf(i: Int): P3 = cur[i] ?: P3(0f, 0f, 0f)

    // ---------------------------------------------------------------- meshes and materials
    private val quadMesh = quad(0f, 0f, 1f, 1f)
    private val glowTex by lazy {
        OptGeo.lumTex(48, 48) { u, v ->
            val r2 = ((u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f)) * 4f
            (exp(-r2 * 14f) * 1.0f + 0.30f * exp(-r2 * 3.2f)).coerceAtMost(1f) * (1f - r2).coerceIn(0f, 1f)
        }
    }
    private val atlasImg: Image by lazy {
        val img = Image(atlas.w, atlas.h)
        for (k in 0 until atlas.w * atlas.h) img.px[k] = ((atlas.alpha[k].toInt() and 0xFF) shl 24) or 0xFFFFFF
        img
    }
    private val tileMeshes = HashMap<Int, MeshData>()
    private fun tileMesh(i: Int, en: Boolean): MeshData = tileMeshes.getOrPut(i * 2 + if (en) 1 else 0) {
        val r = atlas.rect(i, en); val W = atlas.w.toFloat(); val H = atlas.h.toFloat()
        quad((r[0] + 0.5f) / W, (r[1] + 0.5f) / H, (r[0] + r[2] - 0.5f) / W, (r[1] + r[3] - 0.5f) / H)
    }
    private fun aspect(i: Int, en: Boolean): Float { val r = atlas.rect(i, en); return r[2].toFloat() / r[3] }

    private fun dim(c: Rgb, k: Float) = Rgb(c.r * k, c.g * k, c.b * k)
    private val dotBright by lazy { CAT.map { Mat.Unlit(it, tex = glowTex, blend = Blend.ADD) } }
    private val dotDim by lazy { CAT.map { Mat.Unlit(dim(it, 0.42f), tex = glowTex, blend = Blend.ADD) } }
    private val dotHot by lazy { CAT.map { Mat.Unlit(ProcTex.okLerp(it, Rgb(1f, 1f, 1f), 0.45f), tex = glowTex, blend = Blend.ADD) } }
    private val tileMats by lazy { CAT.map { Mat.Unlit(ProcTex.okLerp(it, Rgb(1f, 1f, 1f), 0.55f), tex = atlasImg, blend = Blend.TRANSPARENT) } }
    private val tileHot by lazy { Mat.Unlit(Rgb(1f, 1f, 1f), tex = atlasImg, blend = Blend.TRANSPARENT) }
    private val tileGold by lazy { Mat.Unlit(Rgb(1f, 0.90f, 0.55f), tex = atlasImg, blend = Blend.TRANSPARENT) }
    private val threadMats = List(11) { a -> Mat.Unlit(Rgb(0.80f, 0.90f, 1.0f), alpha = a / 10f, blend = Blend.ADD) }
    private val goldMat = Mat.Unlit(GOLD, blend = Blend.ADD)
    private val cyanMat = Mat.Unlit(CYAN, alpha = 0.85f, blend = Blend.ADD)
    private val fillMats = List(6) { a -> Mat.Unlit(Rgb(1.0f, 0.93f, 0.70f), alpha = a * 0.011f, blend = Blend.ADD) }
    private val pinMat by lazy { Mat.Unlit(Rgb(0.85f, 0.92f, 1.0f), alpha = 0.9f, tex = glowTex, blend = Blend.ADD) }
    private val ringGold = Mat.Unlit(GOLD, blend = Blend.ADD)
    private val ringSilver = Mat.Unlit(Rgb(0.75f, 0.80f, 0.90f), alpha = 0.8f, blend = Blend.ADD)
    private val ringSel = Mat.Unlit(Rgb(0.95f, 0.97f, 1.0f), blend = Blend.ADD)
    private val residMat = Mat.Unlit(Rgb(1.0f, 0.55f, 0.35f), alpha = 0.9f, blend = Blend.ADD)
    private val slabTex by lazy { ProcTex.bakePlanet(256, 128, Rgb(0.05f, 0.055f, 0.07f), Rgb(0.09f, 0.10f, 0.12f), Rgb(0.15f, 0.16f, 0.19f), 0f, 91) }
    private val slab get() = Mat.Pbr(Rgb(1f, 1f, 1f), metallic = 0f, roughness = 0.55f, tex = slabTex)
    private val alu = Mat.Pbr(Rgb(0.80f, 0.82f, 0.86f), metallic = 1f, roughness = 0.30f)
    private val turnMat = Mat.Unlit(Rgb(0.30f, 0.55f, 0.85f), alpha = 0.55f, blend = Blend.ADD)
    private val knobMat = Mat.Pbr(Rgb(0.72f, 0.74f, 0.78f), metallic = 1f, roughness = 0.28f, emissive = Rgb(0.05f, 0.12f, 0.22f))
    private val axisMat = Mat.Unlit(Rgb(0.22f, 0.30f, 0.42f), alpha = 0.7f, blend = Blend.ADD)

    // ---------------------------------------------------------------- nodes
    private lateinit var cloud: Node
    private val dots = HashMap<Int, Node>()
    private val tiles = HashMap<Int, Node>()
    private lateinit var pins: Array<Node>
    private lateinit var threads: Array<Node>
    private lateinit var arrAB: Pair<Node, Node>
    private lateinit var arrCT: Pair<Node, Node>
    private lateinit var dashBC: Array<Node>
    private lateinit var dashAT: Array<Node>
    private lateinit var fill: Node
    private lateinit var fillMesh: MeshData
    private lateinit var targetRing: Node
    private lateinit var resid: Node
    private lateinit var rankRings: Array<Node>
    private lateinit var selRings: Array<Node>
    private lateinit var knob: Node
    private const val DASHES = 7
    private const val PIN_S = 0.022f

    override fun buildSpec(cacheDir: File): Node {
        // reset EVERYTHING (rebuild idempotence)
        preset = 0; fromView = 0; morph = 1f; showAll = false; spinIdx = 0; yaw = -0.35f; yawTravel = 0f
        sel.clear(); grow = 1f; dirty = true; enShown = -1; lastMsg = Bi("", ""); neighbours = emptyList(); cands = emptyList()
        spinDragging = false; cur.clear(); tileUp.clear(); dots.clear(); tiles.clear(); stepYaw = 0f; fromSnapshot = null; travelAtStep = 0f

        val root = Node("EmbRoot")
        root.add("Plinth", MeshGen.box(1.70f, 0.03f, 1.12f), slab, P3(0f, -0.255f, 0f))
        root.add("PlinthBand", MeshGen.box(1.71f, 0.008f, 1.13f), alu, P3(0f, -0.255f, 0f))
        // turntable: two glowing rings on the plinth under the cloud
        root.add("Turn0", MeshGen.annulus(0.585f, 0.592f, 96), turnMat, P3(0f, -0.238f, -0.02f))
        root.add("Turn1", MeshGen.annulus(0.30f, 0.304f, 72), turnMat, P3(0f, -0.238f, -0.02f))
        knob = root.add("SpinKnob", MeshGen.cylinder(0.045f, 0.024f, 32), knobMat, P3(0f, -0.226f, 0.46f)).also { it.tapRadius = 0.055f }
        knob.add("KnobMark", MeshGen.box(0.008f, 0.004f, 0.05f), goldMat, P3(0f, 0.013f, 0f))

        cloud = root.add(Node("Cloud").also { it.pos = C0 })
        val cyl = MeshGen.cylinder(1f, 1f, 8)
        // faint PCA axes through the cloud centre
        for ((k, d) in listOf(P3(1f, 0f, 0f), P3(0f, 1f, 0f), P3(0f, 0f, 1f)).withIndex()) {
            val len = if (k == 1) 2 * H_Y else 2 * R_XZ
            cloud.add("Axis$k", cyl, axisMat).also { it.scale = P3(0.0012f, len, 0.0012f); it.rot = Q.fromTo(0f, 1f, 0f, d.x, d.y, d.z) }
        }
        for (i in shown) {
            dots[i] = cloud.add("Dot$i", quadMesh, dotDim[ws.category[i]]).also { it.billboard = true; it.uniformScale(0.03f) }
            tiles[i] = cloud.add("Tile$i", tileMesh(i, false), tileMats[ws.category[i]]).also { it.billboard = true; it.visible = false }
        }
        // pinchable words carry a small white core (the halo hugs it); pinch radius 0.05 m
        pins = Array(8) { k -> cloud.add("Pin$k", quadMesh, pinMat).also { it.billboard = true; it.uniformScale(PIN_S); it.tapRadius = 0.05f / PIN_S } }
        threads = Array(5) { k -> cloud.add("Thr$k", cyl, threadMats[5]).also { it.visible = false } }
        val cone = coneMesh(12)
        fun arrow(n: String) = Pair(cloud.add("${n}s", cyl, goldMat).also { it.visible = false }, cloud.add("${n}h", cone, goldMat).also { it.visible = false })
        arrAB = arrow("ArrAB"); arrCT = arrow("ArrCT")
        dashBC = Array(DASHES) { k -> cloud.add("DashBC$k", cyl, cyanMat).also { it.visible = false } }
        dashAT = Array(DASHES) { k -> cloud.add("DashAT$k", cyl, cyanMat).also { it.visible = false } }
        fillMesh = MeshData(FloatArray(24), FloatArray(24).also { for (k in 0 until 8) it[k * 3 + 2] = if (k < 4) 1f else -1f }, FloatArray(16),
            intArrayOf(0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6))
        fill = cloud.add("Fill", fillMesh, fillMats[0]).also { it.visible = false }
        val ring = ringMesh(0.80f, 1.0f)
        targetRing = cloud.add("Target", ring, ringGold).also { it.billboard = true; it.visible = false; it.uniformScale(0.026f) }
        resid = cloud.add("Resid", cyl, residMat).also { it.visible = false }
        rankRings = Array(3) { k -> cloud.add("Rank$k", ring, if (k == 0) ringGold else ringSilver).also { it.billboard = true; it.visible = false } }
        selRings = Array(3) { k -> cloud.add("SelRing$k", ring, ringSel).also { it.billboard = true; it.visible = false } }

        applyPositions()
        return root
    }

    // ---------------------------------------------------------------- actions (also used by the lesson)

    fun setPreset(p: Int) {
        if (p == preset && morph >= 1f) return
        // freeze the current (possibly mid-morph) layout as the start of the new morph
        val start = HashMap(cur)
        fromSnapshot = start
        preset = p.coerceIn(0, VIEWS - 1); morph = 0f; dirty = true
    }
    private var fromSnapshot: Map<Int, P3>? = null

    fun clearSelection() { sel.clear(); grow = 1f; dirty = true }

    /** Pinch word [i]: A, then B (difference vector), then C (analogy); a 4th pinch starts over. */
    fun pick(i: Int) {
        if (sel.size >= 3) sel.clear()
        if (sel.isNotEmpty() && sel.last() == i) { sel.removeAt(sel.size - 1); dirty = true; return }
        if (i in sel) return
        sel += i; grow = if (sel.size >= 2) 0f else 1f; dirty = true
    }

    fun runAnalogy(a: Int, b: Int, c: Int) { sel.clear(); sel += a; sel += b; sel += c; grow = 0f; dirty = true }

    fun rotateBy(rad: Float) { stepYaw += rad }
    private var stepYaw = 0f

    val analogyShown get() = sel.size == 3 && grow >= 1f
    private fun isAnalogy(a: String, b: String, c: String) = sel.size == 3 && sel[0] == w(a) && sel[1] == w(b) && sel[2] == w(c)

    // ---------------------------------------------------------------- frame

    override fun update(dt: Float) {
        val en = if (tr("zh", "en") == "en") 1 else 0
        if (en != enShown) { enShown = en; for ((i, t) in tiles) t.setMesh(tileMesh(i, en == 1)); dirty = true }

        // rotation: auto-spin + animated 45° steps
        val spin = SPEEDS[spinIdx.coerceIn(0, 3)] * PI.toFloat() / 180f * dt
        val step = if (abs(stepYaw) > 1e-4f) stepYaw * min(1f, dt * 6f) else stepYaw.also { stepYaw = 0f }
        stepYaw -= step
        yaw += spin + step; yawTravel += abs(spin + step)
        cloud.rot = Q.axisAngle(0f, 1f, 0f, yaw)

        var moving = false
        if (morph < 1f) { morph = min(1f, morph + dt / 0.9f); moving = true; dirty = true }
        if (grow < 1f) { grow = min(1f, grow + dt / 1.3f); dirty = true }
        if (moving || cur.isEmpty()) applyPositions()
        if (dirty) { dirty = false; applySelection() }
        updateLabels()
        readoutText = readout3()
    }

    private fun smooth(t: Float) = t * t * (3f - 2f * t)

    private fun applyPositions() {
        val s = smooth(morph); val from = fromSnapshot
        for (i in shown) {
            val b = viewPos[preset][i]!!
            val a = from?.get(i) ?: b
            cur[i] = P3(a.x + (b.x - a.x) * s, a.y + (b.y - a.y) * s, a.z + (b.z - a.z) * s)
        }
        if (morph >= 1f) fromSnapshot = null
        for (i in shown) { val p = posOf(i); dots[i]!!.pos = p; tiles[i]!!.pos = P3(p.x, p.y + (tileUp[i] ?: TILE_UP), p.z) }
        val pw = PIN_WORDS[preset]
        for (k in 0 until 8) pins[k].pos = posOf(w(pw[k]))
    }

    private fun applySelection() {
        val en = enShown == 1
        val view = viewWords[preset]
        val pinSet = PIN_WORDS[preset].map { w(it) }.toSet()
        neighbours = if (sel.size == 1) ws.nearest(sel[0], 5) else emptyList()
        cands = if (sel.size == 3) ws.analogy(sel[0], sel[1], sel[2], 3) else emptyList()
        val hot = HashSet<Int>(sel); neighbours.forEach { hot += it.first }; cands.forEach { hot += it.first }
        val defaultTiles = if (preset < 3) view else pinSet + listOf("cat", "tiger", "apple", "lemon", "red", "blue", "big", "bigger", "Beijing", "London").map { w(it) }
        for (i in shown) {
            val c = ws.category[i]
            val inView = i in view
            val d = dots[i]!!
            d.setMat(if (i in hot) dotHot[c] else if (inView) dotBright[c] else dotDim[c])
            d.uniformScale(if (i in sel) 0.070f else if (i in hot) 0.058f else if (inView) 0.048f else 0.028f)
            val t = tiles[i]!!
            val vis = showAll || i in hot || i in defaultTiles
            t.visible = vis
            if (vis) {
                val h = if (i in hot) 0.040f else if (inView) 0.032f else 0.022f
                t.scale = P3(h * aspect(i, en), h, 1f)
                val up = if (i in sel || (cands.isNotEmpty() && i == cands[0].first)) 0.036f + h / 2 else 0.012f + h / 2
                tileUp[i] = up; val p = posOf(i); t.pos = P3(p.x, p.y + up, p.z)
                t.setMat(if (i in sel) tileHot else if (cands.firstOrNull()?.first == i) tileGold else tileMats[c])
            }
        }
        // neighbour threads
        for (k in 0 until 5) {
            val n = neighbours.getOrNull(k)
            if (n == null) { threads[k].visible = false; continue }
            line(threads[k], posOf(sel[0]), posOf(n.first), 0.0016f + 0.0026f * n.second.toFloat().coerceIn(0f, 1f))
            threads[k].setMat(threadMats[(n.second * 10).toInt().coerceIn(2, 10)])
        }
        for (k in 0 until 3) {
            val r = selRings[k]; val s = sel.getOrNull(k)
            r.visible = s != null
            if (s != null) { r.pos = posOf(s); r.uniformScale(0.034f) }
        }
        // difference vector B -> A
        if (sel.size >= 2) {
            val a = posOf(sel[0]); val b = posOf(sel[1])
            val g = if (sel.size == 2) smooth(grow) else 1f
            arrow(arrAB, b, b + (a - b) * g)
        } else hide(arrAB)
        // analogy: C + (A − B) = T, parallelogram B, A, T, C
        if (sel.size == 3) {
            val a = posOf(sel[0]); val b = posOf(sel[1]); val c = posOf(sel[2])
            val t = a - b + c
            val g = smooth(grow)
            arrow(arrCT, c, c + (a - b) * g)
            val g2 = ((grow - 0.55f) / 0.45f).coerceIn(0f, 1f)
            dashed(dashBC, b, c, g2); dashed(dashAT, a, t, g2)
            fill.visible = g2 > 0f
            if (g2 > 0f) { setFill(b, a, t, c); fill.setMat(fillMats[(g2 * 5).toInt().coerceIn(1, 5)]) }
            targetRing.visible = grow >= 1f; targetRing.pos = t
            val top = cands.firstOrNull()?.first
            if (top != null && grow >= 1f) { line(resid, t, posOf(top), 0.0018f) } else resid.visible = false
            for (k in 0 until 3) {
                val cnd = cands.getOrNull(k); val r = rankRings[k]
                r.visible = cnd != null && grow >= 1f
                if (cnd != null) { r.pos = posOf(cnd.first); r.uniformScale(if (k == 0) 0.042f else 0.034f) }
            }
        } else {
            hide(arrCT); dashBC.forEach { it.visible = false }; dashAT.forEach { it.visible = false }
            fill.visible = false; targetRing.visible = false; resid.visible = false; rankRings.forEach { it.visible = false }
        }
    }

    private fun hide(a: Pair<Node, Node>) { a.first.visible = false; a.second.visible = false }

    private fun line(n: Node, p: P3, q: P3, r: Float) {
        val d = q - p; val len = d.length
        if (len < 1e-4f) { n.visible = false; return }
        n.visible = true
        n.pos = (p + q) * 0.5f
        n.rot = Q.fromTo(0f, 1f, 0f, d.x, d.y, d.z)
        n.scale = P3(r, len, r)
    }

    private fun arrow(a: Pair<Node, Node>, p: P3, q: P3) {
        val d = q - p; val len = d.length
        if (len < 0.01f) { hide(a); return }
        val head = min(0.032f, len * 0.45f)
        val u = d * (1f / len)
        line(a.first, p, q - u * head, 0.0042f)
        a.second.visible = true
        a.second.pos = q - u * head
        a.second.rot = Q.fromTo(0f, 1f, 0f, u.x, u.y, u.z)
        a.second.scale = P3(0.012f, head, 0.012f)
    }

    private fun dashed(ns: Array<Node>, p: P3, q: P3, g: Float) {
        val n = ns.size
        for (k in 0 until n) {
            val t0 = (k + 0.15f) / n; val t1 = (k + 0.75f) / n
            if (t1 > g || g <= 0f) { ns[k].visible = false; continue }
            line(ns[k], p + (q - p) * t0, p + (q - p) * t1, 0.0022f)
        }
    }

    private fun setFill(b: P3, a: P3, t: P3, c: P3) {
        val ps = fillMesh.pos
        val corners = listOf(b, a, t, c)
        for (s in 0..1) for ((k, v) in corners.withIndex()) { val o = (s * 4 + k) * 3; ps[o] = v.x; ps[o + 1] = v.y; ps[o + 2] = v.z }
        meshEdited(fillMesh)
    }

    /** Cloud-local point -> volume coordinates (for labels). */
    private fun world(p: P3): P3 {
        val c = cos(yaw); val s = sin(yaw)
        return P3(C0.x + c * p.x + s * p.z, C0.y + p.y, C0.z - s * p.x + c * p.z)
    }

    private val VIEW_NAME = listOf(Bi("性别", "gender"), Bi("首都", "capitals"), Bi("比较级", "comparatives"), Bi("全部", "all words"))

    private fun updateLabels() {
        labels.clear()
        labels += LabelSpec(tr("迷你词向量：%d 词 · %d 维 · 自动生成的语料".format(ws.v, ws.d), "Mini embeddings: %d words · %d-D · generated corpus".format(ws.v, ws.d)),
            P3(-0.50f, 0.43f, -0.30f))
        val fitted = viewWords[preset].size
        labels += LabelSpec(tr("视图：%s · PCA 拟合 %d 个词".format(VIEW_NAME[preset].zh, fitted), "View: %s · PCA fit to %d words".format(VIEW_NAME[preset].en, fitted)),
            P3(0.50f, 0.43f, -0.30f))
        if (sel.size == 3 && grow >= 1f) {
            val a = posOf(sel[0]); val b = posOf(sel[1]); val c = posOf(sel[2])
            labels += LabelSpec("A − B + C", world(P3(a.x - b.x + c.x, a.y - b.y + c.y - 0.045f, a.z - b.z + c.z)))
            for ((k, cnd) in cands.withIndex()) {
                val p = posOf(cnd.first)
                labels += LabelSpec("#%d %s  %.2f".format(k + 1, name(cnd.first), cnd.second), world(P3(p.x, p.y + 0.075f + 0.0f * k, p.z)))
            }
        }
    }

    private fun readout3(): String {
        val nW = ws.v; val nS = shown.size
        return when (sel.size) {
            0 -> listOf(
                tr("捏一个词看它的 5 个近邻；再捏第二个看差向量；第三个：A − B + C 落在哪？",
                    "Pinch a word: 5 nearest neighbours. A 2nd: the difference vector. A 3rd: where does A − B + C land?"),
                tr("视图：%s · 3-D 是 32 维向量的 PCA 投影；余弦和类比都在 32 维里算".format(VIEW_NAME[preset].zh),
                    "View: %s · 3-D is a PCA projection of the 32-D vectors; cosines and analogies use all 32 dims".format(VIEW_NAME[preset].en)),
                tr("迷你词向量：%d 词（显示 %d），%d 维，skip-gram 在自动生成的语料上训练".format(nW, nS, ws.d),
                    "Mini embeddings: %d words (%d shown), %d-D, skip-gram trained on a generated corpus".format(nW, nS, ws.d)),
            )
            1 -> listOf(
                tr("选中：%s —— 线连向余弦最大的 5 个词".format(name(sel[0])), "Selected: %s — threads go to its 5 most similar words".format(name(sel[0]))),
                tr("近邻（cos，32 维）：", "Neighbours (cos, 32-D): ") + neighbours.joinToString(" · ") { "%s %.2f".format(name(it.first), it.second) },
                tr("它只学过「预测旁边的词」，可用法相近的词自己挤到了一起", "It only learned to predict nearby words, yet words used alike ended up close together"),
            )
            2 -> {
                val (a, b) = sel[0] to sel[1]
                val rel = ws.relationOf(b, a)
                listOf(
                    tr("差向量：%s − %s（金色箭头从 %s 指向 %s）".format(name(a), name(b), name(b), name(a)),
                        "Difference vector: %s − %s (gold arrow from %s to %s)".format(name(a), name(b), name(b), name(a))),
                    tr("cos(%s, %s) = %.2f · 关系表：%s".format(name(a), name(b), ws.cos(a, b), relZh(rel?.first ?: 3)),
                        "cos(%s, %s) = %.2f · relation table: %s".format(name(a), name(b), ws.cos(a, b), relEn(rel?.first ?: 3))),
                    tr("再捏第三个词 C：把这支箭头搬到 C 上，会落在哪个词旁边？", "Pinch a third word C: move this arrow to C — which word does it land next to?"),
                )
            }
            else -> {
                val (a, b, c) = Triple(sel[0], sel[1], sel[2])
                val exp = ws.expected(a, b, c)
                val top = cands.firstOrNull()?.first ?: -1
                val ok = exp >= 0 && top == exp
                val l1 = "%s − %s + %s ≈ %s".format(name(a), name(b), name(c), name(top)) +
                    (if (exp < 0) tr("（关系表里没有这组）", " (not a pair in the relation table)") else if (ok) " ✓" else tr("（应为 %s）".format(name(exp)), " (expected %s)".format(name(exp))))
                val l2 = tr("候选（已排除 A、B、C）：", "Candidates (A, B, C excluded): ") + cands.joinToString(" · ") { "%s %.2f".format(name(it.first), it.second) } +
                    (if (exp >= 0) tr(" · %s 排第 %d".format(name(exp), ws.rankOf(a, b, c, exp)), " · %s ranks #%d".format(name(exp), ws.rankOf(a, b, c, exp))) else "")
                val l3 = if (exp >= 0) tr("两条金边平行：同一个方向在两组词里表示同一种关系 —— 类比就是向量加减",
                    "The gold edges are parallel: one direction means one relation in both pairs — analogy is vector arithmetic")
                else tr("A、B、C 不在同一组关系里：平行四边形的第四个角没有对应的词", "A, B, C share no relation: the parallelogram's 4th corner has no matching word")
                listOf(l1, l2, l3)
            }
        }.joinToString("\n")
    }
    private fun relZh(r: Int) = listOf("性别", "国家→首都", "原级→比较级", "无")[r]   // i18n-ok (used only inside tr())
    private fun relEn(r: Int) = listOf("gender", "country→capital", "base→comparative", "none")[r]

    // ---------------------------------------------------------------- interaction

    override val tapTargets: List<TapTarget> get() {
        val l = ArrayList<TapTarget>()
        for ((k, e) in PIN_WORDS[preset].withIndex()) { val i = w(e); l += TapTarget("Pin$k", ws.zh[i], e) { i in sel } }
        l += TapTarget("SpinKnob", "旋钮 → 转 45°", "Knob → turn 45°")
        return l
    }

    override fun isTarget(name: String) = name == "SpinKnob" || name.startsWith("Pin")
    override fun isDragTarget(name: String) = name == "SpinKnob"
    override fun onDrag(name: String, dx: Float, dy: Float, dz: Float) { if (name == "SpinKnob") { yaw += dx * 5f; yawTravel += abs(dx * 5f) } }
    override fun tapMessage(name: String): String? = lastMsg.text.ifEmpty { null }

    override fun onTap(name: String) {
        when {
            name == "SpinKnob" -> { rotateBy(PI.toFloat() / 4f); lastMsg = Bi("转 45°：从侧面看平行四边形还平吗？", "Turned 45°: is the parallelogram still flat from the side?") }
            name.startsWith("Pin") -> {
                val k = name.removePrefix("Pin").toIntOrNull() ?: return
                val i = w(PIN_WORDS[preset].getOrNull(k) ?: return)
                pick(i)
                lastMsg = when {
                    i !in sel -> Bi("取消 %s".format(ws.zh[i]), "Deselected %s".format(ws.en[i]))
                    sel.size == 1 -> Bi("A = %s：5 个近邻亮起".format(ws.zh[i]), "A = %s: 5 nearest neighbours light up".format(ws.en[i]))
                    sel.size == 2 -> Bi("B = %s：金箭头 = A − B".format(ws.zh[i]), "B = %s: gold arrow = A − B".format(ws.en[i]))
                    else -> Bi("C = %s：A − B + C 落在哪？".format(ws.zh[i]), "C = %s: where does A − B + C land?".format(ws.en[i]))
                }
            }
        }
    }

    /** Volume anchor for tap notes: above the object, clamped under the ceiling. */
    override fun anchorOf(name: String): FloatArray? {
        val a = super.anchorOf(name) ?: return null
        return floatArrayOf(a[0], min(a[1], 0.44f), a[2])
    }

    // ---------------------------------------------------------------- panel

    @Composable
    override fun Panel() {
        var p by FieldRef(::preset)
        var all by FieldRef(::showAll)
        var si by FieldRef(::spinIdx)
        LabPanel(readout) {
            SegmentControl {
                VIEW_NAME.forEachIndexed { k, s -> SegmentItem(selected = p == k, onClick = { setPreset(k) }, title = { Text(s.text) }) }
            }
            LabSwitch(tr("全部词名", "All names"), all) { all = it; dirty = true }
            SegmentControl {
                listOf(tr("停", "Stop"), tr("慢", "Slow"), tr("中", "Mid"), tr("快", "Fast")).forEachIndexed { k, s ->
                    SegmentItem(selected = si == k, onClick = { si = k }, title = { Text(s) })
                }
            }
            LabRow {
                Button(onClick = { exampleAnalogy() }) { Text(tr("示例类比", "Example")) }
                Button(onClick = { clearSelection() }) { Text(tr("清除", "Clear")) }
                Button(onClick = { resetAll() }) { Text(tr("重置", "Reset")) }
            }
        }
    }

    /** The view's canonical analogy (panel button). */
    fun exampleAnalogy() {
        when (preset) {
            1 -> runAnalogy(w("Tokyo"), w("Japan"), w("France"))
            2 -> runAnalogy(w("bigger"), w("big"), w("small"))
            else -> runAnalogy(w("king"), w("man"), w("woman"))
        }
    }

    fun resetAll() { setPreset(0); clearSelection(); showAll = false; spinIdx = 0; stepYaw = 0f; yaw = -0.35f }

    // ---------------------------------------------------------------- guided track (导学)

    private var travelAtStep = 0f
    private val lessonExcluded by lazy { setOf(listOf(w("king"), w("man"), w("woman")), listOf(w("Tokyo"), w("Japan"), w("France"))) }
    private fun challengeOk(): Boolean {
        if (!analogyShown) return false
        if (sel.toList() in lessonExcluded) return false
        val e = ws.expected(sel[0], sel[1], sel[2])
        return e >= 0 && cands.firstOrNull()?.first == e
    }

    // ---- background card (📖): context for a viewer who has never met this topic
    override val about = About(listOf(
        About.Page(
            Bi("这个实验在做什么？", "What is this experiment?"),
            Bi("转台上方漂浮着 96 个发光的词，颜色 = 类别（男性、女性、国家、首都、形容词、动物……）。每个词的位置都是模型学出来的。捏一个词，看和它最像的 5 个词；再捏第二个，金色箭头 = 两个词的差；捏第三个，同一支箭头搬过去，看落在哪个词旁边。台面上的旋钮转动整团云，面板可切换「性别」「首都」「比较级」视图。", "Above a turntable float 96 glowing words, coloured by category (male, female, country, capital, adjective, animal…). Their positions were learned. Pinch a word to see its 5 most similar words; pinch a second and a gold arrow shows their difference; pinch a third and the same arrow is moved there — which word does it land next to? The knob on the plinth spins the cloud; the panel switches gender, capitals and comparatives views."),
        ),
        About.Page(
            Bi("先认识几个词", "A few words first"),
            Bi("计算机不认识「国王」这两个字，它只会算数。「词向量」就是给每个词分配一串数（这里是 32 个），相当于它在 32 维空间里的坐标。两个向量方向越接近，「余弦相似度」越接近 1，说明两个词的用法越像。「训练」就是不断调整这些坐标，让模型更会预测一个词旁边常出现哪些词。", "A computer doesn't understand the word “king”; it only does arithmetic. A “word vector” gives every word a list of numbers (32 here), like coordinates in a 32-dimensional space. The closer two vectors point the same way, the closer their “cosine similarity” is to 1, meaning the words are used alike. “Training” keeps adjusting these coordinates so the model gets better at predicting which words appear near a word."),
        ),
        About.Page(
            Bi("背景：看邻居识词", "Background: know a word by its neighbours"),
            Bi("1957 年，语言学家弗斯提出：看一个词常和哪些词在一起，就能知道它的意思。2013 年，谷歌的米科洛夫等人发布 word2vec：模型只学「预测旁边的词」，却发现 国王 − 男人 + 女人 最接近 王后 —— 空间里的方向本身带上了意义。从此，「把词变成向量」成了几乎所有语言模型的第一步。", "In 1957 the linguist J. R. Firth argued that you can tell what a word means from the words it keeps company with. In 2013 Tomas Mikolov and colleagues at Google released word2vec: a model trained only to predict neighbouring words placed king − man + woman closest to queen — directions in the space themselves carried meaning. Turning words into vectors became the first step of nearly every language model since."),
        ),
        About.Page(
            Bi("核心想法：方向就是关系", "Key idea: directions are relations"),
            Bi("想象一张地图：从「男人」走到「女人」，要朝某个方向走一段；从「国王」到「王后」，几乎是同一个方向、同样远。所以把这支箭头搬到「国王」上，就落在「王后」附近 —— 类比变成了向量加减：A − B + C ≈ D，四个词围成一个平行四边形。没人教过它「性别」，这个方向是从用法里自己长出来的。", "Think of a map: walking from “man” to “woman” means going some distance in one direction; from “king” to “queen” is almost the same direction and distance. So move that arrow onto “king” and it lands near “queen” — an analogy becomes vector arithmetic: A − B + C ≈ D, four words forming a parallelogram. Nobody taught it “gender”; the direction grew out of how the words are used."),
        ),
        About.Page(
            Bi("怎么看这个 3D 场景", "How to read the 3-D scene"),
            Bi("每个光点 = 一个词，颜色 = 类别，上方的小字牌是词名。光点的位置 = 32 维向量用 PCA 投影到 3-D（只留最主要的 3 个方向），所以看到的距离只是近似，近邻和类比都在 32 维里算。细线 = 5 个最近邻；金色箭头 = 差向量 A − B；平行四边形的第四个角 = A − B + C，带圈的是前 3 名候选。", "Each glowing dot = a word, colour = category, with its name on a small tile above. A dot's position = its 32-number vector projected to 3-D by PCA (keeping the 3 main directions), so the distances you see are approximate; neighbours and analogies are computed in 32-D. Threads = the 5 nearest neighbours; the gold arrow = the difference A − B; the parallelogram's fourth corner = A − B + C, with the top 3 candidates ringed."),
        ),
        About.Page(
            Bi("有多大？今天在哪里？", "How big? Where is it today?"),
            Bi("这个模型只有 199 个词、32 维，用模板自动造出的句子训练。谷歌 2013 年公开的 word2vec 向量有约 300 万个词和短语、300 维，学自约 1000 亿词的新闻。今天的搜索引擎、推荐系统、AI 助手查资料（「向量检索」），都先把文字、图片变成向量，再找方向最接近的那些。", "This model has just 199 words and 32 dimensions, trained on sentences built from templates. The word2vec vectors Google released in 2013 cover about 3 million words and phrases in 300 dimensions, learned from about 100 billion words of news. Today search engines, recommenders and AI assistants looking things up (“vector search”) first turn text and images into vectors, then find the ones pointing most alike."),
        ),
    ))

    override val lesson: Lesson by lazy {
        val k = ws.analogy(w("king"), w("man"), w("woman"))
        val t = ws.analogy(w("Tokyo"), w("Japan"), w("France"))
        Lesson(
            listOf(
                LessonStep(
                    say = Bi("模型只学过一件事：预测一个词旁边会出现哪些词。猜一猜：国王 − 男人 + 女人 ≈ ？", "The model learned just one thing: predict the words around a word. Guess: king − man + woman ≈ ?"),
                    options = listOf(Bi("王后", "queen"), Bi("公主", "princess"), Bi("女人", "woman"), Bi("王子", "prince")),
                    answer = 0,
                ),
                LessonStep(
                    say = Bi("在「性别」视图里依次捏 国王、男人、女人（或点下面的按钮）：A − B + C。", "In the gender view pinch king, then man, then woman (or tap the buttons below): A − B + C."),
                    target = "Pin0",
                    onEnter = { if (preset != 0) setPreset(0) },
                    done = { isAnalogy("king", "man", "woman") && grow >= 1f },
                    autoplay = { setPreset(0); clearSelection(); onTap("Pin0"); onTap("Pin2"); onTap("Pin3") },
                ),
                LessonStep(
                    say = Bi("把云转起来（捏台面上的旋钮，或选「中」速）：两条金色的边从侧面看也平行吗？", "Spin the cloud (pinch the knob on the plinth, or pick Mid speed): are the two gold edges parallel from the side too?"),
                    target = "SpinKnob",
                    onEnter = { travelAtStep = yawTravel },
                    done = { yawTravel - travelAtStep > 1.0f },
                    autoplay = { spinIdx = 3 },
                    reveal = true,
                    insight = Bi("第 1 名是 %s，cos = %.2f（第 2 名 %s 只有 %.2f）。男人→女人 和 国王→王后 几乎是同一支箭头：「性别」成了空间里的一个方向。".format(ws.zh[k[0].first], k[0].second, ws.zh[k[1].first], k[1].second),
                        "#1 is %s, cos = %.2f (#2 %s only %.2f). man→woman and king→queen are almost the same arrow: gender became a DIRECTION in the space.".format(ws.en[k[0].first], k[0].second, ws.en[k[1].first], k[1].second)),
                ),
                LessonStep(
                    say = Bi("再猜一个：东京 − 日本 + 法国 ≈ ？", "One more guess: Tokyo − Japan + France ≈ ?"),
                    options = listOf(Bi("伦敦", "London"), Bi("巴黎", "Paris"), Bi("罗马", "Rome"), Bi("日本", "Japan")),
                    answer = 1,
                ),
                LessonStep(
                    say = Bi("切到「首都」视图，先捏 东京，再捏 日本：金色箭头就是差向量 东京 − 日本。", "Switch to the capitals view, pinch Tokyo, then Japan: the gold arrow is the difference vector Tokyo − Japan."),
                    target = "Pin3",
                    onEnter = { spinIdx = 0 },
                    done = { preset == 1 && sel.size == 2 && grow >= 1f },
                    autoplay = { setPreset(1); clearSelection(); onTap("Pin3"); onTap("Pin2") },
                    reveal = true,
                    insight = Bi("这支箭头是「国家→首都」。把它搬到 法国 上：%s 排第 1（cos %.2f）—— 再捏 法国 就能看到它落在 巴黎 旁边。".format(ws.zh[t[0].first], t[0].second),
                        "This arrow means country→capital. Move it onto France: %s ranks #1 (cos %.2f) — pinch France next and watch it land by Paris.".format(ws.en[t[0].first], t[0].second)),
                ),
            ),
            challenge = Challenge(
                say = Bi("找出另一组平行四边形：用 A − B + C 让第一名候选正确（不算上面两组）。", "Find another parallelogram: make A − B + C's #1 candidate correct (not the two above)."),
                progress = {
                    when {
                        sel.size < 3 -> tr("已选 %d/3 个词".format(sel.size), "%d/3 words picked".format(sel.size))
                        sel.toList() in lessonExcluded -> tr("这组导学里做过了，换一组", "Already done in the lesson — try another")
                        ws.expected(sel[0], sel[1], sel[2]) < 0 -> tr("这三个词不在同一组关系里", "These three words share no relation")
                        challengeOk() -> tr("第 1 名 %s 正确 ✓".format(name(cands[0].first)), "#1 %s is right ✓".format(name(cands[0].first)))
                        else -> tr("第 1 名不对，再试一组", "#1 is wrong — try another set")
                    }
                },
                done = { challengeOk() },
                autoplay = { setPreset(0); clearSelection(); onTap("Pin4"); onTap("Pin6"); onTap("Pin7") },
            ),
        )
    }

    // ---------------------------------------------------------------- review states

    override fun reviewVariants(): Map<String, () -> Unit> = mapOf(
        "king" to { setPreset(0); runAnalogy(w("king"), w("man"), w("woman")); grow = 1f },
        "capital" to { setPreset(1); runAnalogy(w("Tokyo"), w("Japan"), w("France")); grow = 1f },
        "neighbours" to { setPreset(3); clearSelection(); pick(w("cat")) },
        "comparative" to { setPreset(2); runAnalogy(w("bigger"), w("big"), w("small")); grow = 1f },
    )

    // ---------------------------------------------------------------- meshes

    private fun quad(u0: Float, v0: Float, u1: Float, v1: Float) = MeshData(
        floatArrayOf(-0.5f, -0.5f, 0f, 0.5f, -0.5f, 0f, 0.5f, 0.5f, 0f, -0.5f, 0.5f, 0f),
        floatArrayOf(0f, 0f, 1f, 0f, 0f, 1f, 0f, 0f, 1f, 0f, 0f, 1f),
        floatArrayOf(u0, v1, u1, v1, u1, v0, u0, v0),
        intArrayOf(0, 1, 2, 0, 2, 3))

    /** Flat ring in the XY plane (faces +Z; used as billboard). */
    private fun ringMesh(rIn: Float, rOut: Float, n: Int = 40): MeshData {
        val pos = FloatArray((n + 1) * 6); val nrm = FloatArray((n + 1) * 6); val uv = FloatArray((n + 1) * 4)
        for (k in 0..n) for (j in 0..1) {
            val a = 2.0 * PI * k / n; val r = if (j == 0) rIn else rOut; val v = k * 2 + j
            pos[v * 3] = (r * cos(a)).toFloat(); pos[v * 3 + 1] = (r * sin(a)).toFloat(); nrm[v * 3 + 2] = 1f
            uv[v * 2] = j.toFloat(); uv[v * 2 + 1] = k.toFloat() / n
        }
        val idx = IntArray(n * 6); var t = 0
        for (k in 0 until n) { val a0 = k * 2; idx[t++] = a0; idx[t++] = a0 + 1; idx[t++] = a0 + 2; idx[t++] = a0 + 1; idx[t++] = a0 + 3; idx[t++] = a0 + 2 }
        return MeshData(pos, nrm, uv, idx)
    }

    /** Cone along +Y: base radius 1 at y = 0, apex at y = 1. */
    private fun coneMesh(n: Int): MeshData {
        val pos = ArrayList<Float>(); val nrm = ArrayList<Float>(); val uv = ArrayList<Float>(); val idx = ArrayList<Int>()
        fun v(x: Float, y: Float, z: Float, nx: Float, ny: Float, nz: Float): Int { pos += x; pos += y; pos += z; nrm += nx; nrm += ny; nrm += nz; uv += 0f; uv += 0f; return pos.size / 3 - 1 }
        val k = 1f / sqrt(2f)
        for (s in 0 until n) {
            val a0 = 2 * PI * s / n; val a1 = 2 * PI * (s + 1) / n; val am = (a0 + a1) / 2
            val p0 = v(cos(a0).toFloat(), 0f, sin(a0).toFloat(), cos(a0).toFloat() * k, k, sin(a0).toFloat() * k)
            val p1 = v(cos(a1).toFloat(), 0f, sin(a1).toFloat(), cos(a1).toFloat() * k, k, sin(a1).toFloat() * k)
            val ap = v(0f, 1f, 0f, cos(am).toFloat() * k, k, sin(am).toFloat() * k)
            idx += p0; idx += ap; idx += p1
            val c = v(0f, 0f, 0f, 0f, -1f, 0f); val q0 = v(cos(a0).toFloat(), 0f, sin(a0).toFloat(), 0f, -1f, 0f); val q1 = v(cos(a1).toFloat(), 0f, sin(a1).toFloat(), 0f, -1f, 0f)
            idx += c; idx += q0; idx += q1
        }
        return MeshData(pos.toFloatArray(), nrm.toFloatArray(), uv.toFloatArray(), idx.toIntArray())
    }
}
