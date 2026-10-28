// CnnLab — the demo's pure-Kotlin brain: which trained model is live (task × kernel bank × sensor), which
// sample is shown, cached forward passes, and two measurements the guided track uses. No Android.

package com.bytedance.ailab.core.cnn

/**
 * Six models (tools/train_cnn.py), all the same skeleton C×28×28 → 8 → 16 → N:
 *   digits_{learned,hand}   RGB digits, 10 classes (softmax)
 *   scene4_{learned,hand}   R,G,B,NIR tiles, 5 presence labels (sigmoid)
 *   scene3_{learned,hand}   the same tiles through an RGB-only sensor
 * "hand" = conv1 frozen to 8 hand-made kernels during training, everything after it learned.
 */
class CnnLab(private val load: (String) -> ByteArray) {
    val models: Map<String, CnnModel> = NAMES.associateWith { CnnModel.parse(it, load("cnn/$it.bin")) }
    val digits: CnnSet = CnnSet.parse(load("cnn/digits_eval.bin"))
    val scenes: CnnSet = CnnSet.parse(load("cnn/scenes_eval.bin"))
    val metrics: Map<String, String> = Metrics.parse(String(load("cnn/metrics.txt"), Charsets.UTF_8))

    fun modelName(scene: Boolean, learned: Boolean, nir: Boolean): String =
        (if (!scene) "digits" else if (nir) "scene4" else "scene3") + if (learned) "_learned" else "_hand"

    fun model(scene: Boolean, learned: Boolean, nir: Boolean): CnnModel = models.getValue(modelName(scene, learned, nir))

    /** The input tensor the model sees: a digit (3 ch) or a scene (4 ch, or 3 without the NIR band). */
    fun input(scene: Boolean, nir: Boolean, idx: Int): Tensor =
        if (!scene) digits.sample(idx) else scenes.sample(idx, if (nir) 4 else 3)

    private val cache = HashMap<String, CnnModel.Acts>()
    /** Forward pass, cached per (model, sample): the demo only re-runs inference when the user changes something. */
    fun acts(scene: Boolean, learned: Boolean, nir: Boolean, idx: Int): CnnModel.Acts =
        cache.getOrPut("${modelName(scene, learned, nir)}|$idx") { model(scene, learned, nir).forward(input(scene, nir, idx)) }

    /** Held-out accuracy printed by the training script (fonts / scenes never seen in training). */
    fun heldOut(name: String): Float = metrics["$name.heldout_acc"]?.toFloatOrNull() ?: Float.NaN
    fun labelAcc(name: String, label: String): Float = metrics["$name.label_acc_$label"]?.toFloatOrNull() ?: Float.NaN

    /** Shipped digits shown by the demo: the first 20 (2 per class, 0..9, put first by the training script). */
    val digitShow: IntArray get() = IntArray(minOf(20, digits.count)) { it }
    /** Shipped scenes shown: the 4 composed showcase tiles, then 4 random held-out tiles. */
    val sceneShow: IntArray get() = IntArray(minOf(8, scenes.count)) { it }

    companion object {
        val NAMES = listOf("digits_learned", "digits_hand", "scene4_learned", "scene4_hand", "scene3_learned", "scene3_hand")
    }
}

/**
 * "Which first-layer kernel likes the slant of a 7?" For a bright-on-dark digit: stroke pixels = luminance above
 * the midpoint of the image's range. The top bar = stroke rows within 3 rows of the topmost stroke row, the slant
 * = stroke rows 6 or more below it. Selectivity of conv1 map k = mean ReLU response over the (1-px dilated) slant
 * minus the same over the bar — positive: the kernel lights the diagonal more than the horizontal.
 * conv1 cell (i, j) is centred on input pixel (i + 1, j + 1) (valid 3×3 convolution).
 */
object SlantProbe {
    fun masks(x: Tensor): Pair<BooleanArray, BooleanArray> {
        val n = x.h * x.w
        val lum = FloatArray(n) { p -> var s = 0f; for (c in 0 until minOf(3, x.c)) s += x.d[c * n + p]; s / minOf(3, x.c) }
        val lo = lum.min(); val hi = lum.max(); val mid = (lo + hi) / 2
        val stroke = BooleanArray(n) { lum[it] > mid }
        val top = (0 until x.h).firstOrNull { y -> (0 until x.w).any { stroke[y * x.w + it] } } ?: 0
        fun dilate(rows: IntRange): BooleanArray {
            val m = BooleanArray(n)
            for (y in rows) for (xx in 0 until x.w) if (stroke[y * x.w + xx])
                for (dy in -1..1) for (dx in -1..1) { val yy = y + dy; val xq = xx + dx; if (yy in 0 until x.h && xq in 0 until x.w) m[yy * x.w + xq] = true }
            return m
        }
        return dilate(top..minOf(x.h - 1, top + 3)) to dilate(minOf(x.h - 1, top + 6) until x.h)
    }

    /** Selectivity (slant − bar) of every conv1 map of [a]. */
    fun selectivity(a: CnnModel.Acts): FloatArray {
        val (bar, slant) = masks(a.input)
        val w = a.input.w
        fun mean(k: Int, m: BooleanArray): Float {
            var s = 0f; var c = 0
            for (p in m.indices) if (m[p]) {
                val i = p / w - 1; val j = p % w - 1
                if (i in 0 until a.conv1.h && j in 0 until a.conv1.w) { s += maxOf(0f, a.conv1[k, i, j]); c++ }
            }
            return if (c == 0) 0f else s / c
        }
        return FloatArray(a.conv1.c) { k -> mean(k, slant) - mean(k, bar) }
    }
}

/** Display helpers that must be reproducible (and are tested): robust per-layer colour scale. */
object CnnScale {
    /** The [q]-quantile of |values| (falls back to max |v|, then 1): a few hot pixels must not darken a whole layer. */
    fun robust(d: FloatArray, q: Float = 0.98f): Float {
        if (d.isEmpty()) return 1f
        val a = FloatArray(d.size) { kotlin.math.abs(d[it]) }; a.sort()
        val v = a[((a.size - 1) * q).toInt()]
        return if (v > 1e-6f) v else if (a.last() > 1e-6f) a.last() else 1f
    }
}
