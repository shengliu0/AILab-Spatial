package com.bytedance.ailab.lab

/** Subjects = the catalogue's tabs. */
enum class Subject(private val zh: String, private val en: String) {
    BASICS("神经网络基础", "Basics"), VISION("视觉", "Vision"), LANGUAGE("语言", "Language"), GENERATIVE("生成模型", "Generative");  // i18n-ok (zh/en pairs)
    val label: String get() = tr(zh, en)
}

/**
 * The catalogue. Order inside a subject = order on the cards. Adding a demo = one line here, plus a
 * thumbnail (render it with LabReviewTest, then `python3 tools/make_thumbs.py`).
 * Every AI demo makes the data and the network visible as 3-D objects — tensors as blocks, layers as walls.
 */
object DemoRegistry {
    val catalog: List<Pair<Demo, Subject>> = listOf(
        com.bytedance.ailab.demos.mlp.MlpDemo to Subject.BASICS,           // 神经网络 · 手写数字  Rumelhart 1986
        com.bytedance.ailab.demos.descent.DescentDemo to Subject.BASICS,   // 梯度下降 · 损失曲面  Cauchy 1847
        com.bytedance.ailab.demos.backprop.BackpropDemo to Subject.BASICS,
        com.bytedance.ailab.demos.kernel.KernelDemo to Subject.BASICS,
        com.bytedance.ailab.demos.cnn.CnnDemo to Subject.VISION,           // 卷积神经网络  LeCun 1989
        com.bytedance.ailab.demos.resnet.ResnetDemo to Subject.VISION,
        com.bytedance.ailab.demos.gpt.GptDemo to Subject.LANGUAGE,         // Transformer · GPT  Vaswani et al. 2017
        com.bytedance.ailab.demos.embedding.EmbeddingDemo to Subject.LANGUAGE,
        com.bytedance.ailab.demos.diffusion.DiffusionDemo to Subject.GENERATIVE, // 扩散模型  Ho, Jain & Abbeel 2020
    )
    val all: List<Demo> = catalog.map { it.first }

    /** The demo the app opens on. */
    const val FIRST = "mlp"

    /** English titles for the catalogue / top bar (each demo's own `title` is the Chinese one). */
    private val TITLE_EN = mapOf<String, String>(
        "resnet" to "ResNet · Skip Connections",
        "embedding" to "Word Embeddings",
        "kernel" to "Kernel Trick · Lifting to 3-D",
        "backprop" to "Backpropagation · Chain Rule",
        "mlp" to "Neural Network · Digits",
        "descent" to "Gradient Descent · Loss Landscape",
        "cnn" to "Convolutional Neural Network",
        "gpt" to "Transformer · GPT",
        "diffusion" to "Diffusion Model",
    )
    fun titleOf(d: Demo): String = tr(d.title, TITLE_EN[d.id] ?: d.title)
    /** Catalogue-card title: one line in a 225 dp card, so English drops its " · subtitle" (Chinese too when > 12 chars). */
    fun cardTitleOf(d: Demo): String = tr(if (d.title.length > 12) d.title.substringBefore(" · ") else d.title,
        (CARD_EN[d.id] ?: TITLE_EN[d.id] ?: d.title).substringBefore(" · "))
    private val CARD_EN = mapOf<String, String>(
        "resnet" to "ResNet",
        "embedding" to "Word Embeddings",
        "kernel" to "Kernel Trick",
        "backprop" to "Backpropagation",
        "mlp" to "Neural Network",
        "descent" to "Gradient Descent",
        "cnn" to "CNN",
        "gpt" to "GPT",
        "diffusion" to "Diffusion",
    )
    fun subjectOf(id: String) = catalog.firstOrNull { it.first.id == id }?.second
}
