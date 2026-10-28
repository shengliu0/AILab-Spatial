package com.bytedance.ailab.render

/**
 * The Unity SolarSystemDemo.BuildPlanets colour stops, verbatim (sRGB). deep / mid / high are the
 * three OKLab stops BakePlanet interpolates through; band 1 = Jovian belts, 0 = rocky blotches.
 */
object Palette {
    val deep = listOf(
        Rgb(0.26f, 0.24f, 0.22f), Rgb(0.70f, 0.58f, 0.33f), Rgb(0.04f, 0.15f, 0.40f),
        Rgb(0.33f, 0.15f, 0.08f), Rgb(0.40f, 0.28f, 0.19f), Rgb(0.55f, 0.45f, 0.28f),
        Rgb(0.36f, 0.65f, 0.70f), Rgb(0.08f, 0.20f, 0.54f),
    )
    val mid = listOf(
        Rgb(0.44f, 0.41f, 0.37f), Rgb(0.88f, 0.78f, 0.54f), Rgb(0.13f, 0.42f, 0.20f),
        Rgb(0.62f, 0.32f, 0.16f), Rgb(0.80f, 0.68f, 0.50f), Rgb(0.86f, 0.76f, 0.52f),
        Rgb(0.58f, 0.83f, 0.85f), Rgb(0.19f, 0.36f, 0.76f),
    )
    val high = listOf(
        Rgb(0.63f, 0.59f, 0.53f), Rgb(0.98f, 0.94f, 0.78f), Rgb(0.92f, 0.94f, 0.97f),
        Rgb(0.82f, 0.55f, 0.34f), Rgb(0.96f, 0.91f, 0.81f), Rgb(0.97f, 0.92f, 0.76f),
        Rgb(0.79f, 0.94f, 0.94f), Rgb(0.46f, 0.62f, 0.93f),
    )
    val band = floatArrayOf(0.0f, 0.30f, 0.15f, 0.10f, 1.0f, 1.0f, 0.50f, 0.50f)
    val caps = booleanArrayOf(false, false, true, true, false, false, false, false)

    // Moon: grey/tan maria field (Unity BuildMoon)
    val moonDeep = Rgb(0.13f, 0.12f, 0.11f)
    val moonMid = Rgb(0.33f, 0.31f, 0.29f)
    val moonHigh = Rgb(0.58f, 0.56f, 0.51f)

    /** Unity OrbitColour: planet tint pushed 25% toward white (OKLab). */
    fun orbit(i: Int): Rgb = ProcTex.okLerp(mid[i], Rgb(1f, 1f, 1f), 0.25f)
}
