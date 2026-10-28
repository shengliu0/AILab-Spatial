// LabAssets — read a file shipped in app/src/main/assets/ (trained weights, tables) from pure demo code.
// On the headset LabVolume points [open] at the APK's AssetManager; in JVM tests (working dir = app/) the
// default reads the source tree, so cores can be tested against the very same bytes that ship.

package com.bytedance.ailab.lab

import java.io.File

object LabAssets {
    @Volatile var open: (String) -> ByteArray? = { path -> File("src/main/assets/$path").takeIf { it.isFile }?.readBytes() }
    fun bytes(path: String): ByteArray = open(path) ?: error("asset not found: $path")
}
