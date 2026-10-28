// Small float 3-vector used by the scene graph and the demos (volume metres).

package com.bytedance.ailab.core

import kotlin.math.sqrt

data class P3(val x: Float, val y: Float, val z: Float) {
    operator fun plus(o: P3) = P3(x + o.x, y + o.y, z + o.z)
    operator fun minus(o: P3) = P3(x - o.x, y - o.y, z - o.z)
    operator fun times(s: Float) = P3(x * s, y * s, z * s)
    val length get() = sqrt(x * x + y * y + z * z)
}
