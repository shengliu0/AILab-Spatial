package com.bytedance.ailab.scene

import com.bytedance.ailab.lab.Q

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sqrt

/**
 * The lighting shells are only right if their +Y pole lands on the sun (planets) or the viewer
 * (limb darkening). That is pure quaternion maths, so it is checked here rather than discovered on
 * the headset as "the night side is lit".
 */
class OrientationTest {

    private fun norm(v: FloatArray): FloatArray { val l = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); return floatArrayOf(v[0] / l, v[1] / l, v[2] / l) }
    private fun dot(a: FloatArray, b: FloatArray) = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @Test
    fun fromToMapsYOntoAnyDirection() {
        val dirs = listOf(
            floatArrayOf(1f, 0f, 0f), floatArrayOf(0f, 0f, -1f), floatArrayOf(0f, -1f, 0f),
            floatArrayOf(0f, 1f, 0f), floatArrayOf(-0.3f, 0.2f, 0.9f), floatArrayOf(0.001f, -1f, 0.0005f),
        )
        for (d in dirs) {
            val q = Q.fromTo(0f, 1f, 0f, d[0], d[1], d[2])
            val y = q.rotate(floatArrayOf(0f, 1f, 0f))
            assertEquals("fromTo ${d.toList()}", 1f, dot(norm(y), norm(d)), 1e-4f)
        }
    }

    @Test
    fun limbShellPoleFacesViewer() {
        // +90 deg about X must take the shell's +Y pole to the pivot's +Z (the LookAt forward)
        val y = Q.axisAngle(1f, 0f, 0f, (PI / 2).toFloat()).rotate(floatArrayOf(0f, 1f, 0f))
        assertEquals(0f, y[0], 1e-6f); assertEquals(0f, y[1], 1e-6f); assertEquals(1f, y[2], 1e-6f)
    }

    @Test
    fun planetShellPoleLandsOnSunThroughTiltAndOblateness() {
        // Saturn-like: 26.73 deg tilt, 9.8% flattening, sun direction arbitrary
        val tilt = Q.axisAngle(0f, 0f, 1f, Math.toRadians(26.73).toFloat())
        val f = 0.098f
        val sun = norm(floatArrayOf(-0.7f, 0.05f, 0.71f))
        // what SolarScene.orientShell does
        val d = tilt.conj().rotate(sun); d[1] /= (1f - f)
        val shellQ = Q.fromTo(0f, 1f, 0f, d[0], d[1], d[2])
        // forward chain: shell +Y -> squash scale (1,1-f,1) -> tilt rotation -> root frame
        val p = shellQ.rotate(floatArrayOf(0f, 1f, 0f)); p[1] *= (1f - f)
        val world = norm(tilt.rotate(p))
        val c = dot(world, sun)
        println("shell pole vs sun: cos = $c")
        assertTrue("pole misses the sun: cos=$c", c > 0.99999f)
    }
}
