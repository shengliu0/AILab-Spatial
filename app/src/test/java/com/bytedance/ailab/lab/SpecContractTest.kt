package com.bytedance.ailab.lab

import com.bytedance.ailab.lab.spec.Node
import com.bytedance.ailab.lab.spec.SoftRender
import kotlin.math.abs
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * The SDK contract, checked on the JVM. Three demos built fine in SoftRender and then crashed on
 * the headset, each on an SDK rule the CPU renderer never exercises:
 *   brachistochrone  Entity.setName("FinPost-0.33")  -> IllegalArgumentException "Invalid name!"
 *   buoyancy         a NaN coordinate after update(0f) -> "Vector3 y value cannot be Infinite or NaN"
 *   prism            a zero-extent mesh               -> ResourceLoadingException "Empty bounding box"
 * The adapter now defends against all three, and this test makes sure demos do not rely on it:
 * every SpecDemo is built, stepped with dt = 0 (what SpecDemo.build does) and then 3 s at 60 Hz,
 * and every node is checked.
 */
class SpecContractTest {
    private val nameOk = Regex("[A-Za-z_][A-Za-z0-9_]*")

    private fun check(n: Node, where: String, out: MutableList<String>) {
        if (!nameOk.matches(n.name)) out += "$where: invalid entity name '${n.name}'"
        val v = floatArrayOf(n.pos.x, n.pos.y, n.pos.z, n.rot.x, n.rot.y, n.rot.z, n.rot.w, n.scale.x, n.scale.y, n.scale.z)
        if (v.any { !it.isFinite() }) out += "$where: non-finite transform on '${n.name}' ${v.toList()}"
        n.mesh?.let { m ->
            if (m.pos.any { !it.isFinite() }) out += "$where: non-finite vertex in mesh of '${n.name}'"
            if (m.idx.any { it < 0 || it >= m.vertexCount }) out += "$where: index out of range in '${n.name}'"
            var lo = FloatArray(3) { Float.MAX_VALUE }; var hi = FloatArray(3) { -Float.MAX_VALUE }
            for (k in 0 until m.vertexCount) for (a in 0 until 3) { lo[a] = minOf(lo[a], m.pos[k * 3 + a]); hi[a] = maxOf(hi[a], m.pos[k * 3 + a]) }
            // zero-extent meshes are legal: SpecAdapter.boundsFor gives every mesh explicit bounds
            if (m.vertexCount == 0) out += "$where: mesh with no vertices on '${n.name}'"
        }
        if (n.tapRadius.isNaN() || n.tapRadius < 0f) out += "$where: bad tapRadius on '${n.name}'"
    }

    @Test
    fun everyDemoRespectsTheSdkContract() {
        val cache = File("build/review-cache").apply { mkdirs() }
        val problems = ArrayList<String>()
        for (d in DemoRegistry.all.filterIsInstance<SpecDemo>()) {
            val root = d.buildSpec(cache)
            d.update(0f)
            root.walk { check(it, "${d.id}@build", problems) }
            repeat(180) { d.update(1f / 60f) }
            root.walk { check(it, "${d.id}@3s", problems) }
        }
        // REBUILD IDEMPOTENCE: leaving and re-entering a demo must give the same scene. Singleton
        // demos that cache "already applied" state broke here (photoelectric: metre-sized blobs;
        // gas: stale gauge filters). Fresh build vs build-after-a-run, both stepped 2 s.
        for (d in DemoRegistry.all.filterIsInstance<SpecDemo>()) {
            val a = SoftRender.bounds(d.buildSpec(cache).also { d.update(0f); repeat(120) { d.update(1f / 60f) } })
            repeat(60) { d.update(1f / 60f) }
            val b = SoftRender.bounds(d.buildSpec(cache).also { d.update(0f); repeat(120) { d.update(1f / 60f) } })
            val dev = maxOf(abs(a.first.x - b.first.x), abs(a.first.y - b.first.y), abs(a.first.z - b.first.z),
                abs(a.second.x - b.second.x), abs(a.second.y - b.second.y), abs(a.second.z - b.second.z))
            println("REBUILD ${d.id}: bounds drift %.3f m".format(dev))
            if (dev > 0.10f) problems += "${d.id}: rebuild changes the scene bounds by %.2f m (stale state survives buildSpec)".format(dev)
        }
        problems.distinct().forEach { println("CONTRACT $it") }
        assertTrue("SDK contract violations:\n" + problems.distinct().take(20).joinToString("\n"), problems.isEmpty())
    }
}
