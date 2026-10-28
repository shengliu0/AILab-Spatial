package com.bytedance.ailab.lab

import com.bytedance.ailab.lab.spec.Node
import com.bytedance.ailab.render.MeshData
import java.io.File
import java.util.IdentityHashMap
import org.junit.Test

/**
 * Load-time budget per demo: the device cost of switching to a demo is dominated by creating its
 * entities and uploading its meshes on the main thread (~17 us per vertex measured on the Swan).
 * Prints, per demo: spec build time on this JVM, nodes, distinct meshes, distinct vertices, and the
 * estimated device upload time. Used to decide what to stream / share / defer.
 */
class BuildCostTest {
    @Test
    fun printBuildCosts() {
        val cache = File("build/review-cache").apply { mkdirs() }
        var total = 0.0
        for (d in DemoRegistry.all.filterIsInstance<SpecDemo>()) {
            val t0 = System.nanoTime()
            val root = d.buildSpec(cache)
            val specMs = (System.nanoTime() - t0) / 1e6
            var nodes = 0
            val meshes = IdentityHashMap<MeshData, Boolean>()
            fun rec(n: Node) { nodes++; n.mesh?.let { meshes[it] = true }; n.children.forEach(::rec) }
            rec(root)
            val verts = meshes.keys.sumOf { it.vertexCount.toLong() }
            val estMs = verts * 0.017 + nodes * 0.05
            total += estMs
            println("BUILD %-16s spec %6.0f ms (JVM)  nodes %5d  meshes %4d  verts %7d  -> device upload est %6.0f ms"
                .format(d.id, specMs, nodes, meshes.size, verts, estMs))
        }
        println("BUILD total est %.0f ms".format(total))
    }
}
