package com.bytedance.ailab.lab.spec

import com.bytedance.ailab.core.P3
import com.bytedance.ailab.lab.Q
import com.bytedance.ailab.render.MeshGen
import com.bytedance.ailab.render.ProcTex
import com.bytedance.ailab.render.Rgb
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Day/night must be visible: a white unlit ball under a sun-facing light shell, sun on +X, camera
 * on +Z. The +X half of the disc must be bright and the -X half dark. (Found broken: the Earth in
 * the solar close-up rendered uniformly lit.)
 */
class LightShellTest {
    @Test
    fun terminatorIsVisible() {
        val root = Node("r")
        val ball = root.add("ball", MeshGen.sphere(1f, 32, 64, MeshGen.SphereUv.EQUIRECT), Mat.Unlit(Rgb(1f, 1f, 1f)))
        ball.uniformScale(0.1f)
        val shell = root.add("shell", MeshGen.sphere(1f, 64, 64, MeshGen.SphereUv.POLAR_U),
            Mat.Unlit(tex = ProcTex.bakeLightShell(256), blend = Blend.TRANSPARENT))
        shell.uniformScale(0.1004f)
        shell.rot = Q.fromTo(0f, 1f, 0f, 1f, 0f, 0f)          // shell +Y (sub-solar point) -> +X
        val img = SoftRender.render(root, Cam(P3(0f, 0f, 0.6f), P3(0f, 0f, 0f), 40f), 320, 240, 1)
        SoftRender.writePng(img, File("build/review/lightshell_test.png"))
        fun lum(x: Int, y: Int) = img.px[y * 320 + x].let { ((it ushr 16) and 0xFF) + ((it ushr 8) and 0xFF) + (it and 0xFF) } / 3
        val day = lum(160 + 45, 120); val night = lum(160 - 45, 120); val mid = lum(160, 120)
        println("LIGHTSHELL day=$day mid=$mid night=$night")
        assertTrue("day side not bright: $day", day > 180)
        assertTrue("night side not dark: $night", night < 30)
    }
}
