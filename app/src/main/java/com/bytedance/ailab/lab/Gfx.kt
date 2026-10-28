// Gfx — the shared SpatialSDK toolkit every demo builds on. It is the only place that knows how a
// baked Image becomes a texture, how a MeshData becomes a mesh, and which material settings are
// safe on PICO OS 6. Lessons already paid for on the headset are encoded here:
//   * textures are STRAIGHT alpha (setPremultiplied(false) before setPixels)
//   * unlit materials skip tone mapping (bakes are already display-referred)
//   * transparent/additive overlays never write depth and render double-sided
//   * gaze targets get a minimum WORLD radius (eye-tracking error is ~1-1.5 deg; at 2.5 m that is
//     ~45 mm), independent of how small the visible body is

package com.bytedance.ailab.lab

import android.graphics.Bitmap
import android.util.Log
import com.bytedance.ailab.core.P3
import com.bytedance.ailab.render.Image
import com.bytedance.ailab.render.MeshData
import com.bytedance.ailab.render.ProcTex
import com.bytedance.ailab.render.Rgb
import com.pico.spatial.core.ecs.CollisionComponent
import com.pico.spatial.core.ecs.Entity
import com.pico.spatial.core.ecs.HoverEffectComponent
import com.pico.spatial.core.ecs.InteractableComponent
import com.pico.spatial.core.ecs.ModelComponent
import com.pico.spatial.core.ecs.TransformComponent
import com.pico.spatial.core.ecs.resource.BlendingMode
import com.pico.spatial.core.ecs.resource.Material
import com.pico.spatial.core.ecs.resource.MaterialCullingMode
import com.pico.spatial.core.ecs.resource.MeshModel
import com.pico.spatial.core.ecs.resource.MeshResource
import com.pico.spatial.core.ecs.resource.PhysicallyBasedMaterial
import com.pico.spatial.core.ecs.resource.PhysicsMaterialResource
import com.pico.spatial.core.ecs.resource.ShapeResource
import com.pico.spatial.core.ecs.resource.TextureResource
import com.pico.spatial.core.ecs.resource.UnlitMaterial
import com.pico.spatial.core.math.Color4
import com.pico.spatial.core.math.Quat
import com.pico.spatial.core.math.Vector2
import com.pico.spatial.core.math.Vector3
import java.io.File
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

private const val TAG = "Gfx"

// ---------------------------------------------------------------- colour

/** sRGB -> linear Color4 (Color4 inputs are linear; bakes/textures are sRGB-decoded by the GPU). */
fun lin(c: Rgb, a: Float = 1f) = Color4(ProcTex.toLinear(c.r), ProcTex.toLinear(c.g), ProcTex.toLinear(c.b), a)
fun lin(r: Float, g: Float, b: Float, a: Float = 1f) = lin(Rgb(r, g, b), a)

// ---------------------------------------------------------------- resources

fun tex(img: Image): TextureResource {
    val bmp = Bitmap.createBitmap(img.w, img.h, Bitmap.Config.ARGB_8888)
    bmp.setPremultiplied(false)            // straight alpha, set BEFORE the pixels go in
    bmp.setPixels(img.px, 0, img.w, 0, 0, img.w, img.h)
    return TextureResource(bmp)
}

fun toModel(d: MeshData): MeshModel {
    val n = d.vertexCount
    val pos = ArrayList<Vector3>(n); val nrm = ArrayList<Vector3>(n); val uv = ArrayList<Vector2>(n)
    for (k in 0 until n) {
        pos += Vector3(d.pos[k * 3], d.pos[k * 3 + 1], d.pos[k * 3 + 2])
        nrm += Vector3(d.nrm[k * 3], d.nrm[k * 3 + 1], d.nrm[k * 3 + 2])
        uv += Vector2(d.uv[k * 2], d.uv[k * 2 + 1])
    }
    return MeshModel(pos, d.idx.toList(), nrm, null, uv)
}

fun mesh(d: MeshData, name: String): MeshResource = MeshResource.createWithMeshModel(toModel(d), null, name)

/** Replace a mesh's geometry in place (animated tubes, growing arcs, wave surfaces). */
fun MeshResource.set(d: MeshData) { replaceWithMeshModel(toModel(d)) }

// ---------------------------------------------------------------- materials

/** Unlit, tone-mapping off. OPAQUE back-culls; overlays (TRANSPARENT/ADD) are double-sided, no depth write. */
fun unlit(mode: BlendingMode = BlendingMode.OPAQUE, texture: TextureResource? = null,
          color: Color4? = null, opacity: Float = 1f): UnlitMaterial =
    UnlitMaterial.create(mode).apply {
        setApplyToneMapping(false)
        if (texture != null) setBaseColorTexture(texture)
        if (color != null) setBaseColor(color)
        if (mode == BlendingMode.OPAQUE) setCullingMode(MaterialCullingMode.BACK)
        else { setCullingMode(MaterialCullingMode.NONE); setDepthWrite(false); setOpacity(opacity) }
    }

/**
 * Physically based: lit by the system's environment (IBL from the real room in MR) plus any scene
 * lights. Use for lab hardware — brass, steel, glass, water — so props sit in the room the user
 * is standing in instead of floating as flat colour.
 */
fun pbr(color: Color4, metallic: Float = 0f, roughness: Float = 0.5f,
        mode: BlendingMode = BlendingMode.OPAQUE, opacity: Float = 1f,
        texture: TextureResource? = null, emissive: Color4? = null): PhysicallyBasedMaterial =
    PhysicallyBasedMaterial.create(mode).apply {
        setBaseColor(color); setMetallic(metallic); setRoughness(roughness)
        if (texture != null) setBaseColorTexture(texture)
        if (emissive != null) setEmissiveColor(emissive) else setEmissiveColor(Color4(0f, 0f, 0f, 1f))
        if (mode != BlendingMode.OPAQUE) { setOpacity(opacity); setDepthWrite(false) }
    }

// ---------------------------------------------------------------- entities

fun model(mesh: MeshResource, mat: Material, name: String) = Entity().apply {
    setName(name)
    components.set(ModelComponent(mesh, mat))
}

fun group(name: String, parent: Entity? = null) = Entity().apply { setName(name); parent?.addChild(this) }

fun collider(radiusLocal: Float) =
    CollisionComponent(listOf(ShapeResource.createSphere(radiusLocal)), PhysicsMaterialResource())

fun boxCollider(x: Float, y: Float, z: Float) =
    CollisionComponent(listOf(ShapeResource.createBox(Vector3(x, y, z))), PhysicsMaterialResource())

/** Make [e] a gaze+pinch target with a sphere collider of [radiusLocal] (entity-local units). */
fun makeTappable(e: Entity, radiusLocal: Float, hover: Boolean = true) {
    e.components.set(collider(radiusLocal))
    e.components.set(InteractableComponent())
    if (hover) e.components.set(HoverEffectComponent())
}

// ---------------------------------------------------------------- transforms

fun tf(e: Entity) = e.components[TransformComponent::class.java]!!
fun pos(e: Entity, p: P3) { tf(e).position = Vector3(p.x, p.y, p.z) }
fun pos(e: Entity, x: Float, y: Float, z: Float) { tf(e).position = Vector3(x, y, z) }
fun rot(e: Entity, q: Q) { tf(e).quaternion = Quat(q.x, q.y, q.z, q.w) }
fun setScale(e: Entity, s: Float) { tf(e).scaleVector = Vector3(s, s, s) }
fun scale3(e: Entity, x: Float, y: Float, z: Float) { tf(e).scaleVector = Vector3(x, y, z) }

// ---------------------------------------------------------------- bake cache

/**
 * Bakes are pure CPU work (often seconds on the headset's cores) and deterministic, so each demo
 * caches its images in the app cache keyed by [key]-[version]. First launch on Swan measured 9.1 s
 * serial for the solar system; the cached read is ~10 ms. Bump [version] whenever a bake changes.
 */
object BakeCache {
    @Volatile var lastSource = "?"

    fun load(dir: File, key: String, version: Int, produce: () -> List<Image>): List<Image> {
        val f = File(dir, "bake-$key-v$version.bin")
        if (f.isFile) try {
            java.io.DataInputStream(java.io.BufferedInputStream(f.inputStream(), 1 shl 16)).use { inp ->
                val n = inp.readInt()
                val imgs = List(n) {
                    val w = inp.readInt(); val h = inp.readInt()
                    val bytes = ByteArray(w * h * 4).also { b -> inp.readFully(b) }
                    Image(w, h).also { img -> java.nio.ByteBuffer.wrap(bytes).asIntBuffer().get(img.px) }
                }
                lastSource = "cache"
                return imgs
            }
        } catch (e: Exception) {
            Log.w(TAG, "bake cache $key unreadable, rebaking", e)
        }
        val imgs = produce()
        try {
            val tmp = File(dir, "bake-$key.tmp")
            java.io.DataOutputStream(java.io.BufferedOutputStream(tmp.outputStream(), 1 shl 16)).use { out ->
                out.writeInt(imgs.size)
                for (img in imgs) {
                    out.writeInt(img.w); out.writeInt(img.h)
                    val bb = java.nio.ByteBuffer.allocate(img.px.size * 4)
                    bb.asIntBuffer().put(img.px)
                    out.write(bb.array())
                }
            }
            tmp.renameTo(f)
            dir.listFiles()?.filter { it.name.startsWith("bake-$key-") && it.name != f.name }?.forEach { it.delete() }
        } catch (e: Exception) {
            Log.w(TAG, "could not write bake cache $key", e)
        }
        lastSource = "baked"
        return imgs
    }
}

// ---------------------------------------------------------------- quaternion

/** Tiny quaternion (x,y,z,w) so orientation maths does not depend on SDK angle conventions. */
data class Q(val x: Float, val y: Float, val z: Float, val w: Float) {
    fun conj() = Q(-x, -y, -z, w)
    operator fun times(o: Q) = Q(
        w * o.x + x * o.w + y * o.z - z * o.y,
        w * o.y - x * o.z + y * o.w + z * o.x,
        w * o.z + x * o.y - y * o.x + z * o.w,
        w * o.w - x * o.x - y * o.y - z * o.z)
    fun rotate(v: FloatArray): FloatArray {
        val r = this * Q(v[0], v[1], v[2], 0f) * conj()
        return floatArrayOf(r.x, r.y, r.z)
    }
    companion object {
        fun identity() = Q(0f, 0f, 0f, 1f)
        fun axisAngle(ax: Float, ay: Float, az: Float, rad: Float): Q {
            val s = sin(rad / 2); val l = sqrt(ax * ax + ay * ay + az * az).coerceAtLeast(1e-12f)
            return Q(ax / l * s, ay / l * s, az / l * s, cos(rad / 2))
        }
        /** Shortest rotation taking vector a onto b. */
        fun fromTo(ax: Float, ay: Float, az: Float, bx0: Float, by0: Float, bz0: Float): Q {
            val la = sqrt(ax * ax + ay * ay + az * az); val lb = sqrt(bx0 * bx0 + by0 * by0 + bz0 * bz0)
            if (la < 1e-12f || lb < 1e-12f) return identity()
            val ux = ax / la; val uy = ay / la; val uz = az / la
            val bx = bx0 / lb; val by = by0 / lb; val bz = bz0 / lb
            val d = ux * bx + uy * by + uz * bz
            if (d < -0.999999f) {
                var px = -uy; var py = ux; var pz = 0f
                if (px * px + py * py < 1e-6f) { px = 0f; py = -uz; pz = uy }
                return axisAngle(px, py, pz, PI.toFloat())
            }
            val cx = uy * bz - uz * by; val cy = uz * bx - ux * bz; val cz = ux * by - uy * bx
            val q = Q(cx, cy, cz, 1f + d)
            val l = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
            return Q(q.x / l, q.y / l, q.z / l, q.w / l)
        }
    }
}
