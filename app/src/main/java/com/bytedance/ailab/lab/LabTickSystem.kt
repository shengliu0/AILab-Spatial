package com.bytedance.ailab.lab

import com.pico.spatial.core.ecs.SceneUpdateContext
import com.pico.spatial.core.ecs.System

/** Per-frame driver. Systems are instantiated by the engine (no-arg constructor), so all state
 *  lives in LabHost / the demo objects and this class only forwards the frame. */
class LabTickSystem : System() {
    override fun update(context: SceneUpdateContext) {
        LabHost.tick(context.deltaTime, context.scene)
    }
}
