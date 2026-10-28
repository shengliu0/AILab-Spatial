package com.bytedance.ailab

import com.bytedance.ailab.content.LabVolume
import com.bytedance.ailab.lab.DemoRegistry
import com.bytedance.ailab.lab.LabHost
import com.pico.spatial.ui.design.PicoTheme
import com.pico.spatial.ui.foundation.dsl.DefaultWindowContainer
import com.pico.spatial.ui.foundation.dsl.SpatialAppScope

fun mainApp(scope: SpatialAppScope) =
    with(scope) {
        LabHost.register(DemoRegistry.all)
        DefaultWindowContainer {
            PicoTheme {
                LabVolume(firstDemo = DemoRegistry.FIRST)
            }
        }
    }
