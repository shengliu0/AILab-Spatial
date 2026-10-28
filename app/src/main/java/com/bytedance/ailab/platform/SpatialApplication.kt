package com.bytedance.ailab.platform

import android.app.Application
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ApplicationInfo
import android.os.Handler
import android.os.Looper
import android.util.Log
import com.bytedance.ailab.lab.LabHost
import com.bytedance.ailab.mainApp
import com.pico.spatial.ui.foundation.dsl.launch

class SpatialApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        com.bytedance.ailab.lab.Lang.init(this)
        com.bytedance.ailab.lab.LabSettings.init(this)
        launch(::mainApp)
        // DEBUG builds only: `adb shell am broadcast -a com.bytedance.ailab.SHOW --es demo <id>`
        // switches demo without gaze/pinch, so the screenshot review loop can run unattended.
        if (applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0) {
            val main = Handler(Looper.getMainLooper())
            registerReceiver(object : BroadcastReceiver() {
                override fun onReceive(c: Context, i: Intent) {
                    if (i.getStringExtra("stress") != null) { main.post { LabHost.stressRequested = true }; return }
                    val id = i.getStringExtra("demo") ?: return
                    Log.i("LabHost", "debug switch -> $id")
                    main.post { LabHost.requested = id }
                }
            }, IntentFilter("com.bytedance.ailab.SHOW"), Context.RECEIVER_EXPORTED)
        }
    }
}
