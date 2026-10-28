// I18n — the display-language switch (中文 / English).
//
// Every user-visible string is written as tr("中文", "English") at the point it is shown. `Lang.en`
// is Compose snapshot state, so panels recompose the moment it flips; readouts and 3D labels are
// rebuilt every frame from update() and follow on the next frame. Keep format placeholders and
// ${…} templates identical in both halves. Never translate strings used as identities/keys (node
// names, ids, `when` branches) — only what is drawn.

package com.bytedance.ailab.lab

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

object Lang {
    private const val PREFS = "lab"
    private const val KEY = "lang_en"
    private var prefs: android.content.SharedPreferences? = null

    var en by mutableStateOf(false)
        private set

    /** Restore the last choice (called once from the Application). */
    fun init(context: Context) {
        prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        en = prefs?.getBoolean(KEY, false) ?: false
    }

    fun set(english: Boolean) {
        en = english
        prefs?.edit()?.putBoolean(KEY, english)?.apply()
    }
}

/** The one translation primitive: pick the half that matches the current display language. */
fun tr(zh: String, en: String): String = if (Lang.en) en else zh
