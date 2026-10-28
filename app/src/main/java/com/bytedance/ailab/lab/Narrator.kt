// Narrator — reads the background card aloud with the headset's system text-to-speech, in the display
// language. Optional: if the engine has no voice for that language the read-aloud button is hidden.

package com.bytedance.ailab.lab

import android.content.Context
import android.speech.tts.TextToSpeech
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import java.util.Locale

object Narrator {
    private var tts: TextToSpeech? = null
    private var zhOk = false
    private var enOk = false
    /** Compose-observed: the engine is ready. */
    var ready by mutableStateOf(false)
        private set
    var speaking by mutableStateOf(false)
        private set

    fun init(context: Context) {
        if (tts != null) return
        tts = runCatching {
            TextToSpeech(context.applicationContext) { status ->
                val t = tts ?: return@TextToSpeech
                if (status == TextToSpeech.SUCCESS) {
                    zhOk = t.isLanguageAvailable(Locale.SIMPLIFIED_CHINESE) >= TextToSpeech.LANG_AVAILABLE
                    enOk = t.isLanguageAvailable(Locale.US) >= TextToSpeech.LANG_AVAILABLE
                    android.util.Log.i("Narrator", "tts ready: zh=$zhOk en=$enOk")
                    ready = true
                }
            }
        }.getOrNull()
    }

    /** A voice exists for the current display language. */
    val available: Boolean get() = ready && (if (Lang.en) enOk else zhOk)

    fun speak(text: String) {
        val t = tts ?: return
        if (!available) return
        t.language = if (Lang.en) Locale.US else Locale.SIMPLIFIED_CHINESE
        t.speak(text, TextToSpeech.QUEUE_FLUSH, null, "about")
        speaking = true
    }

    fun stop() { runCatching { tts?.stop() }; speaking = false }
}
