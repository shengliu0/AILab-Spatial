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
                if (status == TextToSpeech.SUCCESS) probe(t, context.applicationContext.cacheDir)
            }
        }.getOrNull()
    }

    /**
     * The Swan's engine (Qualcomm) answers isLanguageAvailable(zh) = yes but then logs "no language model found"
     * and stays silent. So each language is proven by synthesising a short phrase to a file: only a non-empty
     * file counts, and the read-aloud button shows only for a language that really speaks.
     */
    private fun probe(t: TextToSpeech, dir: java.io.File) {
        val tests = listOf(Locale.SIMPLIFIED_CHINESE to "你好", Locale.US to "Hello")
        val files = tests.map { java.io.File(dir, "tts_probe_${it.first.language}.wav").also { f -> f.delete() } }
        var left = tests.size
        t.setOnUtteranceProgressListener(object : android.speech.tts.UtteranceProgressListener() {
            override fun onStart(id: String) {}
            override fun onDone(id: String) { if (id.startsWith("probe")) finish(id, true) else speaking = false }
            @Deprecated("API") override fun onError(id: String) { if (id.startsWith("probe")) finish(id, false) else speaking = false }
            override fun onError(id: String, code: Int) { if (id.startsWith("probe")) finish(id, false) else speaking = false }
            private fun finish(id: String, done: Boolean) {
                val k = id.removePrefix("probe").toInt()
                val ok = done && files[k].length() > 1000
                if (k == 0) zhOk = ok else enOk = ok
                android.util.Log.i("Narrator", "tts probe ${tests[k].first}: ${if (ok) "speaks" else "SILENT"} (${files[k].length()} B)")
                if (--left == 0) ready = true
            }
        })
        tests.forEachIndexed { k, (loc, text) ->
            val lang = t.isLanguageAvailable(loc) >= TextToSpeech.LANG_AVAILABLE
            if (!lang) { if (k == 0) zhOk = false else enOk = false; if (--left == 0) ready = true; return@forEachIndexed }
            t.language = loc
            t.synthesizeToFile(text, null, files[k], "probe$k")
        }
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
