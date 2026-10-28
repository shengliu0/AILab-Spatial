package com.bytedance.ailab.lab.ui

import kotlin.reflect.KMutableProperty0
import kotlin.reflect.KProperty

/**
 * `var x by FieldRef(::demoField)` — a panel variable that IS the demo's field (reads and writes go
 * straight through). Panels used to keep `remember { mutableStateOf(field) }` COPIES, so pinching an
 * object changed the demo but the panel kept showing the old choice. The panel recomposes on every
 * readout change and on every pinch (LabHost.uiEpoch), so reading the field directly is enough.
 */
class FieldRef<T>(private val p: KMutableProperty0<T>) {
    operator fun getValue(thisRef: Any?, property: KProperty<*>): T = p.get()
    operator fun setValue(thisRef: Any?, property: KProperty<*>, value: T) = p.set(value)
}

/** Same, through a type conversion (e.g. a Double field driven by a Float slider). */
class MappedRef<F, T>(private val p: KMutableProperty0<F>, private val to: (F) -> T, private val from: (T) -> F) {
    operator fun getValue(thisRef: Any?, property: KProperty<*>): T = to(p.get())
    operator fun setValue(thisRef: Any?, property: KProperty<*>, value: T) = p.set(from(value))
}
