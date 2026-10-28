# AI Lab · Spatial — PICO SpatialSDK (Kotlin) AI-algorithm demos for the Swan headset

One APK (`com.bytedance.ailab.spatial`, label AI 算法实验室 Spatial), one volumetric window, a top menu of demos,
a per-demo control panel at the front and a floating guided-track (导学) card. Framework shared with the sister apps
(physics / chemistry / math): `lab/` (SpecDemo, LabHost, Lesson), `content/` (volume, menu, lesson card), `render/`.

Model rules: every model is trained locally with PyTorch on synthetic data by `tools/train_<id>.py` (fixed seeds,
deterministic), weights ship in `app/src/main/assets/<id>/`, read through `LabAssets`; Kotlin inference must match
PyTorch reference vectors (≤ 1e-4) in a JVM test. Never ship real-world text or images; generated corpora only.
Package names below that mention physics come from the shared framework template — the code root is
`app/src/main/java/com/bytedance/ailab/`.

## Build / test / review

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@17 ANDROID_HOME=/opt/homebrew/share/android-commandlinetools
./gradlew :app:testDebugUnitTest                                   # all JVM tests (physics + review)
./gradlew :app:testDebugUnitTest --tests "*LabReviewTest*" -Dreview.only=<id>   # render one demo
python3 ../visual_review/score_shots.py app/build/review/shots      # score (0-100, higher = better)
./gradlew assembleDebug                                             # APK
```

The review test renders every `SpecDemo` headless (2 s simulated at 60 Hz) with the CPU renderer
`lab/spec/SoftRender` into `app/build/review/shots/demo_<id>_{hero,side,detail}.png` (scored) and
`app/build/review/viewer/demo_<id>_viewer.png` (what the user sees, 2.5 m in front). LOOK at the PNGs
(the Read tool shows images) — the score is a guide, your eyes are the judge. SoftRender approximates
the headset (Unlit exact; PBR = key+fill+hemisphere ambient); it is for composition, colour, contrast,
readability. Scores are only comparable between versions of the SAME demo in the SAME renderer.

On a device (debug builds): `adb shell am broadcast -a com.bytedance.physicslab.SHOW --es demo <id>
-p com.bytedance.physicslab.spatial` switches demo. `adb screencap` does NOT work on the Swan.

## Architecture

| package | Android/SDK? | what |
|---|---|---|
| `core/<topic>/` | no | physics, pure Kotlin, JVM-tested. Port of the Unity `Assets/PhysicsLab/*Core.cs` |
| `render/` | no | `ProcTex` (noise, OKLab, bakes), `MeshGen` (sphere/tube/annulus/cylinder/box/tubeInto), `Palette` |
| `lab/spec/` | no | `Spec.kt` scene graph (`Node`, `Mat.Unlit`, `Mat.Pbr`, `Blend`, `LabelSpec`), `SoftRender` |
| `lab/` | yes | `SpecDemo` base, `SpecAdapter` (Spec -> entities), `Gfx`, `LabHost`, `DemoRegistry`, `ui/LabUi` |
| `demos/<id>/` | Compose only | one `object XxxDemo : SpecDemo()` per experiment |
| `demos/solar/` | yes | the solar system (predates Spec; drives entities directly — do not copy its style) |

**The exemplar is `demos/pendulum/PendulumDemo.kt` + `core/pendulum/PendulumCore.kt`.** Read both first.

### Writing a demo (SpecDemo)

```kotlin
object FooDemo : SpecDemo() {
    override val id = "foo"; override val title = "中文名"; override val credit = "Who · year"
    override fun buildSpec(cacheDir: File): Node { /* build the tree ONCE; pure Kotlin */ }
    override fun update(dt: Float) { /* physics step + mutate node pos/rot/scale/visible/mat; set readoutText; fill labels */ }
    override fun isTarget(name: String) = name.startsWith("Bob")      // gaze+pinch targets (node names)
    override fun onTap(name: String) { ... }
    @Composable override fun Panel() { LabPanel(readout) { /* SegmentControl / LabSlider / LabSwitch / Button */ } }
}
```

* Nodes: `root.add(name, mesh, mat, pos)`. Per frame only change `pos/rot/scale/visible`, `setMat(...)`,
  or edit a MeshData IN PLACE (`MeshGen.tubeInto`) then `meshEdited(md)`. Never create a new MeshData
  per frame (the adapter caches GPU meshes by identity — that would leak).
* Share MeshData/Image instances between nodes. Unit meshes + `scale` are the norm (`MeshGen.cylinder(1f,1f)`).
* Rotations: `lab.Q` (`Q.axisAngle(x,y,z,rad)`, `Q.fromTo(...)`). Right-handed, +Y up, +Z toward viewer.
* Materials (sRGB colours): `Mat.Pbr(color, metallic, roughness, alpha<1 = glass/water, tex, emissive)`
  for hardware (lit by the real room on device); `Mat.Unlit(color, alpha, tex, Blend.OPAQUE|TRANSPARENT|ADD)`
  for beams, glows, traces, fields, overlays. ADD = light (beams, sparks, field glow).
* Textures: bake with `ProcTex` into `render.Image` (packed ARGB, row 0 = top). Bakes run on a worker
  thread at build; anything over ~50 ms on the Mac should go through `lab.BakeCache.load(cacheDir, key,
  version) { listOf(...) }` (works in JVM tests too) and bump `version` whenever the bake changes.
* Labels: up to 6 `LabelSpec(text, pos)` in `labels` (volume coordinates), rendered as SpatialUI chips.
* Readout: `readoutText` = 3 short lines (Chinese first, key numbers live). Published at 10 Hz.
* Gaze targets: set `node.tapRadius` (node-local units) so the WORLD radius is >= 0.045 m
  (eye tracking error ~1-1.5 deg at 2.5 m). Visible size can be smaller. List each one in
  `tapTargets` too (see "Interaction contract").

### Volume layout (metres, origin = volume centre)

Content: |x| < 0.9, |z| < 0.6, -0.27 < y < 0.45. Top menu sits at y = 0.50 (front), control panel at
y = -0.36, z = 0.92 (front, ~0.3 m tall) — keep the important action in -0.20 < y < 0.42 so the panel
never covers it. Default viewer stands ~2.5 m in front at eye height ≈ y 0.1: things smaller than
~1 cm are invisible, text belongs in labels/readout, and depth-only motion is hard to read — orient
motions across the line of sight.

## Rules

- 2-D UI is SpatialUI only (`com.pico.spatial.ui.design.*`) inside `PicoTheme`. No Material/Material3.
  Use `lab/ui/LabUi.kt` (`LabPanel`, `LabSlider`, `LabSwitch`) so every panel looks the same.
- Speed: every time-based demo offers 3 speeds (slow / medium / real) — the user asked for this.
- Physics lives in `core/<topic>/` with JVM tests that check real numbers (closed forms, conservation
  laws, textbook/tabulated values). AUDIT the Unity source: the pendulum port found the Unity app used
  the small-angle formula at 32 deg and claimed amplitude-independent periods — that is the kind of
  error to find and fix, and to report.
- Keep `core/`, `render/`, `lab/spec/` free of Android/SDK imports.
- Device cost (measured on the Swan): re-uploading a mesh costs ~20 µs per VERTEX (the SDK only takes
  boxed per-vertex lists). Per frame animate with node pos/rot/scale/visible and `setMat`; particles and
  molecules are pooled nodes sharing one mesh; `meshEdited` per frame only for small meshes (< ~300
  vertices per frame in total); keep visible nodes < ~600. `SpecAdapter` budgets uploads at 3 ms/frame.
- UI text through `tr(zh, en)`, Chinese default. Teach one clear idea; make the user DO something.
- Solar-system physics changes must keep `SolarCoreParityTest` green.

## Adding a demo (checklist)

1. `core/<topic>/` physics + JVM tests; `demos/<id>/XxxDemo.kt` (a `SpecDemo`).
2. One line in `lab/DemoRegistry.catalog` with its `Subject` (= catalogue tab).
3. `./gradlew :app:testDebugUnitTest` — must stay green, including:
   - `SpecContractTest`: SDK rules the CPU renderer never exercises — entity names `[A-Za-z0-9_]`
     only (`setName` throws "Invalid name!"), no NaN/Inf after `update(0f)` (the device build calls
     it once with dt = 0), and REBUILD IDEMPOTENCE (leaving and re-entering a demo must give the same
     scene: reset every cached "already applied" field and every filter in `buildSpec`).
4. `python3 tools/make_thumbs.py` -> `res/drawable-nodpi/thumb_<id>.png` (the catalogue card picture,
   cropped from the review render), then `assembleDebug`.
5. Store assets: `./gradlew :app:testDebugUnitTest --tests "*ShowcaseRenderTest*" -Dshowcase=true` (HD stills,
   auto-framed), copy to `docs/screenshots/`, then `python3 tools/make_poster.py`. The icon (store + PICO
   3-D layers + SDF relief maps) is `python3 tools/make_icons.py`.

## Menu (content/LabMenu.kt)

Top bar (always): 实验目录 / ‹ / current title + credit / › / EN|中文, every target 96 dp (7.7 cm, 1.8 deg).
Catalogue (on demand, scene + panel hidden while open): subject tabs (GazeUi `SegmentControl`, 76 dp)
+ 5 x 3 picture cards 225 x 188 dp (4.1 x 3.4 deg), 15 per page. Gaze feedback = system hover via
`gazeFeedback(grow = 1.10, dim = 0.72)`: the looked-at card grows, every other card dims to 72 %.
Rule of thumb for any gaze target: >= 2 deg (>= 9 cm at 2.5 m); the old 0.7 deg menu was unusable.

## Interaction contract (every demo)

Users cannot tell what in a 3-D scene is touchable, so every pinchable object has a PANEL TWIN:
- `override val tapTargets = listOf(TapTarget(node, zh, en))` — `LabPanel` appends a
  "✋ 也可直接捏合 / Or pinch directly" row whose buttons call `LabHost.onTap(node)`, i.e. exactly the
  same code path as pinching the object. Two-way sync: panel controls bind to demo fields through
  `FieldRef(::field)` / `MappedRef`, and `LabHost.onTap` bumps `uiEpoch`, which re-keys the panel.
- Breathing halos (`SpecDemo.ensureHalos`): every node with `tapRadius > 0` gets a light-blue billboard
  ring, radius `min(tapRadius, 1.6 x drawn mesh radius)`; strong pulse for 15 s after the demo opens,
  then faint. Nothing to do per demo — set `tapRadius`, the halo follows.
- The controls attachment is never empty (an empty attachment never got a size on device -> panel
  missing): it shows `Loading…` until the demo exists and is hidden via its entity while the
  catalogue is open. Log line `LabPanel <id> panel WxH px` proves it laid out.
- Drag: `isDragTarget(name)` + `onDragStart/onDrag(name, dx, dy, dz)/onDragEnd` (metres in volume
  space; px -> dp -> m via `LocalPhysicalLengthConverter`, Y flipped). Shared Space only reports the
  hand DURING a pinch-drag; there is no hover hand position.
- Text: every user-visible string is `tr("中文", "English")` (`lab/I18n.kt`, persisted). `I18nTest`
  fails on a CJK literal outside `tr(` (exempt: comments, `Log.`, `TapTarget(`, `// i18n-ok`).
- Buttons/segments: use `lab/ui/GazeUi.kt` drop-ins (`SegmentControl`, `SegmentItem`, `Button`, min
  76 x 112 dp) instead of the SpatialUI ones (~40 dp, below the eye-tracking error).
- Rows of controls are `LabRow { }` (a FlowRow), never `Row`: English is ~1.6x wider than Chinese and a
  plain Row squeezed its last chip until the text stood vertically ("C/l/o/s/e/-/u/p"). `SegmentControl`
  is a flow row too. No "提示：…" lines — the pinch row says it ("object → action").
- Tap feedback is automatic: `LabHost.onTap` flashes the object's halo (0.5 s ping) and shows a note
  above it for 2.6 s. Override `tapMessage(name)` when the generic "✓ <TapTarget label>" would be wrong
  or vague (toggles: say the NEW state; predictions: say what to watch). `TapTarget(..) { isOn }` lights
  the panel twin while that object is on. `TapFeedbackTest` taps every target in both languages.
- Readout: 3 lines that each FIT the panel (984 dp at 22 sp) in both languages.
- Pinch = tap, our way (read from the SDK bytecode, see `LabVolume`): `detectSpatialTapGesture`
  re-hit-tests at RELEASE, needs the same entity at down and up and an OS "TAP" verdict — moving
  targets, a glance away, slow pinches failed silently while the highlight showed. 3-D taps now use
  `detectSpatialPointerEvent` (entity locked at the down; release within 1.5 s, no drag = tap). 2-D
  controls use `Modifier.gazeClick` (never `clickable`, which cancels when the hand-driven pointer
  drifts out of bounds) and no `clip` on the hit node (the hit test uses the clip's rounded outline).
  Hover `grow` <= 1.05: the OS draws it, the hit area stays 1.0x. Every pinch logs `LabTap` with its
  duration and travel; debug builds also turn on the SDK logger (`src/debug/AndroidManifest.xml`):
  `adb logcat -s LabTap SUISpatialGesture`.
- LOOK at the UI: `./gradlew :app:testDebugUnitTest --tests "*UiPreviewTest*"` renders every panel, the
  top bar and the catalogue (Robolectric native graphics, the real Compose code) to
  `app/build/review/ui/{zh,en}_*.png`, and FAILS if any readout line wraps. The PICO glass material does
  not render there (flat grey backdrop, light-theme text colours).

### Solar comets (demos/solar/CometSandbox.kt + core/comet/CometCore.kt)

Pinch the ☄ icon (front right, volume (0.70, 0.05, 0.52)), swing, release: the comet follows the hand (gain 1.2) with a live
predicted orbit; on release speed = k(v_hand) x local circular speed, k(v) = 0.2 + 1.441 tanh(0.695 (v-0.1))
(drop -> sun-grazer, 0.5 m/s -> ellipse, 1 m/s -> ~circle, 2 m/s -> escapes, saturates at 1.64).
v_hand = peak 40 ms-window speed in the last 0.25 s (people decelerate before opening the pinch).
Direction = hand direction with 30 % of its out-of-plane part; mostly-vertical throws go prograde.
Gravity mu is derived from the DRAWN Earth orbit, time in display days, so the time slider applies.
Max 5 comets; tapping the icon (or the panel button) launches a k = 1 prograde comet.

## Device facts learned the hard way (PICO OS 6 / Swan, SDK 0.13.3)

- Straight-alpha bitmaps; unlit materials skip tone mapping; overlays never write depth (all in `Gfx`).
- `rayCast` against colliders works (self-check 8/8) but only ~1 s after the colliders are created.
- Small gaze targets fail; the solar system ran 90.0 Hz (worst frame 12.4 ms) on device.
- Cold texture bakes are ~15x slower on the headset than on the Mac JVM — cache or keep them small.
- Volume content outside the window bounds is clipped (the panel was cut off once).
- `MeshResource.createWithMeshModel` throws "Empty bounding box" for zero-extent meshes; the adapter
  now passes explicit bounds (min 1 mm half-extent) for every mesh.
- `adb screencap`/`screenrecord` fail on the Swan; only the user's in-headset system screenshot works.
