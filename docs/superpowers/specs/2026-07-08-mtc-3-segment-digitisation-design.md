# MTC 3-segment split — digitisation/mapping design

Date: 2026-07-08
Component: `SND/MTC`, `python/SciFiMapping.py`, `python/detectors/MTCDetector.py`

## Goal

The MTC is split longitudinally into **3 blocks** of transverse size 40×40, 50×50,
60×60 cm (smallest upstream), 15 layers each, total length unchanged (~336 cm).
The **geometry** for this is already implemented (commit `61270de8`). This spec
covers the remaining half: making the **SiPM mapping and digitisation** chain
block-aware so the full sim → digi chain works for all three sizes.

## Current state (already done, not re-designed here)

`ConstructGeometry` builds one shared `MTC` envelope (60×60 × total length) and,
per block `iB` (0..2):

- one iron volume `MTC_ironVol_<iB>` (size `lW[iB]`, field applied), and
- one sensitive-module assembly `MTC_layer_<iB>` containing the SciFi (U/V) and
  scintillator planes, built via `CreateSciFiModule`/`CreateScintModule` with the
  block width/height.

Layers are placed into the envelope with the **global** copy number `i` (0..44);
the block is `iB = i / fnLayPerBlock` (`fnB = 3`, `fnLayPerBlock = fLayers/fnB`).

Segment sizes are hardcoded in C++: `lW = {fWidth-20, fWidth-10, fWidth}` and
likewise `lH`. `fWidth`/`fHeight` (60, from `MTC_config.yaml`) define the envelope
maximum only.

### Detector-ID scheme (UNCHANGED)

`ProcessHits` sets `fVolumeID = 1e8 + layer*1e6 + local_id`, where `layer` comes
from `nav->GetMother(3)->GetNumber()` = the global copy number `i` (0..44). The
node hierarchy `…/MTC_layer_<iB>_<i>/MTC_scifi_U_0/MTC_epoxyMat_0/FiberVol_<n>`
keeps `GetMother(3)` returning `i`.

Consequences:

- No new det-ID digit is required. **Block = `layer / fnLayPerBlock`.**
- SiPM global channel `1e8 + layer*1e6 + (plane*1e5 + mat*1e4 + sipm*1e3 + chan)`
  stays collision-free: `layer` distinguishes the block, and the widest block
  needs `ceil(width / (fFiberPitch·fChannelAggregated)) = ceil(60 / (0.025·4)) =
  ceil(60/0.1) = 600` channels (40→400, 50→500), all `< kMaxChannelsPerSiPM
  (1000)`, so the 3-digit channel slot never overflows and `fNSiPMs` stays 1 for
  all blocks.

## Problems in the current SiPM code (single-width assumptions)

1. **Stale navigation paths.** `GetPosition`, `GetLocalPos`, `GetSiPMPosition`
   navigate to `/cave/MTC_1/MTC_layer_<station>/…`, but nodes are now
   `MTC_layer_<iB>_<station>`. Inner names (`MTC_scifi_U_0`, `MTC_epoxyMat_0`,
   `FiberVol_<n>`) are shared across blocks — fine for explicit `nav->cd(path)`,
   only a problem for global `FindVolumeFast`.
2. **Single-width state overwritten per block.** `fFiberLength`, `fNSiPMChan`,
   `fNSiPMs`, `SiPMPos_U/V`, and the four fibre↔SiPM maps are scalars/single-maps.
   After construction `fFiberLength` holds only block-2's value; one map cannot
   represent 400/500/600 channels.
3. **`FindVolumeFast` ambiguity.** `SiPMmapping` uses
   `FindVolumeFast("MTC_scifi_U")`/`("SiPMmapVolU")`, which returns only block-0's
   volume because inner names are duplicated across blocks.
4. **Array-overflow guard.** `iB = i / fnLayPerBlock` overflows the size-`fnB`
   arrays if `fLayers % fnB != 0` (safe at 45, latent otherwise).

## Design

Make all width-dependent SiPM state **per block**, indexed `0..fnB-1`.

### Data structures (`MTCDetector.h`)

Replace the single containers with `std::vector` sized `fnB`:

```
std::vector<std::map<Int_t, std::map<Int_t, std::array<float,2>>>>
    fibresSiPM_U, siPMFibres_U, fibresSiPM_V, siPMFibres_V;
std::vector<std::map<Int_t, float>> SiPMPos_U, SiPMPos_V;
std::vector<Int_t>    fNSiPMChan, fNSiPMs;
std::vector<Double_t> fFiberLength;   // per block
```

`fnB`/`fnLayPerBlock` already exist. Add a helper `Int_t BlockForLayer(Int_t
layer) const { return std::min(layer / fnLayPerBlock, fnB - 1); }` (also used to
clamp the geometry loop, fixing problem 4).

### C++ methods

- **`SiPMOverlap(Int_t iB)`** — parametrise by block: use `lW[iB]` (recompute from
  `fWidth` and block index, or store an `fBlockWidth` vector), build helper volumes
  named `SiPMmapVolU_<iB>`/`SiPMmapVolV_<iB>` with the block's channel count, store
  `fNSiPMChan[iB]`, `fNSiPMs[iB]`.
- **`SiPMmapping()`** — loop `iB` in `0..fnB-1`: call `SiPMOverlap(iB)`. Enumerate
  the block's fibres via the **uniquely named** assembly `MTC_layer_<iB>`
  (`FindVolumeFast` is unambiguous on it) → `MTC_scifi_U_0`/`_V_0` →
  `MTC_epoxyMat_0` → `FiberVol_<n>`. For each fibre, call `GetPosition` with the
  effective global id `fibre_local_number + repStation*1e6`, `repStation =
  iB*fnLayPerBlock`, so navigation lands on this block's geometry and returns the
  block-width x. Compute overlaps against the `SiPMmapVol*_<iB>` channels and fill
  `fibresSiPM_*[iB]`, `siPMFibres_*[iB]`, `SiPMPos_*[iB]`. (Fibre transverse
  positions are identical for every station within a block, so one representative
  station per block suffices.)
- **`GetPosition` / `GetLocalPos` / `GetSiPMPosition`** — compute `iB =
  BlockForLayer(station)`; build the path with `MTC_layer_<iB>_<station>`; use
  `fFiberLength[iB]` and `SiPMPos_*[iB]`.
- **Geometry loop** — store the per-block width (`fBlockWidth[iB]`) and
  `fFiberLength[iB]` at build time so the mapping stage doesn't re-derive them; clamp
  `iB` via `BlockForLayer`.

### Getters + Python

- Getters take a block index, e.g. `GetSiPMmapU(Int_t iB)`, `GetFibresMapU(iB)`,
  `GetSiPMPos_U(iB)`, plus `GetNBlocks()` and `GetNLayersPerBlock()`.
- **`SciFiMapping.py`** — build one map set per block (loop `iB`), storing
  `sipm_to_fibre_map_U[iB]` etc. `make_mapping()` calls `SiPMmapping()` once, then
  reads the per-block getters.
- **`python/detectors/MTCDetector.py`** — decode `layer = (det_id // 1e6) % 100`,
  `segment = layer // nLayersPerBlock`, select the block's map before the existing
  fibre→SiPM lookup. Global-channel construction is unchanged (already keyed on
  `det_id // 1e6`, which carries the layer/block).

## Out of scope

- YAML-driven segment config (decision: sizes hardcoded in C++).
- ACTS `MTCBuilder` reco geometry (`python/ACTSReco.py` — marked "to be updated").
- Any change to the number of blocks (fixed at 3) or the 15/15/15 split.

## Verification

1. `python python/experimental/check_overlaps.py` on the built geometry → no new
   overlaps/extrusions.
2. A short `run_simScript.py` run producing `MTCDetPoint`s → confirm hits appear in
   all three blocks (layers 0–14, 15–29, 30–44).
3. Assert per-block SiPM channel counts are 400/500/600 and that digitised
   `MTCDetHit` global channels decode to the expected block.
4. Event display renders the stepped geometry.
