# LoRA Scale Investigation Note (2026-02-10)

## Context
- User reported warning: `No LoRA scaling attributes found to modify` while LoRA was loaded.
- Recent upstream fix (PR #320) tightened scaling logic to avoid touching non-LoRA modules.
- Tightened logic reduced false positives but exposed compatibility gaps across PEFT runtime variants.

## Root Cause
- Previous slider implementation used runtime discovery heuristics each time scale changed:
  - module name substring checks (`lora_`)
  - optional `scaling` attribute checks
  - best-effort adapter name discovery
- PEFT APIs vary by version/adapter type (`scaling`, `set_scale`, `scale_layer`, adapter structures).
- Result: behavior was non-deterministic across environments and could become a no-op.

## Design Decision
- Move from *discover-at-scale-time* to *register-at-load-time*.
- Build explicit adapter-to-target references immediately after `load_lora`.
- Apply slider scale only through registered targets for the active adapter.

## Implemented Solution
- Added deterministic registry state:
  - `_lora_adapter_registry`: adapter name -> explicit target list
  - `_lora_active_adapter`: selected adapter
- Added robust adapter-name collection from common PEFT surfaces.
- Added registration pass after LoRA load:
  - `scaling` dict targets (adapter-specific)
  - `set_scale(adapter, value)` targets
  - constrained single-adapter fallbacks for `scale_layer` / scalar `scaling`
- `set_lora_scale` now:
  - uses active adapter only
  - applies scale through registry targets (no broad module scanning)
  - logs deterministic adapter+modified target count
- Added `set_active_lora_adapter(adapter_name)` as forward-compatible API for multi-LoRA controls.

## Why This Is Better
- Deterministic behavior for a given loaded adapter/runtime.
- Clear failure mode when adapter has no registered scale targets.
- Natural path to multi-LoRA:
  - per-adapter target sets already modeled
  - active adapter switching already available

## Known Limitations
- Some adapter/runtime combinations may still expose only coarse `scale_layer` semantics.
- For multi-adapter models without adapter-specific APIs, explicit per-adapter scaling may remain limited.

## Next Steps (for future PRs)
- Add UI controls for selecting active adapter when multiple adapters are loaded.
- Add tests for:
  - repeated slider updates (idempotency)
  - adapter discovery shapes (string/list/mapping)
  - multi-adapter deterministic selection behavior.
