# quant-platform implementation notes

## Phase status
- Phase 1: implemented (frozen JSONs, exact baseline file set, module mapping artifacts)
- Phase 2: implemented baseline + conservative fixes; some components remain scaffold_only
- Phase 3: scaffold_only for paper-trading control plane; live trading intentionally placeholder
- Speculative research extensions: scaffold_only / speculative_extension

## JSON-to-module mapping
- STRATEGY_SPEC_JSON -> strategy_spec.py, universe.py, signals/mean_reversion.py, risk.py, costs.py, portfolio.py, backtest/engine.py
- EXPERIMENT_PLAN_JSON -> experiment_plan.py, validation/walk_forward.py, validation/multiple_testing.py, validation/overfitting.py, metrics.py, stopping.py
- REPO_BLUEPRINT_JSON -> exact baseline file tree under src/ and tests/; public shims preserved at src/*.py

## Ambiguities and conservative fixes
- Residual model formula was underspecified; conservative baseline uses sector-demeaned 5-day returns and labels richer regression residualization as future extension.
- Universe churn handling was underspecified; conservative default is monthly reconstitution with daily hard-failure handling, scaffolded in universe builder config.
- Impact function shape was underspecified; baseline keeps explicit simple functional form consistent with the frozen cost text.
- Borrow availability proxy source was unspecified; treated as required input, not synthesized.
- White-style test could be Reality Check or SPA-style equivalent; baseline implements White-style RC and notes SPA as extension.
- PBO procedure was specified conceptually but not operationally; baseline uses rank-based estimate with explicit placeholder for richer CSCV combinatorics.
- Final locked test-touch enforcement requires persistent registry; scaffolded in experiment_registry.py, not yet wired to durable state beyond explicit IO boundaries.

## Safety posture
- Signal formed at close t
- Orders generated after close t
- Baseline execution at open t+1
- Same-close mode is diagnostic-only and disabled by default
- Internal netting across tranches before turnover/costs is required baseline behavior
- Paper trading first; live trading remains blocked behind explicit future safety flags
