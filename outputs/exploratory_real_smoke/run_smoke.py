import json
from pathlib import Path
from quant_platform.data_access import LocalTableDataAdapter
from quant_platform.research import BaselineResearchConfig, ResidualMomentumCycleConfig, run_baseline_research, run_residual_momentum_cycle

out = Path('outputs/exploratory_real_smoke')
out.mkdir(parents=True, exist_ok=True)

bundle = LocalTableDataAdapter('data/real_us_daily_stooq_smoke', preferred_format='csv').load_bundle()
registry_path = Path('registry/experiments.json')
registry = {'experiments': {}}
if registry_path.exists():
    registry = json.loads(registry_path.read_text())

baseline = run_baseline_research(bundle, BaselineResearchConfig())
(out / 'baseline_result.json').write_text(json.dumps({'metrics': baseline.metrics, 'diagnostics': baseline.diagnostics}, indent=2, default=str))

cycle = run_residual_momentum_cycle(
    bundle,
    BaselineResearchConfig(),
    ResidualMomentumCycleConfig(
        lookbacks=(20,),
        skip_windows=(5,),
        residual_models=('industry_beta_log_mcap',),
        seed=0,
        stage='development',
    ),
    registry=registry,
)
(registry_path).parent.mkdir(parents=True, exist_ok=True)
registry_path.write_text(json.dumps(cycle.registry, indent=2, default=str))
(out / 'residual_momentum_cycle.json').write_text(json.dumps({
    'experiment_id': cycle.experiment_id,
    'candidate_family': cycle.candidate_family,
    'multiple_testing': cycle.multiple_testing,
    'overfitting': cycle.overfitting,
    'comparison': cycle.comparison,
}, indent=2, default=str))
print(json.dumps({
    'baseline_metrics': baseline.metrics,
    'baseline_history_quality': baseline.diagnostics.get('history_quality'),
    'residual_best_candidate': cycle.comparison['best_residual_momentum_candidate'].get('candidate_id'),
    'residual_best_metrics': cycle.comparison['best_residual_momentum_candidate'].get('metrics', {}),
}, indent=2, default=str))
