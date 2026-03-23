import json
from pathlib import Path
from quant_platform.data_access import LocalTableDataAdapter
from quant_platform.research import BaselineResearchConfig, build_baseline_signals, _build_market_data
from quant_platform.backtest.engine import BacktestConfig, simulate_tranches, _aggregate_book, _constraint_config
from quant_platform.portfolio import enforce_constraints
from quant_platform.data_contracts import PortfolioWeights

bundle = LocalTableDataAdapter('data/real_us_daily_stooq_smoke', preferred_format='csv').load_bundle()
signals = build_baseline_signals(bundle, BaselineResearchConfig())
market_data = _build_market_data(bundle)
config = BacktestConfig(holding_period_days=5, execution_time='open_t_plus_1', min_longs=1, min_shorts=1, max_sector_weight=1.0)
tranche_state = simulate_tranches(signals, holding_period=config.holding_period_days)
intended_book = _aggregate_book(tranche_state)
execution_dates = list(intended_book.index)
inspected=[]
for i, signal_date in enumerate(execution_dates[:12]):
    rec={'i': i, 'signal_date': str(signal_date.date())}
    sig = signals.get(signal_date)
    rec['signal_values_exist'] = bool(sig is not None and len(sig) > 0)
    rec['ranked_candidates_exist'] = bool(sig is not None and int((sig.fillna(0)!=0).sum()) > 0)
    rec['signal_nonzero_count']=0 if sig is None else int((sig.fillna(0)!=0).sum())
    rec['signal_longs']=0 if sig is None else int((sig>0).sum())
    rec['signal_shorts']=0 if sig is None else int((sig<0).sum())
    if i + 1 >= len(execution_dates):
        rec['missing_next_day_execution'] = True
        rec['engine_discarded'] = True
        rec['engine_discard_reason']='no_next_execution_date'
        inspected.append(rec)
        continue
    execution_date = execution_dates[i+1]
    rec['execution_date']=str(execution_date.date())
    target = intended_book.loc[signal_date].copy()
    rec['target_weights_nonzero'] = bool(int((target.fillna(0)!=0).sum()) > 0)
    rec['target_nonzero_count']=int((target.fillna(0)!=0).sum())
    rec['target_longs']=int((target>0).sum())
    rec['target_shorts']=int((target<0).sum())
    md = market_data.xs(execution_date).reindex(target.index)
    rec['missing_next_day_open_count']=int(md['open'].isna().sum()) if 'open' in md.columns else None
    rec['execution_eligibility_blocks_trade'] = bool(rec['missing_next_day_open_count'] and rec['missing_next_day_open_count'] > 0)
    constrained = enforce_constraints(PortfolioWeights(target, execution_date), _constraint_config(config), md)
    cw = constrained.weights.weights
    rec['constraint_modifies_book'] = bool(len(constrained.events) > 0)
    rec['constraint_rejects_book'] = bool(constrained.failed)
    rec['constrained_nonzero_count']=int((cw.fillna(0)!=0).sum())
    rec['constrained_longs']=int((cw>0).sum())
    rec['constrained_shorts']=int((cw<0).sum())
    rec['breadth_failure'] = any('insufficient longs' in r or 'insufficient shorts' in r for r in constrained.failure_reasons)
    rec['bucket_size_failure'] = bool(rec['ranked_candidates_exist'] and (rec['signal_longs'] == 0 or rec['signal_shorts'] == 0))
    rec['beta_failure'] = any('beta target unsatisfied' in r for r in constrained.failure_reasons)
    rec['sector_failure'] = any('sector cap unsatisfied' in r for r in constrained.failure_reasons)
    rec['liquidity_failure'] = any('missing liquidity' in r for r in constrained.failure_reasons)
    rec['failure_reasons']=list(constrained.failure_reasons)
    rec['constraint_events']=[{'constraint': e.constraint, 'action': e.action, 'before': e.before, 'after': e.after, 'reason': e.reason} for e in constrained.events]
    rec['engine_discarded'] = bool(constrained.failed)
    rec['engine_discard_reason']='constraint_failure' if constrained.failed else 'would_execute'
    inspected.append(rec)
Path('outputs/exploratory_real_smoke/debug_boundary_baseline.json').write_text(json.dumps({
    'tranche_state_shape': list(tranche_state.shape),
    'intended_book_rows': int(len(intended_book)),
    'first_nonzero_signal_date': next((str(d.date()) for d, s in signals.items() if len(s) and int((s.fillna(0)!=0).sum())>0), None),
    'inspected_dates': inspected,
}, indent=2, default=str))
print(json.dumps(inspected, indent=2))
