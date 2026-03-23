# real_us_daily_stooq_smoke

- Source: Stooq public daily CSV endpoint
- Bars path: `data/real_us_daily_stooq_smoke/bars.csv`
- Metadata path: `data/real_us_daily_stooq_smoke/metadata.csv`
- Benchmark path: `data/real_us_daily_stooq_smoke/benchmark.csv`
- Symbol mapping path: `data/real_us_daily_stooq_smoke/symbol_mapping.csv`
- Delistings path: `data/real_us_daily_stooq_smoke/delistings.csv`
- Corporate actions path: `data/real_us_daily_stooq_smoke/corporate_actions.csv`
- Benchmark used: SPY daily returns derived from Stooq close-to-close returns
- Date range: 2025-01-01 to 2025-03-31
- Approximate symbol count: 12

Quality limitations:
- small convenience universe, not survivorship-free
- static hand-entered metadata
- empty delistings / corporate actions
- benchmark is ETF proxy, not total-return index
