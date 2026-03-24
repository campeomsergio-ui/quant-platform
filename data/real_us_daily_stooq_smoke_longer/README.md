# real_us_daily_stooq_smoke_longer

- Source: Stooq public daily CSV endpoint
- Bars path: `data/real_us_daily_stooq_smoke_longer/bars.csv`
- Metadata path: `data/real_us_daily_stooq_smoke_longer/metadata.csv`
- Benchmark path: `data/real_us_daily_stooq_smoke_longer/benchmark.csv`
- Symbol mapping path: `data/real_us_daily_stooq_smoke_longer/symbol_mapping.csv`
- Delistings path: `data/real_us_daily_stooq_smoke_longer/delistings.csv`
- Corporate actions path: `data/real_us_daily_stooq_smoke_longer/corporate_actions.csv`
- Benchmark used: SPY daily returns derived from Stooq close-to-close returns
- Date range requested: 2024-01-01 to 2025-03-31
- Approximate symbol count: 12

Quality limitations:
- small convenience universe, not survivorship-free
- static hand-entered metadata
- empty delistings / corporate actions
- benchmark is ETF proxy, not total-return index
