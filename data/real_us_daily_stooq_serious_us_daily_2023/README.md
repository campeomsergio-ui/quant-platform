# real_us_daily_stooq_serious_us_daily_2023

- Source: Stooq public daily CSV endpoint
- Construction: filtered the broader large-cap US sample to symbols with complete benchmark-aligned daily history across the full window
- Date range requested: 2023-01-03 to 2025-03-31
- Benchmark used: SPY daily returns derived from Stooq close-to-close returns

Improvements vs prior smoke datasets:
- materially longer sample window than the 2024-only smoke datasets
- stronger symbol consistency through time
- benchmark-aligned full-history filtering
- explicit reproducible construction rule

Remaining limitations:
- convenience universe, not survivorship-free
- static hand-entered metadata
- empty delistings / corporate actions
- benchmark is ETF proxy, not total-return index
