# real_us_daily_stooq_higher_truth_2023

- Source: Stooq public daily CSV endpoint
- Construction: start from a broader large-cap US candidate list and retain only symbols whose daily history exactly matches the SPY benchmark date set over the full window
- Date range requested: 2023-01-03 to 2025-03-31
- Symbol count retained: 31
- Benchmark used: SPY daily returns derived from Stooq close-to-close returns

Improvements vs real_us_daily_stooq_serious_us_daily_2023:
- materially broader stable universe if more names survive the full-history filter
- stronger explicit inclusion rule from a broader candidate set
- rejected names recorded for transparency
- same full-history benchmark-aligned continuity rule preserved

Remaining limitations:
- convenience universe, not survivorship-free
- static hand-entered metadata
- empty delistings / corporate actions
- benchmark is ETF proxy, not total-return index
