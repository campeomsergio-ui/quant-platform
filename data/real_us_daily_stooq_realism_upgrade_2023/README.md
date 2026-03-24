# real_us_daily_stooq_realism_upgrade_2023

- Sources: Stooq public daily CSV endpoint for equities and ^SPX benchmark
- Construction: same 31-name full-history large-cap universe as real_us_daily_stooq_higher_truth_2023, but with benchmark upgraded from SPY ETF returns to ^SPX price-index returns
- Date range requested: 2023-01-03 to 2025-03-31
- Symbol count retained: 31
- Benchmark used: S&P 500 price index (^SPX) close-to-close returns from Stooq

What changed vs real_us_daily_stooq_higher_truth_2023:
- benchmark realism improved from ETF proxy to price-index benchmark
- same stable 31-name universe retained for apples-to-apples sleeve comparison

Remaining limitations:
- convenience universe, not survivorship-free
- static hand-entered metadata
- empty delistings / corporate actions
- benchmark is price index, not total-return index
