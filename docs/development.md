# Development

## Layout

```
src/tradebot/
  core/        ops (NumPy/torch time-series primitives), OHLC container, registry + plugin loader, models, timeframes
  indicators/  BaseIndicator and the built-in indicators
  algorithms/  BaseAlgorithm, built-in strategies, the combo builder, the pool
  analysis/    loop-based pattern detection (Elliott waves), halving schedule, pairs statistics, Monte Carlo
  data/        providers (ccxt exchanges, yfinance, Alpha Vantage, FMP), MarketDataEngine, disk cache, currencies
  brokers/     BaseBroker; ccxt brokers (Binance, KuCoin, Pionex); semi-automated Scalable / Trade Republic; paper
  backtest/    vectorized engine, metrics, baselines, slippage, report export, pairs backtest
  optimize/    backend selection (CUDA, ROCm, CPU), genetic algorithm, particle swarm, comparison, walk-forward, meta-search
  evolve/      genetic programming of whole strategies: typed primitives, trees, search, plugin export
  live/        TradingBot, pairs bot, risk guards
  portfolio.py, screener.py, scanner.py, scripts.py, news.py, notify.py, config.py, glossary.py, legal.py
  cli/         typer + rich + plotext
  gui/         PyQt6 + pyqtgraph
examples/plugins/   example plugins          docs/   this documentation
tests/              pytest suite             scripts/   benchmarks and the docs check
packaging/          PyInstaller spec for the binaries
```

## Tests

```sh
uv sync --extra gui
uv run pytest
uv run pytest --slow                   # also the minutes-long tests (CI runs them on PRs labelled `slow tests`)
QT_QPA_PLATFORM=offscreen uv run pytest tests/test_gui.py    # GUI tests without a display
uv run python scripts/bench_chart.py   # chart redraw and pan times for 500 / 5,000 / 50,000 candles
uv run python scripts/bench_chart.py --onscreen   # the same on the real display; --opengl with GPU rendering
```

The tests cover indicator accuracy against pandas reference implementations, backtest math on hand-checked series,
NumPy/torch-CPU/CUDA parity of batched fitness, the broker abstractions, data-engine fallback, plugin loading (including
the worked example), the live bot and its risk guards, portfolio and dividend metrics, the screener, news parsing, and
offscreen rendering of every GUI theme. Torch-only tests skip themselves without torch. Tests never touch your config,
cache or the network: they use private temporary folders.

## Continuous integration

* `tests.yml`: runs on every pull request and on pushes to `main`. The Linux job (`pytest`, Python 3.14, offscreen Qt) is a
  **required check** for merging; `pytest-os` runs the same suite on Windows and macOS. Slow tests run only on PRs
  labelled `slow tests`.
* `docs.yml`: checks the documentation, see below.
* `tag.yml`: tags every merge into `main` with the next version.
* `release.yml`: after the tests pass, builds the standalone binaries with PyInstaller (`packaging/tradebot.spec`) and
  publishes them on the GitHub Release of a `v*` tag.

## Branches, pull requests and releases

* Never commit to `main`. Work on a branch (`feature/<issue>-<slug>`, `docs/<issue>-<slug>`), open a pull request that
  says `Closes #<issue>`, and merge it after the checks pass.
* Versions follow [semantic versioning](https://semver.org). **Every merge into `main` is tagged automatically**, and the
  merged PR's label picks the bump:

| Label on the PR | Meaning | Example |
|---|---|---|
| `release: major` | a breaking change | v1.4.2 to v2.0.0 |
| `release: minor` | a new feature | v1.4.2 to v1.5.0 |
| none | a bugfix, a documentation change or purely cosmetic | v1.4.2 to v1.4.3 |

* Don't add third-party tools or broker assets to the repo or the binaries ([Extending](extending.md#rules-for-contributions)).

## Documentation

The pages in `docs/` are plain Markdown that a new user can read in order; start at [the index](index.md). The README is a
short front page that links here. Rules of thumb: short pages, task-oriented, copy-paste examples, and every claim checked
against the code.

```sh
python scripts/check_docs.py           # what CI runs
python scripts/check_docs.py --write   # regenerate docs/cli-reference.md after changing a command or option
```

The check fails when

* a relative link or `#anchor` in the README or `docs/` points nowhere,
* `docs/cli-reference.md` differs from what the Typer commands generate (so a new command or option can't go undocumented),
* `docs/cli.md` doesn't mention a command, or
* `docs/legal.md` no longer contains the exact disclaimer text of `src/tradebot/legal.py` (which the About dialog, the
  `--version` output and the live-trading notice use).

When you add a GUI page or a feature, update its page in the same pull request. Screenshots are optional; if you add any,
make them with the offscreen render (`QT_QPA_PLATFORM=offscreen`) so they can be regenerated.
