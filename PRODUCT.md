# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

<!-- Recorded as `web` because the schema has no desktop value. tradebot's UI is a PyQt6 + pyqtgraph desktop app (`tradebot gui`), not a web app; there is also a terminal CLI. -->

## Users

The open-source community broadly: hobbyist and semi-professional algorithmic traders and developers who download the release binaries or install from source. They work at a desk, mixing research (data, charts, backtests, optimization) with running strategies on their own accounts.

## Product Purpose

An extensible algorithmic trading platform for crypto and stocks. It unifies broker adapters and multi-source OHLC data, with pluggable indicators and strategies, a vectorized backtester, a GPU-accelerated genetic optimizer, a live bot with risk guards, a PyQt6 desktop app and a terminal CLI. Success means a user can take a strategy from idea to a validated, running bot without leaving the tool.

## Positioning

Strategy evolution and optimization is the headline: genetic optimization of parameters and genetic programming of whole strategies, GPU-accelerated, with walk-forward and multi-symbol validation. Free, GPL-3.0-or-later software, not tied to a single broker.

## Operating Context

- One binary is both CLI and GUI; builds for Linux, Windows and macOS (unsigned, CPU-only), or a source install with `gui`, `gpu` and `traderepublic` extras.
- Supported services: Binance, KuCoin, Pionex, Scalable Capital (via the external `sc` CLI), Trade Republic (via `pytr`), Yahoo data. None are affiliated with tradebot.
- Live trading is opt-in (`--live`), with dry-run and sandbox/testnet modes first.

## Capabilities and Constraints

- Desktop GUI: PyQt6, pyqtgraph charts (drawings, indicators), scanner, screener, news, portfolio, backtest reports (Monte Carlo, expectancy, heatmaps, export), optimizer and evolve panels, settings, About.
- Live bot guards: daily loss, drawdown, exposure and order caps, cool-down, kill switch.
- Not financial advice and no warranty; the first live run shows a risk notice the user must accept. The GUI's About button shows the disclaimer.
- Third-party tools and trademarks are named only to say what the product works with; they are never bundled or implied as endorsing.
- Visual world and UI conventions are not yet defined; no DESIGN.md exists.

## Brand Commitments

Name: tradebot (lowercase). No logo or brand assets recorded.

## Evidence on Hand

README.md, `docs/scanner.md`, `TODO.md`, tagged releases on GitHub (daMarxrider/trading_bot_3), `benchmarks/`. No testimonials, user counts or performance claims exist, and none may be invented. Backtest and optimizer results never imply future returns.

## Product Principles

1. Trust over flash: numbers, risk state and live/dry-run/sandbox mode are always unambiguous.
2. Research-to-live in one place: each step (data, backtest, optimize, validate, run) leads to the next.
3. Dense by default, legible always: the users are technical and long sessions are normal.
4. Never imply guaranteed returns; show robustness (out-of-sample, Monte Carlo, drawdown) next to headline gains.
5. Neutral toward brokers: no endorsement, no partnership claims.

## Accessibility & Inclusion

No product-specific requirement established. Open decision.
