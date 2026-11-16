# Getting started

## Install

From source (Python 3.11 or newer):

```sh
uv sync --extra gui --extra gpu      # or: pip install -e '.[gui,gpu]'
uv run tradebot --help
uv run tradebot gui
```

Extras: `gui` (PyQt6, pyqtgraph), `gpu` (PyTorch), `traderepublic` (pytr). The core (CLI, backtests, the CPU
optimizer) needs none of them.

**Third-party tools are not included.** tradebot works with some external tools but never ships them, neither in
the package nor in the binaries. Install one yourself; its own licence and terms apply:

| Tool | Needed for | Get it |
|---|---|---|
| `pytr` | Trade Republic login | `uv sync --extra traderepublic` (source install only; the binaries don't contain it) |
| `sc` | Scalable Capital (official CLI) | [Scalable CLI releases](https://github.com/ScalableCapital/scalable-cli/releases); set its path in [Settings](settings.md) if it isn't on `PATH` |

## The binary

Every release on the [Releases page](https://github.com/daMarxrider/trading_bot_3/releases) has standalone builds
for Linux, Windows and macOS that need no Python. One binary is both the CLI and the desktop app (`tradebot gui`).
The builds are CPU-only (see [GPU acceleration](#gpu-acceleration)) and **unsigned**. Each archive has a `.sha256`
checksum next to it and contains `LICENSE` plus `THIRD_PARTY_NOTICES.txt`, the licences of all bundled libraries.

**Linux (x86_64).** Built on Ubuntu 22.04, so it runs on that and newer distributions (glibc 2.35+).

```sh
sha256sum -c tradebot-v1.0.0-linux-x86_64.tar.gz.sha256
tar -xzf tradebot-v1.0.0-linux-x86_64.tar.gz
./tradebot/tradebot --help
./tradebot/tradebot gui
```

The GUI needs a few system libraries that desktop installs usually already have. On Debian/Ubuntu:

```sh
sudo apt install libegl1 libgl1 libxkbcommon0 libxkbcommon-x11-0 libfontconfig1 libdbus-1-3 libxcb-cursor0
```

**Windows (x86_64, Windows 10/11).** Unzip the archive, then run `tradebot-gui.exe` for the desktop app (no
console window) or `tradebot.exe` in a terminal for the CLI:

```powershell
(Get-FileHash tradebot-v1.0.0-windows-x86_64.zip).Hash   # compare with the .sha256 file
.\tradebot\tradebot.exe --help
```

Because the binaries are unsigned, SmartScreen warns on first start ("Windows protected your PC"). Click
**More info, then Run anyway**. Config, plugins and themes live in `%USERPROFILE%\.config\tradebot\`, the same
layout as on Linux.

**macOS (Apple Silicon `arm64`, Intel `x86_64`).** Unzip the archive and move `tradebot.app` to Applications.
Opening the app starts the desktop app; the CLI is inside the bundle:

```sh
shasum -a 256 -c tradebot-v1.0.0-macos-arm64.zip.sha256
/Applications/tradebot.app/Contents/MacOS/tradebot --help
sudo ln -s /Applications/tradebot.app/Contents/MacOS/tradebot /usr/local/bin/tradebot   # optional
```

The app is not signed or notarized, so Gatekeeper blocks the first start ("Apple could not verify ..."). Try to
open it once, then go to **System Settings, Privacy & Security** and click **Open Anyway**. Alternatively remove the
quarantine flag: `xattr -dr com.apple.quarantine /Applications/tradebot.app`.

**Plugins in the binary.** Plugin files from `~/.config/tradebot/plugins/` (or `$TRADEBOT_PLUGIN_DIR`) load as
usual and can import any `tradebot` module and the bundled libraries. Entry-point plugins from installed packages
are not found by the binary, because it cannot see your Python's packages; use the plugin directory, or install
tradebot from source.

## GPU acceleration

The release binaries are CPU-only: they exclude PyTorch and the optimizer falls back to a multi-threaded NumPy
pool. PyTorch with CUDA/ROCm is several GB and depends on the GPU vendor, driver and platform, so the binaries stay
small and the GPU version runs from source.

**When it is worth it.** Large populations (thousands of parameter sets) and long histories. For small runs the
NumPy pool is as fast or faster. Algorithms marked `period_batched` (`macd_cross`, `sma_cross`) evaluate a whole
population in one GPU call even when their window lengths differ; for the other integer-only algorithms `auto`
picks NumPy, which is faster for them.

**What you need.** Python 3.11+, [`uv`](https://docs.astral.sh/uv/) (or pip), git, disk space for torch, and a
supported GPU: NVIDIA with a recent driver (CUDA), or AMD with ROCm (Linux only, not verified yet, see #1). macOS has
neither: Apple GPUs (MPS) are not used (the backend order is CUDA, ROCm, CPU), so use the binary there.

```sh
git clone https://github.com/daMarxrider/trading_bot_3 && cd trading_bot_3
```

* **Linux + NVIDIA:** `uv sync --extra gui --extra gpu` (or `pip install -e '.[gui,gpu]'`). The default PyPI torch
  wheel for Linux includes CUDA.
* **Windows + NVIDIA:** the plain `gpu` extra gets a CPU-only torch from PyPI. After the sync, install the CUDA
  build from the PyTorch index, for example
  `uv pip install torch --index-url https://download.pytorch.org/whl/cu124` (check pytorch.org for the current CUDA
  tag). Run later commands with `uv run --no-sync` so uv does not swap it back. Not tested.
* **Linux + AMD:** the `gpu` extra installs the CUDA build, under which an AMD card goes unnoticed. After the sync,
  install a ROCm build: `uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.2`, then use
  `uv run --no-sync`. Not tested.

**Verify.** `uv run tradebot device` must print `CUDA: <GPU name>` or `ROCM: ...`, not
`NumPy ... (no CUDA/ROCm GPU found)`. The [Backtest lab](backtest-lab.md) shows the same under *Compute backend*.

**Run it.** `uv run tradebot gui` or `uv run tradebot optimize ...`. Optionally `uv tool install '.[gui,gpu]'` puts
`tradebot` on `PATH` like the binary. Settings, cache and plugins are shared with the binary (same config dir).
`--backend cuda` / `--backend rocm` (alias `amd`) require that vendor and fail with an error instead of falling
back; `--backend gpu` takes either; `--backend numpy` (or `cpu`) forces the CPU pool.

**Troubleshooting.**

* `tradebot device` reports no GPU: check for a CPU-only torch wheel
  (`uv run python -c "import torch; print(torch.version.cuda, torch.version.hip)"` prints `None None` for one), the
  driver version (`nvidia-smi`), and `CUDA_VISIBLE_DEVICES`.
* Out of GPU memory: lower the population.

## First run

1. `tradebot gui` opens the desktop app. The [Chart](chart.md) loads the first symbol of your first watchlist
   (default list: AAPL, MSFT, NVDA, SAP.DE, BTC/USDT, ETH/USDT, SOL/USDT).
2. Add indicators with **Indicators**, pick a strategy in the signal box, and try the
   [Backtest lab](backtest-lab.md).
3. Or in a terminal:

```sh
tradebot list algorithms
tradebot chart BTC/USDT --tf 4h --bars 200 -i bbands -i rsi --algo rsi_threshold
tradebot backtest AAPL macd_cross --bars 2000
```

No account or API key is needed for market data of stocks (Yahoo) and crypto (Binance, KuCoin, Kraken public data).
Keys are only needed for [brokers](live-trading.md) and the optional data providers.

## Configuration in one minute

Two files in the config folder (`~/.config/tradebot/`, see [Reference](reference.md) for other systems):

* `config.toml`: what you edit: theme, watchlists, broker keys, tool paths, slippage default, cache size,
  notification channels. The [Settings](settings.md) dialog writes part of it for you, without touching your
  comments. `$TRADEBOT_CONFIG` points elsewhere.
* `settings.json`: what the app remembers by itself (toggles, loadouts, drawings, pane sizes). Delete it to reset.

```toml
theme = "glass"

[watchlists]
Main = ["AAPL", "MSFT", "SAP.DE", "BTC/USDT", "ETH/USDT"]

[backtest]
slippage_bps = 3
```

All keys are in [Reference](reference.md).

## The data cache

Candles and symbol lists are kept on disk, so the app and the CLI start without waiting for the network. The cache
is safe to delete. `tradebot cache info` shows where it is and how big; `tradebot cache clear` empties it. How it
refreshes and cleans up is described in [Data providers, timeframes and the cache](data.md).
