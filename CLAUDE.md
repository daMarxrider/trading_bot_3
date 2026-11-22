# tradebot

## GUI work (`src/tradebot/gui/`)

- Before changing anything the user sees in the desktop app (layout, colours, fonts, QSS in `theme.py`, dialogs, icons, copy), invoke the `impeccable` skill. The design detector hook does not scan Python files, so this is not automatic.
- `PRODUCT.md` holds the product context. The Graphite redesign mockups are in `mockups/` (open `mockups/index.html`); issue #191 tracks implementing them. Match the mockup for the screen you touch, in both Graphite and Paper.
- The app is PyQt6, not web. The skill's `shot` and `live` tools don't apply, so compare the running app against the mockup instead.
- Up/down figures always pair colour with a drawn triangle. Text must fit at 1440x900 and at the minimum window size.
