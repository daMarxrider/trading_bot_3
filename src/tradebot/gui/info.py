# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Symbol info page of the chart sidebar (#165): valuation, margins, growth, dividend yield and the earnings,
split and ex-dividend dates of the charted symbol. The chart fetches the data (off the GUI thread, cached on
disk by :mod:`tradebot.fundamentals`) and hands it over with :meth:`InfoPanel.show_snapshot`."""

from __future__ import annotations

import html
from datetime import date

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QListWidget, QListWidgetItem, QScrollArea, QVBoxLayout, QWidget

from tradebot import fundamentals as fund
from tradebot.gui.common import up_down_mark
from tradebot.gui.theme import ThemeManager
from tradebot.gui.touch import enable_kinetic_scroll

#: letter beside the marker on the chart, so the kind never depends on colour alone
LETTERS = {"earnings": "E", "split": "S", "dividend": "D"}
SIGNED = ("revenue_growth", "earnings_growth")  # figures that carry a direction: colour plus a triangle


def event_colour(theme, kind: str) -> str:
    return {"earnings": theme.accent, "split": theme.text, "dividend": theme.muted}[kind]


class InfoPanel(QWidget):
    def __init__(self, themes: ThemeManager, parent=None):
        super().__init__(parent)
        self.themes = themes
        self.symbol = ""
        self.snap: fund.Snapshot | None = None
        self.header = QLabel(objectName="Muted", wordWrap=True)
        self.facts = QLabel(wordWrap=True, textFormat=Qt.TextFormat.RichText)
        self.facts.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.note = QLabel(objectName="Muted", wordWrap=True)
        self.events_title = QLabel("Events", objectName="Muted")
        self.events = QListWidget(objectName="SidebarNews")
        self.events.setSpacing(1)
        self.events.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        enable_kinetic_scroll(self.events)
        body = QWidget(objectName="InfoBody")
        body.setStyleSheet("QWidget#InfoBody { background: transparent; }")  # the panel's own fill shows through
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 4, 0)
        col.addWidget(self.facts)
        col.addWidget(self.note)
        col.addWidget(self.events_title)
        col.addWidget(self.events, 1)
        scroll = QScrollArea(widgetResizable=True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        scroll.viewport().setAutoFillBackground(False)
        scroll.setWidget(body)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.header)
        lay.addWidget(scroll, 1)
        self.events.setMinimumHeight(180)
        themes.changed.connect(lambda _: self._render())
        self.show_message("Load a chart to see its fundamentals.")

    def set_symbol(self, symbol: str, name: str | None = None):
        self.header.setText(html.escape(f"{name} ({symbol})" if name and name != symbol else symbol))
        if symbol != self.symbol:
            self.symbol, self.snap = symbol, None
            self.show_message(f"Loading {symbol} …" if fund.supported(symbol)
                              else "Fundamentals exist for stocks only, not for crypto, pairs or indices.")

    def show_message(self, text: str):
        self.facts.setText("")
        self.events.clear()
        self.events_title.setVisible(False)
        self.events.setVisible(False)
        self.note.setText(text)
        self.note.setVisible(bool(text))

    def show_snapshot(self, snap: fund.Snapshot):
        if snap.symbol != self.symbol:
            return  # the chart moved on to another symbol meanwhile
        self.snap = snap
        self._render()

    def _render(self):
        snap = self.snap
        if snap is None:
            return
        if snap.error and not snap.fundamentals.has_data() and not snap.events:
            self.show_message(f"No fundamentals for {snap.symbol} right now.\n{snap.error}")
            return
        t = self.themes.current
        f = snap.fundamentals
        rows = []
        for name, (label, _) in fund.FIELDS.items():
            v = getattr(f, name)
            text = html.escape(fund.format_value(name, v))
            if name in SIGNED and v is not None and v != 0:
                colour = t.up if v > 0 else t.down
                text = f'<span style="color:{colour}">{up_down_mark(v)} {html.escape(fund.format_value(name, abs(v)))}</span>'
            rows.append(f'<tr><td style="color:{t.muted}; padding-right:14px">{label}</td>'
                        f'<td align="right">{text}</td></tr>')
        if f.sector:
            rows.insert(0, f'<tr><td colspan="2" style="color:{t.muted}; padding-bottom:6px">{html.escape(f.sector)}</td></tr>')
        self.facts.setText(f'<table width="100%" cellspacing="0" cellpadding="3">{"".join(rows)}</table>')
        self.note.setVisible(False)
        self._fill_events(snap, t)

    def _fill_events(self, snap, t):
        today = date.today()
        self.events.clear()
        upcoming = sorted(e for e in snap.events if e.day >= today)
        past = sorted((e for e in snap.events if e.day < today), reverse=True)[:10]
        for e in [*upcoming, *past]:
            soon = e.day >= today
            item = QListWidgetItem(f"{LETTERS[e.kind]}  {e.day:%Y-%m-%d}   {e.label}" + ("   · upcoming" if soon else ""))
            item.setToolTip(f"{e.label}, {e.day:%A %d %B %Y}")
            if not soon:
                item.setForeground(qcolor_of(t.muted))
            self.events.addItem(item)
        shown = self.events.count() > 0
        self.events_title.setVisible(shown)
        self.events.setVisible(shown)


def qcolor_of(colour: str):
    from tradebot.gui.theme import qcolor

    return qcolor(colour)
