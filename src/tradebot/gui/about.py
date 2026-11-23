# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The About dialog (#58, #191): version, the disclaimers, licence and source links, one Close button."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from tradebot import __version__
from tradebot.legal import NO_WARRANTY, NOT_AFFILIATED

SOURCE_URL = "https://github.com/daMarxrider/trading_bot_3"
LICENCE_URL = "https://www.gnu.org/licenses/gpl-3.0.html"


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About tradebot")
        self.setMinimumWidth(520)
        title = QLabel("tradebot", objectName="Title")
        version = QLabel(f"v{__version__}", objectName="Muted")
        head = QHBoxLayout()
        head.addWidget(title)
        head.addWidget(version)
        head.addStretch()
        links = QLabel(f"<a href='{LICENCE_URL}'>Licence: GPL-3.0-or-later</a> &nbsp;&nbsp; "
                       f"<a href='{SOURCE_URL}'>Source code</a> &nbsp;&nbsp; "
                       "Licences of the bundled libraries: THIRD_PARTY_NOTICES.txt next to the program",
                       textFormat=Qt.TextFormat.RichText, openExternalLinks=True, wordWrap=True)
        close = QPushButton("Close", objectName="Primary")
        close.clicked.connect(self.accept)
        foot = QHBoxLayout()
        foot.addStretch()
        foot.addWidget(close)
        rule = QFrame(frameShape=QFrame.Shape.HLine)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 16)
        lay.setSpacing(12)
        lay.addLayout(head)
        for text in (NOT_AFFILIATED, NO_WARRANTY):
            lay.addWidget(QLabel(text, objectName="Muted", wordWrap=True))
        lay.addWidget(links)
        lay.addWidget(rule)
        lay.addLayout(foot)
