# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""A simple in-app Python editor for Scanner scripts (#115).

``CodeEdit`` is a plain-text editor with line numbers, a current-line highlight, Python syntax colours, Tab
indents by 4 spaces, Enter keeps the indentation (and adds a level after a ``:``) and Ctrl+/ toggles comments.
No autocomplete, linting or debugger. ``ScriptEditor`` wraps it with the save/reload workflow of
:mod:`tradebot.scripts`: Ctrl+S saves and reloads, auto-reload reloads one second after typing stops
(without writing the file), and the last good version stays active when a reload fails.
"""

from __future__ import annotations

import keyword
import re
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QKeySequence,
    QPainter,
    QShortcut,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextCursor,
    QTextFormat,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from tradebot import scripts
from tradebot.gui.theme import Theme, ThemeManager, qcolor

INDENT = "    "
AUTO_RELOAD_MS = 1000
TRUST_NOTE = "Scripts run with the app's permissions."
_KEYWORDS = set(keyword.kwlist) | {"self", "cls"}
_BUILTINS = {"True", "False", "None"}
_NUMBER = re.compile(r"\b(?:0[xX][0-9a-fA-F_]+|\d[\d_]*\.?\d*(?:[eE][+-]?\d+)?j?|\.\d+)\b")
_WORD = re.compile(r"[A-Za-z_]\w*")
_STRING = re.compile(r"""[rbfRBF]{0,2}(?:'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*")""")


class PythonHighlighter(QSyntaxHighlighter):
    """Keywords, strings (also triple-quoted, over several lines), comments and numbers."""

    def __init__(self, document, theme: Theme):
        super().__init__(document)
        self.set_theme(theme, rehighlight=False)

    def set_theme(self, theme: Theme, rehighlight: bool = True):
        def fmt(color: str, bold=False, italic=False) -> QTextCharFormat:
            f = QTextCharFormat()
            f.setForeground(qcolor(color))
            f.setFontWeight(QFont.Weight.Bold if bold else QFont.Weight.Normal)
            f.setFontItalic(italic)
            return f

        self.f_keyword = fmt(theme.accent, bold=True)
        self.f_string = fmt(theme.up)
        self.f_comment = fmt(theme.muted, italic=True)
        self.f_number = fmt(theme.series[2])
        self.f_const = fmt(theme.series[1])
        if rehighlight:
            self.rehighlight()

    def highlightBlock(self, text: str):
        # block state 1/2: inside a ''' or \"\"\" string that started on an earlier line
        state, i = self.previousBlockState(), 0
        spans: list[tuple[int, int, QTextCharFormat]] = []
        if state in (1, 2):
            delim = "'''" if state == 1 else '"""'
            end = text.find(delim)
            if end < 0:
                self.setFormat(0, len(text), self.f_string)
                self.setCurrentBlockState(state)
                return
            spans.append((0, end + 3, self.f_string))
            i = end + 3
        self.setCurrentBlockState(0)
        while i < len(text):
            ch = text[i]
            if ch == "#":
                spans.append((i, len(text) - i, self.f_comment))
                break
            triple = text[i:i + 3]
            if triple in ("'''", '"""') or (ch in "rbfRBF" and text[i + 1:i + 4] in ("'''", '"""')):
                start = i + (0 if triple in ("'''", '"""') else 1)
                delim = text[start:start + 3]
                end = text.find(delim, start + 3)
                if end < 0:
                    spans.append((i, len(text) - i, self.f_string))
                    self.setCurrentBlockState(1 if delim == "'''" else 2)
                    break
                spans.append((i, end + 3 - i, self.f_string))
                i = end + 3
                continue
            m = _STRING.match(text, i)
            if m and (ch in "'\"" or not (i and (text[i - 1].isalnum() or text[i - 1] == "_"))):
                spans.append((i, m.end() - i, self.f_string))
                i = m.end()
                continue
            m = _WORD.match(text, i)
            if m:
                word = m.group()
                if word in _BUILTINS:
                    spans.append((i, len(word), self.f_const))
                elif word in _KEYWORDS:
                    spans.append((i, len(word), self.f_keyword))
                i = m.end()
                continue
            m = _NUMBER.match(text, i)
            if m:
                spans.append((i, m.end() - i, self.f_number))
                i = m.end()
                continue
            i += 1
        for start, length, f in spans:
            self.setFormat(start, length, f)


class _LineNumbers(QWidget):
    def __init__(self, edit: "CodeEdit"):
        super().__init__(edit)
        self.edit = edit

    def sizeHint(self):
        return QSize(self.edit.number_width(), 0)

    def paintEvent(self, event):
        self.edit.paint_numbers(event)


class CodeEdit(QPlainTextEdit):
    def __init__(self, theme: Theme, parent=None):
        super().__init__(parent)
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        font.setPointSize(max(font.pointSize(), 10))
        self.setFont(font)
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(" ") * len(INDENT))
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.theme = theme
        self.error_line: int | None = None
        self.numbers = _LineNumbers(self)
        self.highlighter = PythonHighlighter(self.document(), theme)
        self.blockCountChanged.connect(self._update_margin)
        self.updateRequest.connect(self._scroll_numbers)
        self.cursorPositionChanged.connect(self._highlight_lines)
        QShortcut(QKeySequence("Ctrl+/"), self, activated=self.toggle_comment,
                  context=Qt.ShortcutContext.WidgetShortcut)
        self._update_margin()
        self._highlight_lines()

    def set_theme(self, theme: Theme):
        self.theme = theme
        self.highlighter.set_theme(theme)
        self._highlight_lines()
        self.numbers.update()

    # -- line numbers ---------------------------------------------------------------------------
    def number_width(self) -> int:
        return 14 + self.fontMetrics().horizontalAdvance("9") * max(3, len(str(self.blockCount())))

    def _update_margin(self, *_):
        self.setViewportMargins(self.number_width(), 0, 0, 0)

    def _scroll_numbers(self, rect, dy):
        if dy:
            self.numbers.scroll(0, dy)
        else:
            self.numbers.update(0, rect.y(), self.numbers.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_margin()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        r = self.contentsRect()
        self.numbers.setGeometry(QRect(r.left(), r.top(), self.number_width(), r.height()))

    def paint_numbers(self, event):
        p = QPainter(self.numbers)
        p.fillRect(event.rect(), qcolor(self.theme.input_bg))
        block = self.firstVisibleBlock()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        current = self.textCursor().blockNumber()
        while block.isValid() and top <= event.rect().bottom():
            height = round(self.blockBoundingRect(block).height())
            if block.isVisible() and top + height >= event.rect().top():
                n = block.blockNumber()
                bad = self.error_line is not None and n + 1 == self.error_line
                p.setPen(qcolor(self.theme.down if bad else self.theme.text if n == current else self.theme.muted))
                p.drawText(0, top, self.numbers.width() - 6, self.fontMetrics().height(),
                           Qt.AlignmentFlag.AlignRight, str(n + 1))
            block = block.next()
            top += height

    # -- highlights -----------------------------------------------------------------------------
    def mark_error(self, line: int | None):
        self.error_line = line
        self._highlight_lines()
        self.numbers.update()

    def _highlight_lines(self):
        extra = []
        if not self.isReadOnly():
            sel = QTextEdit.ExtraSelection()
            color = qcolor(self.theme.selection)
            color.setAlpha(60)
            sel.format.setBackground(color)
            sel.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
            sel.cursor = self.textCursor()
            sel.cursor.clearSelection()
            extra.append(sel)
        if self.error_line is not None and 1 <= self.error_line <= self.blockCount():
            bad = QTextEdit.ExtraSelection()
            color = qcolor(self.theme.down)
            color.setAlpha(70)
            bad.format.setBackground(color)
            bad.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
            bad.cursor = QTextCursor(self.document().findBlockByNumber(self.error_line - 1))
            extra.append(bad)
        self.setExtraSelections(extra)

    # -- editing helpers --------------------------------------------------------------------------
    def keyPressEvent(self, event):
        key, mods = event.key(), event.modifiers()
        cur = self.textCursor()
        if key == Qt.Key.Key_Tab and not mods & ~Qt.KeyboardModifier.ShiftModifier:
            if cur.hasSelection() and cur.selectedText().count(" "):
                self._shift_lines(True)
            else:
                col = cur.positionInBlock()
                cur.insertText(INDENT[: len(INDENT) - col % len(INDENT)])
            return
        if key == Qt.Key.Key_Backtab:
            self._shift_lines(False)
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not cur.hasSelection():
            before = cur.block().text()[: cur.positionInBlock()]
            lead = before[: len(before) - len(before.lstrip(" "))]
            extra = INDENT if before.split("#")[0].rstrip().endswith(":") else ""
            cur.insertText("\n" + lead + extra)
            self.setTextCursor(cur)
            return
        super().keyPressEvent(event)

    def _selected_blocks(self) -> tuple[int, int]:
        cur = self.textCursor()
        start, end = cur.selectionStart(), cur.selectionEnd()
        doc = self.document()
        first = doc.findBlock(start).blockNumber()
        last_block = doc.findBlock(end)
        last = last_block.blockNumber() - (1 if end > start and end == last_block.position() else 0)
        return first, max(first, last)

    def _edit_lines(self, fn: Callable[[str], str]):
        first, last = self._selected_blocks()
        cur = self.textCursor()
        cur.beginEditBlock()
        for n in range(first, last + 1):
            block = self.document().findBlockByNumber(n)
            c = QTextCursor(block)
            c.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
            c.insertText(fn(block.text()))
        cur.endEditBlock()

    def _shift_lines(self, right: bool):
        def shift(line: str) -> str:
            if right:
                return INDENT + line if line.strip() else line
            strip = min(len(INDENT), len(line) - len(line.lstrip(" ")))
            return line[strip:]

        self._edit_lines(shift)

    def toggle_comment(self):
        first, last = self._selected_blocks()
        lines = [self.document().findBlockByNumber(n).text() for n in range(first, last + 1)]
        used = [ln for ln in lines if ln.strip()]
        uncomment = bool(used) and all(ln.lstrip().startswith("#") for ln in used)

        def toggle(line: str) -> str:
            if not line.strip():
                return line
            lead = len(line) - len(line.lstrip(" "))
            if uncomment:
                rest = line[lead:][1:]
                return line[:lead] + (rest[1:] if rest.startswith(" ") else rest)
            return line[:lead] + "# " + line[lead:]

        self._edit_lines(toggle)


class ScriptEditor(QWidget):
    """The editor pane: one script at a time, with Save (Ctrl+S), auto-reload and an error bar."""

    #: a reload succeeded: the indicator names the script defines
    reloaded = pyqtSignal(list)
    closed = pyqtSignal()
    status = pyqtSignal(str)

    def __init__(self, themes: ThemeManager, sample: Callable[[], object] | None = None, parent=None):
        super().__init__(parent)
        self.themes = themes
        self.sample = sample or (lambda: None)  # candles for the dry run (default: a synthetic series)
        self.path: Path | None = None
        self.saved_text = ""
        self.edit = CodeEdit(themes.current)
        self.title = QLabel(objectName="Title")
        self.save_btn = QPushButton("Save", toolTip="Save and reload (Ctrl+S)")
        self.save_btn.clicked.connect(self.save)
        self.auto = QCheckBox("Auto-reload", toolTip="Reload one second after typing stops, without saving the file")
        self.close_btn = QPushButton("✕", toolTip="Close the editor")
        self.close_btn.clicked.connect(self.request_close)
        self.error = QLabel(wordWrap=True, textInteractionFlags=Qt.TextInteractionFlag.TextSelectableByMouse)
        self.error.setObjectName("Muted")
        note = QLabel(TRUST_NOTE, objectName="Muted")
        head = QHBoxLayout()
        for w in (self.title,):
            head.addWidget(w, 1)
        for w in (self.auto, self.save_btn, self.close_btn):
            head.addWidget(w)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(head)
        lay.addWidget(self.edit, 1)
        lay.addWidget(self.error)
        lay.addWidget(note)
        self.timer = QTimer(self, singleShot=True, interval=AUTO_RELOAD_MS)
        self.timer.timeout.connect(self.reload)
        self.edit.textChanged.connect(self._changed)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save, context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
        themes.changed.connect(lambda _: self.edit.set_theme(themes.current))
        self.set_message("Nothing open.")

    # -- state ------------------------------------------------------------------------------------
    @property
    def dirty(self) -> bool:
        return self.path is not None and self.edit.toPlainText() != self.saved_text

    def _update_title(self):
        name = self.path.name if self.path else "No script"
        self.title.setText(f"{'● ' if self.dirty else ''}{name}")

    def set_message(self, text: str, ok: bool | None = None, line: int | None = None):
        color = self.themes.current.up if ok else self.themes.current.down if ok is False else None
        self.error.setStyleSheet(f"color: {color};" if color else "")
        self.error.setText(text)
        self.edit.mark_error(line)

    def open_script(self, path: Path, source: str | None = None, saved: bool = True) -> bool:
        """Show ``path`` (``source`` instead of its file's text for a new script or a copy). False if the user
        cancelled because of unsaved changes."""
        if not self.maybe_close():
            return False
        self.path = path
        if source is None:
            source = path.read_text(encoding="utf-8")
        self.edit.blockSignals(True)
        self.edit.setPlainText(source)
        self.edit.blockSignals(False)
        self.saved_text = source if saved else ""  # an unsaved new script starts out ●
        self._update_title()
        self.set_message("Opened. Press Ctrl+S to save and load it." if not saved else "Opened.")
        self.edit.setFocus()
        return True

    def _changed(self):
        self._update_title()
        if self.auto.isChecked():
            self.timer.start()

    # -- saving and reloading ---------------------------------------------------------------------
    def save(self) -> bool:
        if self.path is None:
            return False
        text = self.edit.toPlainText()
        try:
            scripts.save(self.path, text)
        except OSError as exc:
            self.set_message(f"Couldn't save {self.path.name}: {exc}", ok=False)
            return False
        self.saved_text = text
        self._update_title()
        self.reload()
        return True

    def reload(self) -> bool:
        """Compile, import and dry-run the buffer; the last good version stays if it fails."""
        if self.path is None:
            return False
        self.timer.stop()
        result = scripts.reload_script(self.path, self.edit.toPlainText(), self.sample())
        self.set_message(result.message, ok=result.ok, line=result.line)
        self.status.emit(result.message)
        if result.ok:
            self.reloaded.emit([c.name for c in result.classes])
        return result.ok

    def maybe_close(self) -> bool:
        """True when it's fine to replace or close the buffer (saved, or the user chose not to save)."""
        if not self.dirty:
            return True
        box = QMessageBox(QMessageBox.Icon.Question, "Unsaved changes", f"Save changes to {self.path.name}?",
                          QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
                          | QMessageBox.StandardButton.Cancel, self)
        choice = box.exec()
        if choice == QMessageBox.StandardButton.Save:
            return self.save()
        return choice == QMessageBox.StandardButton.Discard

    def request_close(self):
        if self.maybe_close():
            self.timer.stop()
            self.path, self.saved_text = None, ""
            self.edit.blockSignals(True)
            self.edit.clear()
            self.edit.blockSignals(False)
            self.closed.emit()
