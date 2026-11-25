# SPDX-License-Identifier: GPL-3.0-or-later
"""One source PNG in, every icon out (#213): PNGs, an SVG wrapper, tradebot.ico and tradebot.icns in
src/tradebot/gui/assets/desktop/. Needs PyQt6 (the gui extra) only; the .ico and .icns containers are written here,
so neither Pillow nor macOS' iconutil is needed.

    QT_QPA_PLATFORM=offscreen uv run --extra gui python packaging/linux/make_icons.py [SOURCE.png] [--crop X,Y,W,H]

SOURCE defaults to packaging/linux/icon-source.png (square, 512 px or larger). --crop cuts a square out of it first."""

import base64
import struct
import sys
from pathlib import Path

from PyQt6.QtCore import QBuffer, QIODevice, QRect, Qt
from PyQt6.QtGui import QGuiApplication, QImage

HERE = Path(__file__).parent
OUT = HERE.parents[1] / "src" / "tradebot" / "gui" / "assets" / "desktop"
PNG_SIZES = (32, 48, 64, 128, 256, 512)
ICO_SIZES = (32, 48, 64, 128, 256)
ICNS_TYPES = {32: b"icp5", 64: b"icp6", 128: b"ic07", 256: b"ic08", 512: b"ic09"}  # PNG-compressed icns entries


def png_bytes(image: QImage) -> bytes:
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buf, "PNG")
    return bytes(buf.data())


def ico(pngs: dict[int, bytes]) -> bytes:
    """ICO with PNG-compressed entries (Windows Vista and later; 256 is stored as 0)."""
    sizes = [s for s in ICO_SIZES]
    head = struct.pack("<HHH", 0, 1, len(sizes))
    offset = 6 + 16 * len(sizes)
    entries, data = b"", b""
    for s in sizes:
        entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(pngs[s]), offset + len(data))
        data += pngs[s]
    return head + entries + data


def icns(pngs: dict[int, bytes]) -> bytes:
    body = b"".join(kind + struct.pack(">I", 8 + len(pngs[s])) + pngs[s] for s, kind in ICNS_TYPES.items())
    return b"icns" + struct.pack(">I", 8 + len(body)) + body


def main(argv: list[str]) -> None:
    crop = None
    if "--crop" in argv:
        i = argv.index("--crop")
        crop = QRect(*[int(v) for v in argv[i + 1].split(",")])
        del argv[i:i + 2]
    source = Path(argv[0]) if argv else HERE / "icon-source.png"
    QGuiApplication([])
    image = QImage(str(source))
    if image.isNull():
        sys.exit(f"cannot read {source}")
    if crop is not None:
        image = image.copy(crop)
    if image.width() != image.height():
        sys.exit("the icon must be square (use --crop X,Y,W,H)")
    pngs = {}
    for size in PNG_SIZES:
        scaled = image.scaled(size, size, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
        scaled.save(str(OUT / f"tradebot-{size}.png"))
        pngs[size] = png_bytes(scaled)
    (OUT / "tradebot.ico").write_bytes(ico(pngs))
    (OUT / "tradebot.icns").write_bytes(icns(pngs))
    b64 = base64.b64encode(pngs[512]).decode()
    (OUT / "tradebot.svg").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="512" height="512" viewBox="0 0 512 512"><image width="512" height="512" xlink:href="data:image/png;base64,{b64}"/></svg>\n',
        encoding="utf-8")


main(sys.argv[1:])
