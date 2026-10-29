# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Write THIRD_PARTY_NOTICES.txt: the licence of every library the binary bundles (#58).

The bundled set is tradebot's runtime dependencies plus the ``gui`` extra, resolved from the installed
packages; build and test tools (pyinstaller, pytest) and optional extras like ``pytr`` are not in it.
Python's own licence is added too, since the interpreter ships inside the binary.
    python packaging/third_party_notices.py [out-file]
"""

from __future__ import annotations

import importlib.metadata as md
import re
import sys
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement

ROOT, EXTRAS = "tradebot", ("gui",)


def bundled() -> list[md.Distribution]:
    """The installed distributions reachable from ``tradebot[gui]``, sorted by name."""
    found: dict[str, md.Distribution] = {}
    todo = [(ROOT, EXTRAS)]
    while todo:
        name, extras = todo.pop()
        key = re.sub(r"[-_.]+", "-", name).lower()
        if key in found:
            continue
        try:
            dist = found[key] = md.distribution(name)
        except md.PackageNotFoundError:
            continue  # e.g. a platform-only dependency not installed here
        for spec in dist.requires or []:
            req = Requirement(spec)
            if req.marker is None or any(req.marker.evaluate({**default_environment(), "extra": e})
                                         for e in extras or ("",)):
                todo.append((req.name, tuple(req.extras)))
    return sorted((d for k, d in found.items() if k != ROOT), key=lambda d: d.metadata["Name"].lower())


def licence_texts(dist: md.Distribution) -> list[str]:
    files = [f for f in dist.files or [] if re.search(r"(LICEN[CS]E|COPYING|NOTICE|AUTHORS)", f.name, re.I)
             and f.suffix.lower() in ("", ".txt", ".md", ".rst")]
    return [Path(f.locate()).read_text(errors="replace").strip() for f in files]


def python_licence() -> str | None:
    for base in (Path(sys.base_prefix), Path(sys.base_prefix) / "lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}"):
        for name in ("LICENSE.txt", "LICENSE"):
            if (base / name).is_file():
                return (base / name).read_text(errors="replace").strip()
    return None


def render() -> str:
    out = ["tradebot bundles the following third-party software. Each is distributed under its own licence,",
           "reproduced below. tradebot itself is licensed under the GNU GPL v3.0 or later (see LICENSE).", ""]
    sections = [(f"Python {sys.version.split()[0]}", "PSF-2.0", [python_licence() or "see https://docs.python.org/3/license.html"])]
    for dist in bundled():
        meta = dist.metadata
        licence = meta.get("License-Expression") or (meta.get("License") or "").split("\n")[0]
        texts = licence_texts(dist) or [f"no licence file shipped; see {meta.get('Home-page') or 'the project page on PyPI'}"]
        sections.append((f"{meta['Name']} {dist.version}", licence, texts))
    for title, licence, texts in sections:
        out += ["=" * 100, f"{title}  ({licence})" if licence else title, "=" * 100, *texts, ""]
    return "\n".join(out)


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "THIRD_PARTY_NOTICES.txt")
    target.write_text(render())
    print(f"wrote {target}")
