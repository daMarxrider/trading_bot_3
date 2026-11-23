"""Keep docs/ honest (#130).

    python scripts/check_docs.py            # check relative links, anchors and CLI coverage (CI runs this)
    python scripts/check_docs.py --write    # regenerate docs/cli-reference.md from the Typer commands

* every relative link in README.md and docs/*.md must point at an existing file (and an existing heading, for
  `#anchor` links);
* docs/cli-reference.md must equal what the Typer command tree generates (so a new command or option fails CI until
  the page is regenerated) and every command must be mentioned in docs/cli.md;
* docs/legal.md must contain the disclaimer text of tradebot.legal (shown by the About dialog).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REFERENCE = DOCS / "cli-reference.md"
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)\)")


def slug(heading: str) -> str:
    """GitHub's anchor for a heading."""
    text = re.sub(r"[`*_]", "", heading.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    out, fenced = set(), False
    for line in path.read_text().splitlines():
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and (m := re.match(r"#+\s+(.*)", line)):
            out.add(slug(m.group(1)))
    return out


def check_links() -> list[str]:
    problems = []
    for md in [ROOT / "README.md", *sorted(DOCS.glob("*.md"))]:
        fenced = False
        for n, line in enumerate(md.read_text().splitlines(), 1):
            if line.startswith("```"):
                fenced = not fenced
            if fenced:
                continue
            for target in LINK.findall(re.sub(r"`[^`]*`", "", line)):
                if re.match(r"[a-z]+:", target):  # http:, https:, mailto:
                    continue
                name, _, anchor = target.partition("#")
                dest = (md.parent / name).resolve() if name else md
                if not dest.exists():
                    problems.append(f"{md.relative_to(ROOT)}:{n}: broken link {target}")
                elif anchor and dest.suffix == ".md" and anchor not in anchors(dest):
                    problems.append(f"{md.relative_to(ROOT)}:{n}: no heading #{anchor} in {dest.name}")
    return problems


def commands():
    """``(path, command)`` for every command of the CLI, depth first."""
    import typer

    from tradebot.cli.main import app

    def walk(cmd, path):
        subs = getattr(cmd, "commands", None)
        if subs:
            yield path, cmd
            for name, sub in subs.items():
                yield from walk(sub, f"{path} {name}".strip())
        else:
            yield path, cmd

    yield from walk(typer.main.get_command(app), "")


def cell(text: str) -> str:
    return " ".join((text or "").split()).replace("|", "\\|")


def generate() -> str:
    out = ["# CLI reference", "",
           "Every command and option of `tradebot`, generated from the command definitions by "
           "`python scripts/check_docs.py --write`; CI fails when this page is out of date. "
           "For examples see [Command line](cli.md); `tradebot COMMAND --help` prints the same.", ""]
    for path, cmd in commands():
        out += [f"## `{f'tradebot {path}'.strip()}`", "", cell(cmd.help or ""), ""]
        args = [p for p in cmd.params if getattr(p, "param_type_name", "") == "argument"]
        opts = [p for p in cmd.params if getattr(p, "param_type_name", "") == "option"]
        if args:
            out += ["Arguments: " + ", ".join(f"`{p.name.upper()}`" + ("" if p.required else " (optional)") for p in args), ""]
        if opts:
            out += ["| Option | Default | Description |", "|---|---|---|"]
            for p in opts:
                names = ", ".join(f"`{o}`" for o in [*p.opts, *p.secondary_opts])
                default = p.default if not callable(p.default) else None
                shown = "" if default in (None, [], "", False) else f"`{default}`"
                out.append(f"| {names} | {shown} | {cell(p.help)} |")
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def check_cli() -> list[str]:
    problems = []
    if not REFERENCE.exists() or REFERENCE.read_text() != generate():
        problems.append("docs/cli-reference.md is out of date: run python scripts/check_docs.py --write")
    guide = (DOCS / "cli.md").read_text()
    for path, _ in commands():
        if path and not re.search(rf"tradebot {re.escape(path)}\b", guide):
            problems.append(f"docs/cli.md never mentions `tradebot {path}`")
    return problems


def check_legal() -> list[str]:
    """docs/legal.md must carry the exact disclaimer text of the About dialog (#58)."""
    from tradebot import legal

    text = " ".join((DOCS / "legal.md").read_text().split())
    return [f"docs/legal.md is out of sync with tradebot.legal.{name}"
            for name in ("NOT_AFFILIATED", "NO_WARRANTY") if " ".join(getattr(legal, name).split()) not in text]


def main(argv: list[str]) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    if "--write" in argv:
        REFERENCE.write_text(generate())
        print(f"wrote {REFERENCE.relative_to(ROOT)}")
    problems = check_links() + check_cli() + check_legal()
    print("\n".join(problems) or "docs ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
