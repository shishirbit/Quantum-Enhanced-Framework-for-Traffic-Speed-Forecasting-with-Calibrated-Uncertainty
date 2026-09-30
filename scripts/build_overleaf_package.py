"""Build a self-contained Overleaf submission archive.

The highlighted manuscript compares the current manuscript with the last
pre-review commit and renders additions or replacements in blue. Deletions are
documented in the response letter and therefore have no text to colour in the
revised manuscript.
"""
from __future__ import annotations

import difflib
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manuscript" / "quarts_springer"
OUTPUT = ROOT / "output"
BASE_REVISION = "a2b89a1"


def run(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, encoding="utf-8")


def changed_lines(old: list[str], new: list[str]) -> set[int]:
    changed: set[int] = set()
    matcher = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in {"insert", "replace"}:
            changed.update(range(j1, j2))
    return changed


def environment_spans(lines: list[str]) -> list[tuple[int, int]]:
    tracked = {
        "figure",
        "figure*",
        "table",
        "table*",
        "equation",
        "equation*",
        "align",
        "align*",
        "algorithm",
        "itemize",
        "enumerate",
    }
    begin_re = re.compile(r"\\begin\{([^}]+)\}")
    end_re = re.compile(r"\\end\{([^}]+)\}")
    stack: list[tuple[str, int]] = []
    spans: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        for match in begin_re.finditer(line):
            if match.group(1) in tracked:
                stack.append((match.group(1), index))
        for match in end_re.finditer(line):
            env = match.group(1)
            for position in range(len(stack) - 1, -1, -1):
                if stack[position][0] == env:
                    _, start = stack.pop(position)
                    spans.append((start, index))
                    break
    return spans


def paragraph_span(lines: list[str], index: int, first_body: int, last_body: int) -> tuple[int, int]:
    start = index
    end = index
    while start > first_body and lines[start - 1].strip():
        start -= 1
    while end < last_body and lines[end + 1].strip():
        end += 1
    return start, end


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def highlighted_source(old_text: str, new_text: str) -> str:
    old = old_text.splitlines()
    new = new_text.splitlines()
    changed = changed_lines(old, new)
    first_body = next(index for index, line in enumerate(new) if r"\begin{document}" in line) + 1
    last_body = next(index for index, line in enumerate(new) if r"\end{document}" in line) - 1
    spans = environment_spans(new)
    selected: list[tuple[int, int]] = []
    for index in sorted(changed):
        if index < first_body or index > last_body:
            continue
        containing = [(start, end) for start, end in spans if start <= index <= end]
        if containing:
            selected.append(max(containing, key=lambda item: item[1] - item[0]))
        else:
            selected.append(paragraph_span(new, index, first_body, last_body))
    selected = merge_spans(selected)
    starts = {start: end for start, end in selected}
    ends = {end for _, end in selected}
    output: list[str] = []
    for index, line in enumerate(new):
        if index in starts:
            output.append(r"\color{blue}")
        output.append(line)
        if index in ends:
            output.append(r"\color{black}")
        if line.strip() == r"\maketitle":
            output.extend(
                [
                    "",
                    r"\begin{center}",
                    r"\textcolor{blue}{\small Blue text marks highlighted additions and replacements.}",
                    r"\end{center}",
                ]
            )
    return "\n".join(output) + "\n"


def next_package_directory() -> Path:
    base = OUTPUT / "QUARTS_Overleaf_Submission_Package_20261001"
    if not base.exists():
        return base
    counter = 2
    while True:
        candidate = OUTPUT / f"QUARTS_Overleaf_Submission_Package_20261001_v{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


def copy_referenced_figures(manuscript: str, package: Path) -> None:
    pattern = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")
    for relative in sorted(set(pattern.findall(manuscript))):
        source = SOURCE / relative
        destination = package / relative
        if not source.exists():
            raise FileNotFoundError(f"Referenced figure is missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    package = next_package_directory()
    package.mkdir(parents=True)
    current = (SOURCE / "main.tex").read_text(encoding="utf-8")
    original = run("git", "show", f"{BASE_REVISION}:manuscript/quarts_springer/main.tex")

    shutil.copy2(SOURCE / "main.tex", package / "main.tex")
    (package / "main_highlighted.tex").write_text(
        highlighted_source(original, current), encoding="utf-8"
    )
    for name in (
        "response_to_reviewers.tex",
        "references.bib",
        "sn-jnl.cls",
        "sn-mathphys-num.bst",
    ):
        shutil.copy2(SOURCE / name, package / name)
    shutil.copytree(SOURCE / "bst", package / "bst")
    copy_referenced_figures(current, package)

    readme = """QUARTS OVERLEAF SUBMISSION PACKAGE

Primary clean manuscript: main.tex
Highlighted revision: main_highlighted.tex
Point-by-point response: response_to_reviewers.tex
Bibliography: references.bib

Upload the contents of this folder to a new Overleaf project. Select the desired
main document in Overleaf's project settings. The clean manuscript should be the
primary submission file; the highlighted manuscript and response letter are
provided as separate revision documents. The project uses the included Springer
Nature sn-jnl class and numerical mathematics and physics bibliography style.

Blue text in main_highlighted.tex identifies additions or replacements made
during peer review. Deletions are described in response_to_reviewers.tex.
"""
    (package / "README.txt").write_text(readme, encoding="utf-8")

    tectonic = ROOT / "manuscript" / "tools" / "tectonic" / "tectonic.exe"
    build_dir = ROOT / "tmp" / "overleaf_build" / package.name
    build_dir.mkdir(parents=True)
    for tex in ("main.tex", "main_highlighted.tex", "response_to_reviewers.tex"):
        subprocess.run(
            [str(tectonic), "--outdir", str(build_dir), str(package / tex)],
            cwd=ROOT,
            check=True,
        )
    shutil.copy2(build_dir / "main.pdf", package / "QUARTS_clean_manuscript.pdf")
    shutil.copy2(
        build_dir / "main_highlighted.pdf", package / "QUARTS_highlighted_manuscript.pdf"
    )
    shutil.copy2(
        build_dir / "response_to_reviewers.pdf", package / "QUARTS_response_to_reviewers.pdf"
    )

    archive = shutil.make_archive(str(package), "zip", package.parent, package.name)
    print(package)
    print(archive)


if __name__ == "__main__":
    main()
