import io
import json
import sys
import tokenize
from pathlib import Path

EM_DASH = chr(0x2014)
ESCAPED_EM_DASH = "\\" + "u2014"
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".txt", ".json", ".ipynb", ".toml", ".cfg"}
SKIP_DIRS = {".git", ".venv", "venv", "data", "logs", "__pycache__", ".ipynb_checkpoints"}
ALLOWED_COMMENTS = ("#!", "# noqa", "# type:")


def iter_files(roots):
    for root in map(Path, roots):
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in candidates:
            if path.is_file() and path.suffix in TEXT_SUFFIXES and not SKIP_DIRS & set(path.parts):
                yield path


def em_dash_hits(where, text, escaped):
    needles = (EM_DASH, ESCAPED_EM_DASH) if escaped else (EM_DASH,)
    return [(where, n, "em dash") for n, line in enumerate(text.splitlines(), 1) if any(x in line for x in needles)]


def comment_hits(where, source):
    hits = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT and not tok.string.startswith(ALLOWED_COMMENTS):
                hits.append((where, tok.start[0], "comment"))
    except (tokenize.TokenError, SyntaxError):
        hits.append((where, 0, "could not parse"))
    return hits


def notebook_cells(text):
    for index, cell in enumerate(json.loads(text).get("cells", []), 1):
        if cell.get("cell_type") == "code":
            lines = "".join(cell.get("source", [])).splitlines()
            yield index, "\n".join("" if line.lstrip().startswith(("!", "%")) else line for line in lines)


def check(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    hits = em_dash_hits(str(path), text, path.suffix == ".ipynb")
    if path.suffix == ".py":
        hits += comment_hits(str(path), text)
    if path.suffix == ".ipynb":
        for index, source in notebook_cells(text):
            hits += comment_hits(f"{path} cell {index}", source)
    return hits


def main(roots):
    hits = [hit for path in iter_files(roots or ["."]) for hit in check(path)]
    for where, line, kind in hits:
        print(f"{where}:{line}: {kind}")
    print(f"{len(hits)} problem(s)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
