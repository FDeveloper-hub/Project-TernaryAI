"""Ejecuta, en orden y en un mismo espacio de nombres, los bloques ```python de los manuales.

Uso:  python3 tests/test_manual.py [docs/MANUAL.md docs/MANUAL.en.md]
Cada manual se ejecuta en una carpeta temporal. Los bloques ```text y ```bash no se ejecutan.
"""
import os
import re
import sys
import tempfile
import traceback

FENCE = re.compile(r"^```(\w*)[ \t]*\n(.*?)^```[ \t]*$", re.S | re.M)


def blocks(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return [(text.count("\n", 0, m.start()) + 1, m.group(1), m.group(2)) for m in FENCE.finditer(text)]


def run(path):
    ns = {"__name__": "__manual__"}
    old = os.getcwd()
    n = 0
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            for line, lang, code in blocks(path):
                if lang != "python":
                    continue
                try:
                    exec(compile(code, f"{path}:{line}", "exec"), ns)
                except Exception:
                    print(f"FALLA en {path}, bloque de la línea {line}:")
                    traceback.print_exc()
                    return False, n
                n += 1
        finally:
            os.chdir(old)
    return True, n


def structure(path):
    """Cantidad de bloques por lenguaje y lista de encabezados (para comparar idiomas)."""
    bl = blocks(path)
    with open(path, encoding="utf-8") as f:
        heads = [re.sub(r"[^\d.]", "", l.split(" ")[1]) if re.match(r"#{2,3} \d", l) else None
                 for l in f.read().splitlines() if l.startswith("#")]
    count = {}
    for _, lang, _ in bl:
        count[lang] = count.get(lang, 0) + 1
    return count, [h for h in heads if h]


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    paths = sys.argv[1:] or [os.path.join(root, "docs", "MANUAL.md"), os.path.join(root, "docs", "MANUAL.en.md")]
    paths = [p for p in paths if os.path.exists(p)]
    ok_all = True
    for p in paths:
        ok, n = run(os.path.abspath(p))
        print(("OK " if ok else "MAL"), os.path.basename(p), f"({n} bloques python ejecutados)")
        ok_all &= ok
    if len(paths) == 2:
        a, b = structure(paths[0]), structure(paths[1])
        same = a == b
        print(("OK " if same else "MAL"), "mismos bloques y encabezados numerados en ambos idiomas")
        if not same:
            print(a, b, sep="\n")
        ok_all &= same
    sys.exit(0 if ok_all else 1)
