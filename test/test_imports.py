"""The one-way import graph. model/ is physics and nothing else.

    model     -> torch, numpy
    estimate  -> model
    train     -> model, estimate
    analysis  -> train, estimate      (the only matplotlib)
"""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
FORBIDDEN = {
    "model":    {"matplotlib", "seaborn", "umap", "sklearn", "pandas",
                 "estimate", "train", "analysis"},
    "estimate": {"matplotlib", "seaborn", "umap", "sklearn", "train", "analysis"},
    "train":    {"matplotlib", "seaborn", "umap", "sklearn", "analysis"},
}


def _imported_roots(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]


def test_layers_do_not_import_upward():
    bad = []
    for package, forbidden in FORBIDDEN.items():
        for path in sorted((ROOT / package).rglob("*.py")):
            for name in _imported_roots(path):
                if name in forbidden:
                    bad.append(f"{path.relative_to(ROOT)} imports {name}")
    assert not bad, "import-direction violations:\n  " + "\n  ".join(bad)


def test_nothing_manipulates_sys_path():
    """The old repo hacked sys.path in 68 of 72 task files, in four mutually
    incompatible flavours, five with an absolute path baked in.

    An editable install makes every one of them unnecessary. This fails if one
    creeps back, which it will the first time someone runs a file from a
    directory where the imports do not resolve and reaches for the quick fix
    instead of `pip install -e .`.
    """
    offenders = []
    for package in ("model", "estimate", "train", "analysis", "tasks"):
        for path in sorted((ROOT / package).rglob("*.py")):
            text = path.read_text()
            for marker in ("sys.path.append", "sys.path.insert", "sys.path ="):
                if marker in text:
                    offenders.append(f"{path.relative_to(ROOT)}: {marker}")
    assert not offenders, (
        "sys.path manipulation found; run `pip install -e . --no-deps` instead:\n  "
        + "\n  ".join(offenders))
