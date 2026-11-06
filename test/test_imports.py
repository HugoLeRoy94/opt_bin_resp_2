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
