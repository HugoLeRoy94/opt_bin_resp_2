"""Drawing helpers. The only module in the project that imports matplotlib.

Every function takes an existing `ax` and draws on it. None of them creates a
figure, calls show(), or writes a file. Figures are never saved: the caller
decides what to do with the window, and nothing lands in the repo by accident.

The three functions differ in what they claim:

    band    mean with a min/max envelope over replicates. The honest default
            when replicates are independent worlds and there are few of them.
    curves  a line with explicit error bars, when you have a column saying how
            wide the error is.
    points  every run as its own marker. Use it to check that a band is
            describing a real cluster rather than two runs and an outlier.

A gap between two bands that is no wider than the bands themselves is not a
result, which is why `band` draws the envelope by default rather than on
request.
"""
import matplotlib.pyplot as plt
import numpy as np

# One appearance per arm, so the same condition is the same colour in every
# figure. Keys are whatever a task puts in its grouping column; unknown keys
# fall through to the matplotlib cycle rather than raising, since an exploratory
# plot should not have to register a style first.
STYLE = {
    "heteromers":           ("o-",  "tab:blue"),
    "homomers":             ("s--", "tab:orange"),
    "complete/heteromers":  ("o-",  "tab:blue"),
    "complete/homomers":    ("s--", "tab:blue"),
    "bernoulli/heteromers": ("o-",  "tab:orange"),
    "bernoulli/homomers":   ("s--", "tab:orange"),
}


def style_for(key, index=0):
    """(marker_spec, colour) for a group key, falling back to the colour cycle."""
    if key in STYLE:
        return STYLE[key]
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return "o-", cycle[index % len(cycle)]


def _groups(frame, by):
    """(label, subframe) pairs, with `by` either one column or several."""
    if by is None:
        yield None, frame
        return
    columns = [by] if isinstance(by, str) else list(by)
    for key, part in frame.groupby(columns if len(columns) > 1 else columns[0],
                                   sort=True):
        label = "/".join(map(str, key)) if isinstance(key, tuple) else str(key)
        yield label, part


def band(ax, frame, x, y, by=None, alpha=.15, **kw):
    """One line per group plus its "<y> lo" / "<y> hi" envelope.

    Expects the columns across_replicates() produces. Raises if the envelope
    columns are missing rather than quietly drawing a bare line, because a line
    without its band invites reading a difference that the spread does not
    support.
    """
    lo, hi = f"{y} lo", f"{y} hi"
    if lo not in frame.columns or hi not in frame.columns:
        raise KeyError(
            f"{lo!r} and {hi!r} are not in this frame. Pass it through "
            f"across_replicates() first, or use curves() for a bare line.")
    for i, (label, part) in enumerate(_groups(frame, by)):
        part = part.sort_values(x)
        spec, colour = style_for(label, i)
        ax.plot(part[x], part[y], spec, color=colour, label=label, **kw)
        ax.fill_between(part[x], part[lo], part[hi],
                        color=colour, alpha=alpha, linewidth=0)
    return ax


def curves(ax, frame, x, y, by=None, err=None, **kw):
    """One line per group, with error bars when `err` names a column."""
    for i, (label, part) in enumerate(_groups(frame, by)):
        part = part.sort_values(x)
        spec, colour = style_for(label, i)
        ax.plot(part[x], part[y], spec, color=colour, label=label, **kw)
        if err is not None:
            ok = part[err].notna()
            ax.errorbar(part.loc[ok, x], part.loc[ok, y], yerr=part.loc[ok, err],
                        fmt="none", color=colour, capsize=3)
    return ax


def points(ax, frame, x, y, by=None, jitter=0.0, alpha=.6, **kw):
    """Every run as its own marker, so n and spread are visible.

    `jitter` spreads overlapping markers along x by that fraction of the median
    x spacing. It is cosmetic and never touches y.
    """
    for i, (label, part) in enumerate(_groups(frame, by)):
        _spec, colour = style_for(label, i)
        xs = np.asarray(part[x], dtype=float)
        if jitter:
            spacing = np.median(np.diff(np.unique(xs))) if np.unique(xs).size > 1 else 1.0
            xs = xs + (np.random.default_rng(i).uniform(-.5, .5, xs.size)
                       * jitter * spacing)
        ax.plot(xs, part[y], "o", color=colour, alpha=alpha, markeredgecolor="none",
                linestyle="none", label=label, **kw)
    return ax


def reference(ax, value, label=None, axis="y", **kw):
    """A horizontal or vertical reference line, e.g. a ceiling like log2(B)."""
    style = dict(color="k", linestyle=":", linewidth=1, alpha=.7)
    style.update(kw)
    line = (ax.axhline if axis == "y" else ax.axvline)(value, **style)
    if label:
        line.set_label(label)
    return line
