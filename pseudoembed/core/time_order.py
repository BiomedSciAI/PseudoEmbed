"""Validated construction of a numeric time ordering from ``adata.obs`` labels.

Why this module exists
----------------------
Two call sites used to build their time ordering like this::

    unique_times = sorted(set(adata.obs[time_key].values))
    time_order = {t: i for i, t in enumerate(unique_times)}

That is a *lexicographic* sort of arbitrary labels.  It has no numeric parse, no
validation, and no handling for labels that are not timepoints at all.  Two
consequences were measured on shipped datasets:

* On unpadded day labels ``"D10" < "D2"``, so the ordering is simply wrong.
  Schiebinger 2019 (16-day) sorts to
  ``['D0','D10','D11','D12','D16','D2','D4','D6','D8','D9','iPSCs']``.
* A label that is *not a timepoint* — an established iPSC line, a control arm,
  a NaN — silently receives a time index and becomes eligible as "earliest".
  This is the mechanism behind the recorded Schiebinger root failure where the
  root landed on an iPSC cell and inverted tau.

Both source frames were ``pandas.Categorical`` carrying the *correct* order in
``.categories``, which the string sort threw away.

The contract here
-----------------
:func:`build_time_order` returns a :class:`TimeOrdering`.  It resolves an
ordering from, in strict order of precedence:

1. an explicit user-supplied ``time_order`` (list of labels, or mapping
   label -> numeric position);
2. an already-ordered dtype — ``pandas.Categorical`` categories, or a numeric
   column — because that order is a *declaration*, not an accident;
3. a numeric parse of the label strings ("D0", "day 3", "3h", "0.5", "T12").

If none of those establishes an ordering it raises :class:`TimeOrderError`
rather than falling back to a string sort.  A silent wrong answer is the bug
this module fixes, so there is deliberately no "warn and guess" path.

Non-timepoint labels
--------------------
Labels that cannot be placed on the axis (NaN, or an unparseable label such as
``"iPSCs"`` when its siblings parse as days) are **excluded** from the ordering
and their cells get ``time_numeric = nan``.  They are *not* assigned an invented
day: an established line is a held-out endpoint, and giving it a coordinate
destroys its value as independent validation.  Downstream:

* root candidacy — excluded cells are never root candidates (nan fails the
  ``== earliest`` test);
* temporal bias — a pair involving an excluded cell has ``dt = nan``, which
  compares false against both ``> 0`` and ``< 0`` and so takes the
  ``same_weight`` branch, i.e. no directional claim is made about it;
* axis classification — Spearman against the time reference is computed on the
  cells that *do* have a coordinate.

Excluded cells therefore stay in the hypergraph and keep receiving a pseudotime;
they simply contribute no time-direction evidence.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = ["TimeOrderError", "TimeOrdering", "build_time_order"]


class TimeOrderError(ValueError):
    """Raised when a time ordering cannot be established from the labels."""


# Matches a leading tag ("D", "day", "t", "hr", "h", "wk"…) followed by a
# number, or a bare number, with an optional trailing unit.  Deliberately
# anchored: a label that merely *contains* a digit ("GSM3195694", "Plate 3 E1")
# must not parse, or batch identifiers would silently become timepoints.
_NUM_LABEL = re.compile(
    r"""^\s*
        (?:(?P<pre>[A-Za-z]{0,4})[\s_\-]*)?     # optional leading tag
        (?P<num>[+-]?\d+(?:\.\d+)?)             # the number
        [\s_\-]*
        (?P<post>[A-Za-z]{0,4})                 # optional trailing unit
        \s*$
    """,
    re.VERBOSE,
)

# Leading/trailing tags that are consistent with a time axis.  A tag outside
# this set (e.g. "PC" in "PC3", "well" in "well 4") is rejected rather than
# silently treated as a day.
_TIME_TAGS = {
    "",
    "d",
    "day",
    "days",
    "t",
    "tp",
    "time",
    "h",
    "hr",
    "hrs",
    "hour",
    "hours",
    "m",
    "min",
    "mins",
    "minute",
    "minutes",
    "w",
    "wk",
    "wks",
    "week",
    "weeks",
    "p",
    "e",
    "dpi",
    "dpf",
    "hpf",
    "s",
    "sec",
}

# Units that carry a scale.  Only used when the labels mix units, which is rare;
# a single consistent unit needs no conversion because only the *order* matters.
_UNIT_SCALE = {
    "s": 1.0 / 3600.0,
    "sec": 1.0 / 3600.0,
    "m": 1.0 / 60.0,
    "min": 1.0 / 60.0,
    "mins": 1.0 / 60.0,
    "minute": 1.0 / 60.0,
    "minutes": 1.0 / 60.0,
    "h": 1.0,
    "hr": 1.0,
    "hrs": 1.0,
    "hour": 1.0,
    "hours": 1.0,
    "d": 24.0,
    "day": 24.0,
    "days": 24.0,
    "dpi": 24.0,
    "dpf": 24.0,
    "w": 168.0,
    "wk": 168.0,
    "wks": 168.0,
    "week": 168.0,
    "weeks": 168.0,
}


@dataclass
class TimeOrdering:
    """Resolved time ordering.

    Attributes
    ----------
    time_numeric : ndarray of float, shape (n_cells,)
        Per-cell position on the time axis.  ``nan`` for cells whose label is
        not a timepoint (see module docstring).  Values are *ranks* (0, 1, 2…)
        over the ordered labels, not the parsed magnitudes, so that the
        temporal-bias branch logic and ``== earliest`` root test behave as
        before for well-formed input.
    order : list
        Ordered labels, earliest first.
    index_of : dict
        ``label -> rank``.  Contains only ordered (non-excluded) labels.
    excluded : list
        Labels deliberately left off the axis.
    source : str
        Which precedence rule resolved the ordering, for logging/provenance.
    parsed_value : dict
        ``label -> parsed magnitude IN HOURS`` when a numeric parse was used,
        empty otherwise.  Unit-normalised so that mixed units order correctly
        ("90min" before "3h"), which means a day label does NOT round-trip to
        its face value: ``"D16" -> 384.0``, not 16.0.  Use
        :attr:`parsed_label_value` if you want the number as written.
        Recorded for provenance only; the axis itself uses ranks.
    parsed_label_value : dict
        ``label -> the number as written in the label`` (``"D16" -> 16.0``),
        with no unit scaling.  Convenient as a ground-truth "day" column, but
        NOT safe to compare across mixed units.
    """

    time_numeric: np.ndarray
    order: List[Any]
    index_of: Dict[Any, int]
    excluded: List[Any] = field(default_factory=list)
    source: str = "unknown"
    parsed_value: Dict[Any, float] = field(default_factory=dict)
    parsed_label_value: Dict[Any, float] = field(default_factory=dict)

    @property
    def earliest_index(self) -> int:
        """Rank of the earliest ordered label (always 0 for a non-empty axis)."""
        if not self.index_of:
            raise TimeOrderError("No ordered timepoints, so no earliest index.")
        return min(self.index_of.values())

    @property
    def earliest_label(self) -> Any:
        return self.order[0]

    @property
    def n_excluded_cells(self) -> int:
        return int(np.isnan(self.time_numeric).sum())

    def describe(self) -> str:
        txt = f"  Time points ({self.source}): {self.order}"
        if self.excluded:
            txt += (
                f"\n  Excluded from the time axis: {self.excluded} "
                f"({self.n_excluded_cells} cells → time_numeric = nan; "
                "never root candidates)"
            )
        return txt


def _is_missing(label: Any) -> bool:
    if label is None:
        return True
    try:
        if pd.isna(label):
            return True
    except (TypeError, ValueError):  # array-like or exotic label
        return False
    if isinstance(label, str) and label.strip().lower() in {
        "",
        "na",
        "nan",
        "none",
        "null",
        "unknown",
        "n/a",
        "?",
    }:
        return True
    return False


def _parse_pair(label: Any) -> Optional[tuple[float, float]]:
    """Parse a label to ``(magnitude_in_hours, number_as_written)``, or ``None``
    if it is not a timepoint-shaped label.  Unit-aware; see ``_UNIT_SCALE``.

    Two numbers because they serve different jobs: the scaled one makes mixed
    units order correctly ("90min" < "3h"), while the raw one is what a caller
    means by "the day" (``"D16" -> 16.0``, not 384.0).
    """
    if isinstance(label, (int, float, np.integer, np.floating)) and not _is_missing(
        label
    ):
        return float(label), float(label)
    if not isinstance(label, str):
        return None
    m = _NUM_LABEL.match(label)
    if m is None:
        return None
    pre = (m.group("pre") or "").strip().lower()
    post = (m.group("post") or "").strip().lower()
    # Reject a tag that is not time-shaped, so "PC3" / "well 4" do not parse.
    if pre not in _TIME_TAGS or post not in _TIME_TAGS:
        return None
    if pre and post:
        # e.g. "d3h" — ambiguous, refuse rather than pick one.
        return None
    val = float(m.group("num"))
    unit = post or pre
    return val * _UNIT_SCALE.get(unit, 1.0), val


def _parse_one(label: Any) -> Optional[float]:
    """Unit-scaled magnitude only.  See :func:`_parse_pair`."""
    pair = _parse_pair(label)
    return None if pair is None else pair[0]


def _from_explicit(
    time_order: Union[Sequence[Any], Mapping[Any, Any]],
    present: List[Any],
) -> tuple[List[Any], Dict[Any, int], List[Any]]:
    """Resolve an ordering supplied by the caller.  Takes precedence over every
    inference, including a Categorical's own order."""
    if isinstance(time_order, Mapping):
        items = [(k, float(v)) for k, v in time_order.items()]
        items.sort(key=lambda kv: kv[1])
        declared = [k for k, _ in items]
    else:
        declared = list(time_order)
        if len(set(declared)) != len(declared):
            dup = [x for x in set(declared) if declared.count(x) > 1]
            raise TimeOrderError(
                f"Explicit time_order contains duplicate labels: {sorted(map(str, dup))}."
            )

    present_set = set(present)
    ordered = [lbl for lbl in declared if lbl in present_set]
    if not ordered:
        raise TimeOrderError(
            "Explicit time_order matches none of the labels in the data.\n"
            f"  time_order : {list(map(str, declared))}\n"
            f"  in data    : {sorted(map(str, present_set))}"
        )
    missing = [lbl for lbl in present if lbl not in set(declared)]
    return ordered, {lbl: i for i, lbl in enumerate(ordered)}, missing


def build_time_order(
    labels: Any,
    time_order: Optional[Union[Sequence[Any], Mapping[Any, Any]]] = None,
    *,
    key_name: str = "time",
    verbose: bool = True,
) -> TimeOrdering:
    """Build a validated numeric time ordering.

    Parameters
    ----------
    labels : array-like or pandas Series
        The raw ``adata.obs[time_key]`` column.  A ``Categorical`` with ordered
        (or merely declared) categories is honoured as-is — its category order
        is treated as a user declaration and is never re-sorted.
    time_order : sequence or mapping, optional
        Explicit ordering, earliest first (or ``label -> position``).  Takes
        precedence over any inference.  Labels present in the data but absent
        from this argument are excluded from the axis.
    key_name : str
        Column name, used only in error and log messages.
    verbose : bool
        Print the resolved ordering.

    Returns
    -------
    TimeOrdering

    Raises
    ------
    TimeOrderError
        If no ordering can be established.  There is no string-sort fallback.
    """
    series = labels if isinstance(labels, pd.Series) else pd.Series(np.asarray(labels))

    # Preserve declared category order before any set() destroys it.
    declared_categories: Optional[List[Any]] = None
    if isinstance(series.dtype, pd.CategoricalDtype):
        declared_categories = list(series.cat.categories)

    values = series.to_numpy()
    # ``present`` keeps first-appearance order so messages are stable and we
    # never rely on a sort of the raw labels anywhere in this function.
    present: List[Any] = []
    seen = set()
    for v in values:
        if _is_missing(v):
            continue
        try:
            hv = v if not isinstance(v, np.generic) else v.item()
        except Exception:  # pragma: no cover - exotic label types
            hv = v
        if hv not in seen:
            seen.add(hv)
            present.append(hv)

    if not present:
        raise TimeOrderError(
            f"Column '{key_name}' has no non-missing labels, so no time "
            "ordering can be built."
        )

    parsed_value: Dict[Any, float] = {}
    parsed_label_value: Dict[Any, float] = {}

    # ---------------- 1. explicit user-supplied ordering ----------------
    if time_order is not None:
        ordered, index_of, excluded = _from_explicit(time_order, present)
        source = "explicit time_order"

    # ---------------- 2. an order already declared by the dtype ---------
    elif declared_categories is not None:
        ordered = [c for c in declared_categories if c in seen]
        if not ordered:
            raise TimeOrderError(
                f"Column '{key_name}' is Categorical but none of its categories "
                "appear in the data."
            )
        # A Categorical declares an order, but it may still carry a
        # non-timepoint level ("iPSCs").  If the *other* levels parse
        # numerically and this one does not, it is not a timepoint — exclude it
        # rather than granting it a position.
        pairs = {lbl: _parse_pair(lbl) for lbl in ordered}
        pv = {k: (None if p is None else p[0]) for k, p in pairs.items()}
        raw = {k: p[1] for k, p in pairs.items() if p is not None}
        n_parsed = sum(v is not None for v in pv.values())
        excluded = []
        if n_parsed >= 2 and n_parsed < len(ordered):
            excluded = [lbl for lbl in ordered if pv[lbl] is None]
            ordered = [lbl for lbl in ordered if pv[lbl] is not None]
            parsed_value = {k: v for k, v in pv.items() if v is not None}
            parsed_label_value = dict(raw)
            source = "categorical order (non-timepoint levels excluded)"
        else:
            if n_parsed == len(ordered):
                parsed_value = {k: float(v) for k, v in pv.items()}  # type: ignore[arg-type]
                parsed_label_value = dict(raw)
            source = "categorical order"
        index_of = {lbl: i for i, lbl in enumerate(ordered)}
        # A declared order that disagrees with a clean numeric parse is almost
        # always a stale categories list; say so loudly but honour the
        # declaration, since the user may have meant it.
        if len(parsed_value) == len(ordered) and len(ordered) > 1:
            mags = [parsed_value[lbl] for lbl in ordered]
            if any(b < a for a, b in zip(mags, mags[1:])):
                warnings.warn(
                    f"Column '{key_name}' is Categorical and its declared "
                    "category order is NOT numerically increasing "
                    f"({list(map(str, ordered))} -> {mags}). Honouring the "
                    "declared order. Pass an explicit `time_order` to override.",
                    UserWarning,
                    stacklevel=2,
                )

    # ---------------- 3. numeric parse of the labels --------------------
    else:
        if pd.api.types.is_numeric_dtype(series):
            pairs = {lbl: (float(lbl), float(lbl)) for lbl in present}
        else:
            pairs = {lbl: _parse_pair(lbl) for lbl in present}  # type: ignore[misc]
        pv = {k: (None if p is None else p[0]) for k, p in pairs.items()}
        ok = {k: v for k, v in pv.items() if v is not None}
        parsed_label_value = {k: p[1] for k, p in pairs.items() if p is not None}
        if len(ok) < 2:
            raise TimeOrderError(
                f"Cannot establish a time ordering for column '{key_name}'.\n"
                f"  Labels: {sorted(map(str, present))}\n"
                f"  Parsed as numeric: {sorted(map(str, ok))}\n"
                "Fewer than two labels carry a parseable time value, so any "
                "ordering would be a guess. This function deliberately does NOT "
                "fall back to a lexicographic sort (that was the bug it "
                "replaces: 'D10' sorts before 'D2').\n"
                "Fix by either:\n"
                "  * passing time_order=[...] with the labels earliest-first, or\n"
                "  * making the column an ordered pandas Categorical, or\n"
                "  * using parseable labels ('D0', 'day 3', '3h', '0.5')."
            )
        excluded = [lbl for lbl in present if lbl not in ok]
        ordered = sorted(ok, key=lambda lbl: ok[lbl])
        index_of = {lbl: i for i, lbl in enumerate(ordered)}
        parsed_value = ok
        source = "numeric parse of labels"

    # ---------------- assemble per-cell axis ----------------------------
    time_numeric = np.full(len(values), np.nan, dtype=float)
    for i, v in enumerate(values):
        if _is_missing(v):
            continue
        hv = v.item() if isinstance(v, np.generic) else v
        idx = index_of.get(hv)
        if idx is not None:
            time_numeric[i] = float(idx)

    n_missing = int(sum(_is_missing(v) for v in values))
    if n_missing:
        excluded = list(excluded) + [f"<missing/NaN> x{n_missing}"]

    ordering = TimeOrdering(
        time_numeric=time_numeric,
        order=ordered,
        index_of=index_of,
        excluded=list(excluded),
        source=source,
        parsed_value=parsed_value,
        parsed_label_value=parsed_label_value,
    )

    if not np.isfinite(ordering.time_numeric).any():
        raise TimeOrderError(
            f"Every cell was excluded from the time axis for column "
            f"'{key_name}'. Ordered labels {list(map(str, ordered))} match no cells."
        )

    if verbose:
        print(ordering.describe())

    return ordering
