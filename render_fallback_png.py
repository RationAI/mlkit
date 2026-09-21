#!/usr/bin/env python3
"""Render prov.dot's DOT output to PNG without the graphviz `dot` binary.

Stopgap for environments with no graphviz (the binary is a system package, not
pip-installable). It reads the .dot file provenance.py already writes when
`dot` is missing, so the *content* is exactly the document prov.dot emitted --
only the layout is approximated, by layering nodes along the DAG depth that
`rankdir=BT` implies. Not a graphviz replacement: no edge routing, no label
sizing, overlapping labels on wide layers.

Usage: python render_fallback_png.py output/prov-<name>.dot [out.png]
"""

from __future__ import annotations

import re
import sys
from collections import deque
from pathlib import Path

import matplotlib
import networkx as nx
import pydot


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


# fill by first id substring that matches; activity/agent/annotation/connector/
# model, anything else reads as a plain entity
KINDS: list[tuple[str, tuple[str, str]]] = [
    ("Run_", ("activity", "#9FB1FC")),
    ("run_", ("activity", "#9FB1FC")),
    ("user_", ("agent", "#B5EAD7")),
    ("annotation_", ("annotation", "#EEEEEE")),
    ("forward", ("connector", "#FFD6A5")),
    ("current", ("connector", "#FFD6A5")),
    ("model_", ("model", "#C77FA0")),
]
DEFAULT_KIND = ("entity", "#FCC2D7")
SKIP = {"node", "edge", "graph", ""}


def short(name: str) -> str:
    """Drop the namespace prefix and truncate long ids for a readable label."""
    s = re.sub(r"^(gen|blank|meta):", "", name)
    return s if len(s) <= 44 else s[:41] + "…"


def kind(name: str) -> tuple[str, str]:
    for needle, style in KINDS:
        if needle in name:
            return style
    return DEFAULT_KIND


def load(dot_path: str) -> nx.DiGraph:
    text = Path(dot_path).read_text(encoding="utf-8")
    graphs = pydot.graph_from_dot_data(text)
    if not graphs:
        raise SystemExit(f"pydot could not parse {dot_path}")

    # pydot's stubs type these accessors loosely (str | int | float | FrozenDict);
    # in a prov.dot file every object name is a quoted string, so str() is the
    # narrowing, not a workaround for a real mismatch.
    def qname(value: object) -> str:
        return str(value).strip('"')

    dg: nx.DiGraph = nx.DiGraph()
    for node in graphs[0].get_nodes():
        name = qname(node.get_name())
        if name not in SKIP:
            dg.add_node(name)
    for edge in graphs[0].get_edges():
        src, dst = qname(edge.get_source()), qname(edge.get_destination())
        if src not in SKIP and dst not in SKIP:
            dg.add_edge(src, dst)
    if dg.number_of_nodes() == 0:
        raise SystemExit(f"no nodes parsed from {dot_path}")
    return dg


def layer(dg: nx.DiGraph) -> dict[str, int]:
    """Layer index per node = longest path from any sink.

    prov.dot orients generation edges child -> parent (rankdir=BT), so sinks are
    the chain's outputs and layer 0 sits at the bottom of the drawing.
    """
    depth: dict[str, int] = {}
    queue = deque((n, 0) for n in dg if dg.out_degree(n) == 0)
    while queue:
        node, dist = queue.popleft()
        if depth.get(node, -1) >= dist:
            continue
        depth[node] = dist
        queue.extend((p, dist + 1) for p in dg.predecessors(node))
    for n in dg:
        depth.setdefault(n, 0)  # cycle member the relaxation could not reach
    return depth


def layout(dg: nx.DiGraph, depth: dict[str, int]) -> dict[str, tuple[float, float]]:
    pos: dict[str, tuple[float, float]] = {}
    for level in sorted(set(depth.values())):
        row = sorted(n for n in dg if depth[n] == level)
        for i, name in enumerate(row):
            pos[name] = (i / (len(row) - 1) if len(row) > 1 else 0.5, float(level))
    return pos


def main(argv: list[str]) -> int:
    if not 1 <= len(argv) <= 2:
        raise SystemExit(__doc__)
    dot_path = argv[0]
    default_out = (
        dot_path[: -len(".dot")] + ".png"
        if dot_path.endswith(".dot")
        else dot_path + ".png"
    )
    out = argv[1] if len(argv) == 2 else default_out

    dg = load(dot_path)
    pos = layout(dg, layer(dg))

    fig, ax = plt.subplots(figsize=(17, 11))
    for name, (x, y) in pos.items():
        key, colour = kind(name)
        ax.add_patch(
            Circle(
                (x, y),
                0.026,
                facecolor=colour,
                edgecolor="#444",
                lw=1.7 if key == "connector" else 0.9,
                zorder=3,
            )
        )
        ax.annotate(
            short(name),
            (x, y),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=6.0,
            zorder=4,
        )
    for src, dst in dg.edges():
        ax.annotate(
            "",
            xy=pos[dst],
            xytext=pos[src],
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#7a7a7a",
                "lw": 1.0,
                "shrinkA": 16,
                "shrinkB": 16,
                "connectionstyle": "arc3,rad=0.06",
            },
            zorder=2,
        )

    seen: set[str] = set()
    handles = []
    for name in dg:
        key, colour = kind(name)
        if key not in seen:
            seen.add(key)
            handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    ls="",
                    markerfacecolor=colour,
                    markeredgecolor="#444",
                    markersize=11,
                    label=key,
                )
            )
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title(
        f"{short(dot_path)} — {dg.number_of_nodes()} nodes / {dg.number_of_edges()} "
        "edges, from prov.dot output\n"
        "rendered WITHOUT graphviz: layout approximate, nodes and edges are the "
        "real document",
        fontsize=10,
    )
    ax.margins(0.13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=135)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
