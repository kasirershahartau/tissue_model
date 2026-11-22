"""Choosing which hair cells the mechanical fit ablates.

The SI ablates "4 chosen cells (same cells for each simulation)". Fixed
INDICES on every array do not do that -- index 337 is a different cell on each
array, and often not an HC, while the experiment always ablates an HC. So the
cells are chosen PER PATTERN instead: four HCs far enough apart that the four
holes and their rings of neighbours do not interact, so the experiment's
single-HC ablation is repeated four times per simulation.

They are chosen ONCE per pattern -- from the initial array and its delta
levels, before any mechanics -- and identified by ``id``, the persistent cell
identity. Every grid point therefore ablates the same four cells, so the
ablation term does not pick up a second source of noise across the landscape.

Separation is measured in CONTACT HOPS (the graph distance between cells),
not in microns: it is periodicity-safe by construction (the contact graph has
no boundary) and needs no geometry. It is a CONSTRAINT, not an objective: HCs
are visited in a seeded random order and each is taken if it is at least
``min_separation`` hops from every one already taken. Maximising the
separation instead was tried first and is biased: hops spread more slowly
from a cell with few neighbours, so the farthest-apart cells are
systematically the low-coordination ones (7 of the 12 picked on three
verification patterns had <= 4 neighbours, against a few percent of all
cells). A random HC under a separation constraint is what an experimenter
choosing an HC to ablate does. The seed is derived from the array's name
(:func:`pattern_seed`) and recorded with the choice, so the same pattern
always yields the same cells.

HC NEIGHBOURS. An ablated HC usually has few or no HC neighbours: in the
experiment 6 of the 56 cells around the E17.5 ablations were HCs (one of the
three ablations had none, and that repeat is dropped from the HC/SC ratio),
and 15 of 88 at P0. The cells are NOT chosen to have HC neighbours -- that
would bias the sample towards HC clusters, unlike the experiment -- so a
simulation can likewise end up with none, and its ratio is then undefined and
dropped, exactly as the experimental repeat is. :func:`neighbour_counts`
reports how many each choice has.
"""
from collections import deque

import numpy as np


def cell_adjacency(sheet):
    """``{face label: set(neighbour face labels)}`` from the half-edge twins.

    Built from ``edge_df.opposite`` (the twin half-edge; -1 when there is
    none) rather than ``VirtualSheet.get_neighbors``, which queries edge_df
    once per face. Faces with no edges still appear, with no neighbours.
    """
    edges = sheet.edge_df
    adj = {int(f): set() for f in sheet.face_df.index}
    opp = edges["opposite"].to_numpy()
    has_twin = opp >= 0
    own = edges["face"].to_numpy()[has_twin]
    other = edges.loc[opp[has_twin], "face"].to_numpy()
    for a, b in zip(own, other):
        if a != b:
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))
    return adj


def _hops_from(adj, source):
    """Contact-hop distance from ``source`` to every reachable face."""
    dist = {source: 0}
    queue = deque([source])
    while queue:
        f = queue.popleft()
        for g in adj[f]:
            if g not in dist:
                dist[g] = dist[f] + 1
                queue.append(g)
    return dist


def pattern_seed(name):
    """A stable seed from an array's name (crc32; Python's ``hash`` of a str
    changes between processes)."""
    import zlib
    return int(zlib.crc32(str(name).encode("utf-8")))


def choose_well_separated(adj, candidates, n=4, min_separation=5, seed=0):
    """``n`` labels from ``candidates``, pairwise at least ``min_separation``
    contact hops apart, taken greedily in a random order fixed by ``seed``.

    3 hops is the least that keeps the chosen cells' neighbour rings disjoint
    and not touching; the default 5 also keeps the rings' own neighbours
    apart. If the first random order dead-ends (rare: the arrays are ~22
    cells across), further orders from the same seed are tried before giving
    up with ``ValueError``.
    """
    candidates = sorted(int(c) for c in candidates)
    if len(candidates) < n:
        raise ValueError("only %d candidate cell(s), %d needed"
                         % (len(candidates), n))
    rng = np.random.default_rng(seed)
    for _attempt in range(100):
        chosen, hops = [], {}
        for c in rng.permutation(candidates):
            c = int(c)
            if all(hops[s].get(c, len(adj) + 1) >= min_separation
                   for s in chosen):
                chosen.append(c)
                hops[c] = _hops_from(adj, c)
                if len(chosen) == n:
                    return sorted(chosen)
    raise ValueError("no %d candidate cells found >= %d hop(s) apart"
                     % (n, min_separation))


def neighbour_counts(adj, chosen, is_hc):
    """``[(n_HC_neighbours, n_SC_neighbours)]`` for each chosen label;
    ``is_hc`` maps a label to True/False."""
    return [(sum(bool(is_hc[g]) for g in adj[c]),
             sum(not bool(is_hc[g]) for g in adj[c])) for c in chosen]


def choose_ablation_hcs(sheet, delta, threshold, n=4, min_separation=5,
                        seed=0):
    """The ``id`` values of ``n`` well-separated HCs of one pattern.

    ``sheet`` is the pattern's initial array; ``delta`` is one delta level
    per face of it, in face_df ROW order; a cell is an HC when its delta is
    ABOVE ``threshold`` -- the same rule the fit classifies cells by, and the
    p_D of Eqs. 16-17. Cells already tagged for removal (``type == -1``) are
    never chosen.

    Returns ``(ids, info)`` where ``info`` carries the chosen labels, the
    seed, the minimum separation reached and each cell's HC/SC neighbour
    counts.
    """
    face_df = sheet.face_df
    delta = np.asarray(delta, dtype=float)
    if delta.shape != (len(face_df),):
        raise ValueError("delta has %d entries for %d faces"
                         % (delta.size, len(face_df)))
    labels = face_df.index.to_numpy(int)
    hc = delta > threshold
    alive = (face_df["type"].to_numpy() != -1 if "type" in face_df.columns
             else np.ones(len(face_df), bool))
    adj = cell_adjacency(sheet)
    chosen = choose_well_separated(adj, labels[hc & alive], n=n,
                                   min_separation=min_separation, seed=seed)
    is_hc = dict(zip(labels.tolist(), hc.tolist()))
    hops = [_hops_from(adj, a).get(b) for i, a in enumerate(chosen)
            for b in chosen[i + 1:]]
    ids = [int(face_df.at[c, "id"]) for c in chosen]
    return ids, {"labels": chosen, "ids": ids, "seed": int(seed),
                 "min_separation": int(min_separation),
                 "min_separation_hops": int(min(hops)),
                 "neighbours_hc_sc": neighbour_counts(adj, chosen, is_hc),
                 "n_hc": int(hc.sum()), "n_cells": int(len(face_df))}


def labels_for_ids(face_df, ids):
    """Translate ``id`` values to the face labels of ``face_df``'s frame.

    Returns ``(labels, missing_ids)``. A cell can be missing when it
    delaminated before this frame; the caller decides whether that is fatal.
    """
    by_id = {int(i): int(lab) for lab, i in zip(face_df.index, face_df["id"])}
    labels = [by_id[i] for i in ids if i in by_id]
    missing = [int(i) for i in ids if i not in by_id]
    return labels, missing


def ablation_hcs_for_initial(initial_name, threshold, n=4, min_separation=5):
    """:func:`choose_ablation_hcs` on an initial array as the fit loads it:
    its (single) recorded frame, with delta from the ``delta_levels.npy``
    sidecar beside it, which is keyed by ``unique_id``, and the seed
    :func:`pattern_seed` of the array's name.

    Returns the cells as the RUNS made from this array will identify them
    (their row labels here -- see the note in the body), not the extract's
    own ``id`` column."""
    import os
    from post_processing import load_history_file
    from history_io import RESULTS_DIR
    path = os.path.join(RESULTS_DIR, initial_name, "delta_levels.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "no delta_levels.npy beside %s: the ablated HCs are chosen from "
            "the pattern, so the lateral-inhibition sidecars must exist first"
            % initial_name)
    history = load_history_file(initial_name)
    sheet = history.retrieve(float(np.min(history.time_stamps)))
    sheet.arrange_sheet_from_history()
    by_uid = np.load(path)
    uid = sheet.face_df["unique_id"].to_numpy(int)
    if uid.max() >= by_uid.size:
        raise ValueError("delta_levels.npy beside %s has %d entries but the "
                         "array has unique_id up to %d"
                         % (initial_name, by_uid.size, uid.max()))
    ids, info = choose_ablation_hcs(sheet, by_uid[uid], threshold, n=n,
                                    min_separation=min_separation,
                                    seed=pattern_seed(initial_name))
    # The ids a RUN will know these cells by are NOT the extract's ``id``
    # column: InnerEarModel re-bases ``id`` to the row label on every
    # construction, and in 13 of the 20 initial arrays the stored column is a
    # permutation of the labels (it carries the array build's numbering). A
    # run started from this array therefore calls each cell by its row label
    # here, so that is what is returned. (Handing over the stored column
    # ablated the wrong cells -- found on the P0 verification arrays.)
    info["extract_id_column"] = ids
    info["ids"] = list(info["labels"])
    return list(info["labels"]), info
