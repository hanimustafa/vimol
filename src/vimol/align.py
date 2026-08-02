"""Fast rigid and permutation-invariant molecular alignment.

The global search follows the published rotation/assignment strategy described
by Finkler and Goedecker, J. Chem. Phys. 152, 164106 (2020).  Rigid fits use
Kabsch's proper-rotation solution.  This is an independent MIT-licensed
implementation; no code from the GPL ``RMSD-finder`` program is included.

Coordinates use vimol's row-vector convention::

    aligned = positions @ rotation.T + translation

``subset=True`` solves the useful asymmetric problem: every atom in ``mobile``
is matched, but only a chemically compatible subset of ``reference`` is used.
The returned mapping contains the chosen reference indices.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .molecule import Molecule
from .bonds import perceive_bonds
from .structures import AlignmentResult, Transform

try:  # A substantial acceleration for the opt-in permutation paths.
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
except ImportError:  # pragma: no cover - exercised in a subprocess test
    _scipy_lsa = None


_TWO_PI = 2.0 * np.pi


def _points(value, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 2 or out.shape[1:] != (3,):
        raise ValueError("%s must have shape (N, 3), got %r" % (name, out.shape))
    if not np.all(np.isfinite(out)):
        raise ValueError("%s contains NaN or infinity" % name)
    return np.ascontiguousarray(out)


def kabsch(P, Q) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return the proper rigid transform taking corresponding ``P`` onto ``Q``.

    The result is ``(rotation, translation, rmsd)`` and is valid for the
    row-vector expression ``P @ rotation.T + translation``.  Reflections are
    never admitted.  One- and two-point inputs are supported.
    """
    P = _points(P, "P")
    Q = _points(Q, "Q")
    if P.shape != Q.shape:
        raise ValueError("P and Q must have the same shape")
    n = P.shape[0]
    if n == 0:
        raise ValueError("at least one point is required")

    pc = P.mean(axis=0)
    qc = Q.mean(axis=0)
    p0 = P - pc
    q0 = Q - qc
    covariance = p0.T @ q0
    u, _singular, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = qc - pc @ rotation.T
    delta = P @ rotation.T + translation - Q
    rmsd = float(np.sqrt(np.einsum("ij,ij->", delta, delta) / n))
    return rotation, translation, rmsd


def _linear_sum_assignment_numpy(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Shortest augmenting-path assignment for a rectangular rows<=cols matrix.

    This compact fallback is O(rows**2 * cols), needs only O(cols) workspace,
    and is used only when scipy is unavailable.  It is the standard primal-dual
    shortest augmenting path formulation, independently implemented here.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost must be a matrix")
    n, m = cost.shape
    if n > m:
        raise ValueError("assignment requires at least as many columns as rows")
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    # 1-indexed arrays make the augmenting-path update considerably clearer.
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        j0 = 0
        while True:
            used[j0] = True
            i0 = p[j0]
            unused = np.flatnonzero(~used[1:]) + 1
            cur = cost[i0 - 1, unused - 1] - u[i0] - v[unused]
            better = cur < minv[unused]
            changed = unused[better]
            minv[changed] = cur[better]
            way[changed] = j0
            rel = int(np.argmin(minv[unused]))
            delta = minv[unused[rel]]
            j1 = int(unused[rel])
            matched = p[used]
            u[matched] += delta
            v[np.flatnonzero(used)] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    cols = np.empty(n, dtype=np.int64)
    occupied = np.flatnonzero(p[1:]) + 1
    cols[p[occupied] - 1] = occupied - 1
    return np.arange(n, dtype=np.int64), cols


def _lsa(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if _scipy_lsa is not None:
        return _scipy_lsa(cost)
    return _linear_sum_assignment_numpy(cost)


def _symbol_blocks(mobile_symbols: Sequence[str], reference_symbols: Sequence[str],
                   *, need: str = "mobile", have: str = "reference"
                   ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Pair up same-element atoms, requiring *have* to cover *need*.

    ``need``/``have`` name the two sides in the raised message. Callers that
    search in the inverted direction -- passing a reference selection as the
    mobile side -- must say so, or the error names the wrong structure.
    """
    p_symbols = np.asarray(mobile_symbols, dtype=object)
    q_symbols = np.asarray(reference_symbols, dtype=object)
    blocks = []
    # dict.fromkeys preserves molecule order and makes tie handling deterministic.
    for symbol in dict.fromkeys(mobile_symbols):
        pi = np.flatnonzero(p_symbols == symbol)
        qi = np.flatnonzero(q_symbols == symbol)
        if len(qi) < len(pi):
            raise ValueError(
                "%s has %d %s atom(s), but %s requires %d"
                % (have, len(qi), symbol, need, len(pi)))
        blocks.append((pi, qi))
    return blocks


def _squared_distances(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    # The dot-product form saves a temporary (n,m,3) tensor.
    d = ((P * P).sum(axis=1)[:, None] + (Q * Q).sum(axis=1)[None, :]
         - 2.0 * (P @ Q.T))
    np.maximum(d, 0.0, out=d)
    return d


def _assign_points(P: np.ndarray, Q: np.ndarray,
                   blocks: Sequence[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    mapping = np.empty(len(P), dtype=np.int64)
    for pi, qi in blocks:
        cost = _squared_distances(P[pi], Q[qi])
        rows, cols = _lsa(cost)
        mapping[pi[rows]] = qi[cols]
    return mapping


def _refine(P: np.ndarray, Q: np.ndarray, blocks, rotation: np.ndarray,
            translation: np.ndarray, max_iterations: int = 30
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    old = None
    mapping = None
    for _ in range(max_iterations):
        moved = P @ rotation.T + translation
        mapping = _assign_points(moved, Q, blocks)
        rotation, translation, rmsd = kabsch(P, Q[mapping])
        if old is not None and np.array_equal(mapping, old):
            return rotation, translation, mapping, rmsd
        old = mapping.copy()
    return rotation, translation, mapping, rmsd


def _quaternion_rotations(count: int) -> np.ndarray:
    """Deterministic, low-discrepancy rotations covering SO(3)."""
    if count < 1:
        raise ValueError("trials must be at least 1")
    i = np.arange(count, dtype=np.float64) + 0.5
    u = i / count
    # Two unrelated irrational increments form a deterministic Hopf sequence.
    v = np.mod(i * 0.6180339887498949, 1.0)
    w = np.mod(i * 0.7548776662466927, 1.0)
    a = _TWO_PI * v
    b = _TWO_PI * w
    q = np.column_stack((
        np.sqrt(1.0 - u) * np.sin(a),
        np.sqrt(1.0 - u) * np.cos(a),
        np.sqrt(u) * np.sin(b),
        np.sqrt(u) * np.cos(b),
    ))
    x, y, z, r = q.T
    rotations = np.empty((count, 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotations[:, 0, 1] = 2 * (x * y - z * r)
    rotations[:, 0, 2] = 2 * (x * z + y * r)
    rotations[:, 1, 0] = 2 * (x * y + z * r)
    rotations[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotations[:, 1, 2] = 2 * (y * z - x * r)
    rotations[:, 2, 0] = 2 * (x * z - y * r)
    rotations[:, 2, 1] = 2 * (y * z + x * r)
    rotations[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rotations


def _rotation_lower_bounds(P: np.ndarray, Q: np.ndarray, rotations: np.ndarray,
                           blocks, chunk_bytes: int = 48 * 1024 * 1024
                           ) -> np.ndarray:
    """Relaxed (non-injective) element-aware assignment costs per rotation."""
    n, m = len(P), len(Q)
    bytes_per = max(1, n * m * 4 + n * 3 * 4)
    chunk = max(1, min(len(rotations), chunk_bytes // bytes_per))
    p32 = P.astype(np.float32, copy=False)
    q32 = Q.astype(np.float32, copy=False)
    qn = np.einsum("ij,ij->i", q32, q32)
    out = np.empty(len(rotations), dtype=np.float64)
    for start in range(0, len(rotations), chunk):
        stop = min(start + chunk, len(rotations))
        moved = np.einsum("ij,bkj->bik", p32, rotations[start:stop].astype(np.float32))
        total = np.zeros(stop - start, dtype=np.float64)
        for pi, qi in blocks:
            pp = moved[:, pi]
            dist = (np.einsum("bik,bik->bi", pp, pp)[:, :, None]
                    + qn[qi][None, None, :]
                    - 2.0 * np.einsum("bik,jk->bij", pp, q32[qi]))
            total += np.min(dist, axis=2).sum(axis=1)
        out[start:stop] = total
    return out


def _candidate_indices(bounds: np.ndarray, candidates: int) -> np.ndarray:
    candidates = max(1, min(int(candidates), len(bounds)))
    n_top = (candidates + 1) // 2
    if n_top == len(bounds):
        top = np.arange(len(bounds))
    else:
        top = np.argpartition(bounds, n_top - 1)[:n_top]
    top = top[np.argsort(bounds[top], kind="stable")]
    spread = np.linspace(0, len(bounds) - 1, candidates - n_top, dtype=np.int64)
    # Preserve bound order, then fill collisions from the next-best rotations.
    chosen = list(dict.fromkeys(np.concatenate((top, spread)).tolist()))
    if len(chosen) < candidates:
        for idx in np.argsort(bounds, kind="stable"):
            if int(idx) not in chosen:
                chosen.append(int(idx))
                if len(chosen) == candidates:
                    break
    return np.asarray(chosen, dtype=np.int64)


def permutation_search(P, Q, symbols: Sequence[str], trials: int = 2000,
                       candidates: int = 64,
                       reference_symbols: Optional[Sequence[str]] = None
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Find a low-RMSD proper rotation and element-preserving permutation.

    Returns ``(rotation, translation, mapping, rmsd)``.  Equal-size structures
    are centered before the vectorized SO(3) screen; use :func:`subset_search`
    for a larger reference.
    """
    P = _points(P, "P")
    Q = _points(Q, "Q")
    if len(P) != len(Q):
        raise ValueError("permutation_search requires equal atom counts")
    if len(P) == 0:
        raise ValueError("at least one atom is required")
    reference_symbols = list(symbols if reference_symbols is None else reference_symbols)
    if Counter(symbols) != Counter(reference_symbols):
        raise ValueError("mobile and reference must have identical element counts")
    if int(trials) < 1:
        raise ValueError("trials must be at least 1")
    blocks = _symbol_blocks(symbols, reference_symbols)
    pc = P.mean(axis=0)
    qc = Q.mean(axis=0)
    p0 = P - pc
    q0 = Q - qc

    direct_rotation, _direct_t, _direct_rmsd = kabsch(P, Q)
    if list(symbols) == reference_symbols and _direct_rmsd <= 1e-12:
        # A zero-RMSD legal index mapping is already the global lower bound.
        # This makes identical/rigid trajectory frames as fast as plain
        # Kabsch even though the caller requested full RMSD-finder semantics.
        return (direct_rotation, _direct_t, np.arange(len(P), dtype=np.int64),
                _direct_rmsd)
    rotations = _quaternion_rotations(int(trials))
    # Include identity and a direct index Kabsch seed at essentially zero cost.
    rotations[0] = np.eye(3)
    if len(rotations) > 1:
        rotations[1] = direct_rotation
    bounds = _rotation_lower_bounds(p0, q0, rotations, blocks)
    indices = _candidate_indices(bounds, int(candidates))
    if 0 not in indices:
        indices[-1] = 0
    if len(rotations) > 1 and 1 not in indices:
        indices[-2 if len(indices) > 1 else -1] = 1

    best = None
    for idx in indices:
        rotation, translation0, mapping, rmsd = _refine(
            p0, q0, blocks, rotations[idx], np.zeros(3))
        translation = qc + translation0 - pc @ rotation.T
        result = (rotation, translation, mapping, rmsd)
        if best is None or rmsd < best[3]:
            best = result
            if rmsd <= 1e-12:
                break
    return best


def _anchor_indices(P: np.ndarray) -> Tuple[int, int, int]:
    """Choose a stable, high-area triangle from the query."""
    distances = _squared_distances(P, P)
    a, b = np.unravel_index(int(np.argmax(distances)), distances.shape)
    axis = P[b] - P[a]
    norm2 = float(axis @ axis)
    if norm2 <= 1e-24:
        return 0, min(1, len(P) - 1), min(2, len(P) - 1)
    rel = P - P[a]
    perpendicular = rel - np.outer((rel @ axis) / norm2, axis)
    area = np.einsum("ij,ij->i", perpendicular, perpendicular)
    area[[a, b]] = -1.0  # anchors are atom indices: the third must be distinct
    c = int(np.argmax(area))
    return int(a), int(b), c


def _frame_rotation(P: np.ndarray, Q: np.ndarray) -> Optional[np.ndarray]:
    def frame(X):
        x = X[1] - X[0]
        nx = np.linalg.norm(x)
        if nx <= 1e-12:
            return None
        x = x / nx
        y = X[2] - X[0]
        y = y - x * (y @ x)
        ny = np.linalg.norm(y)
        if ny <= 1e-10:
            return None
        y = y / ny
        z = np.cross(x, y)
        return np.column_stack((x, y, z))
    fp = frame(P)
    fq = frame(Q)
    return None if fp is None or fq is None else fq @ fp.T


def _anchor_seed_transforms(P: np.ndarray, Q: np.ndarray,
                            mobile_symbols: Sequence[str],
                            reference_symbols: Sequence[str], pool: int
                            ) -> Tuple[np.ndarray, np.ndarray]:
    a, b, c = _anchor_indices(P)
    q_symbols = np.asarray(reference_symbols, dtype=object)
    aa = np.flatnonzero(q_symbols == mobile_symbols[a])
    bb = np.flatnonzero(q_symbols == mobile_symbols[b])
    cc = np.flatnonzero(q_symbols == mobile_symbols[c])
    if not len(aa) or not len(bb) or not len(cc):
        raise ValueError("reference does not contain the required element subset")

    dab = np.linalg.norm(P[a] - P[b])
    dac = np.linalg.norm(P[a] - P[c])
    dbc = np.linalg.norm(P[b] - P[c])
    pair_d = np.linalg.norm(Q[aa, None, :] - Q[bb][None, :, :], axis=2)
    pair_error = np.abs(pair_d - dab)
    if mobile_symbols[a] == mobile_symbols[b]:
        pair_error[aa[:, None] == bb[None, :]] = np.inf
    take = min(max(pool * 4, 32), pair_error.size)
    flat = np.argpartition(pair_error.ravel(), take - 1)[:take]
    flat = flat[np.argsort(pair_error.ravel()[flat], kind="stable")]

    rotations = []
    translations = []
    seen = set()
    for item in flat:
        ia, ib = np.unravel_index(int(item), pair_error.shape)
        qa, qb = int(aa[ia]), int(bb[ib])
        err_c = (np.abs(np.linalg.norm(Q[cc] - Q[qa], axis=1) - dac)
                 + np.abs(np.linalg.norm(Q[cc] - Q[qb], axis=1) - dbc))
        if mobile_symbols[c] == mobile_symbols[a]:
            err_c[cc == qa] = np.inf
        if mobile_symbols[c] == mobile_symbols[b]:
            err_c[cc == qb] = np.inf
        for ic in np.argsort(err_c, kind="stable")[:2]:
            qc = int(cc[ic])
            key = (qa, qb, qc)
            if key in seen or not np.isfinite(err_c[ic]):
                continue
            seen.add(key)
            p_anchor = P[[a, b, c]]
            q_anchor = Q[[qa, qb, qc]]
            rotation = _frame_rotation(p_anchor, q_anchor)
            if rotation is None:
                rotation, _t, _rmsd = kabsch(p_anchor, q_anchor)
            translation = q_anchor.mean(axis=0) - p_anchor.mean(axis=0) @ rotation.T
            rotations.append(rotation)
            translations.append(translation)
    if not rotations:
        rotation, translation, _rmsd = kabsch(P[:min(3, len(P))], Q[:min(3, len(P))])
        rotations.append(rotation)
        translations.append(translation)
    return np.asarray(rotations), np.asarray(translations)


def _transform_lower_bounds(P: np.ndarray, Q: np.ndarray, rotations: np.ndarray,
                            translations: np.ndarray, blocks) -> np.ndarray:
    out = np.zeros(len(rotations), dtype=np.float64)
    # Keep peak memory modest for query-against-large-target searches.
    n, m = len(P), len(Q)
    chunk = max(1, min(len(rotations), (48 * 1024 * 1024) // max(1, n * m * 8)))
    qn = np.einsum("ij,ij->i", Q, Q)
    for start in range(0, len(rotations), chunk):
        stop = min(start + chunk, len(rotations))
        moved = (np.einsum("ij,bkj->bik", P, rotations[start:stop])
                 + translations[start:stop, None, :])
        total = np.zeros(stop - start)
        for pi, qi in blocks:
            pp = moved[:, pi]
            dist = (np.einsum("bik,bik->bi", pp, pp)[:, :, None]
                    + qn[qi][None, None, :]
                    - 2.0 * np.einsum("bik,jk->bij", pp, Q[qi]))
            total += np.min(dist, axis=2).sum(axis=1)
        out[start:stop] = total
    return out


def subset_search(P, Q, mobile_symbols: Sequence[str],
                  reference_symbols: Sequence[str], candidates: int = 64,
                  *, need: str = "mobile", have: str = "reference"
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Match every mobile atom to an element-compatible subset of reference.

    The search uses distance-invariant, high-area triangle anchors to generate
    translations and rotations, screens them in one vectorized pass, then runs
    exact rectangular assignment + Kabsch refinement on the best basins.

    ``need``/``have`` name the two sides in raised messages. Some callers
    search in the inverted direction -- ``P`` is a reference selection being
    located inside the whole mobile ``Q`` -- and must rename them to match.
    """
    P = _points(P, "P")
    Q = _points(Q, "Q")
    if not len(P):
        raise ValueError("at least one %s atom is required" % need)
    if len(P) > len(Q):
        raise ValueError("subset search requires %s to be no larger than %s"
                         % (need, have))
    blocks = _symbol_blocks(mobile_symbols, reference_symbols,
                            need=need, have=have)
    candidates = max(1, int(candidates))

    if len(P) == 1:
        qi = blocks[0][1][0]
        rotation = np.eye(3)
        translation = Q[qi] - P[0]
        return rotation, translation, np.array([qi], dtype=np.int64), 0.0
    if len(P) == 2:
        # All chemically legal target pairs; distance is the complete rigid
        # invariant for two points, so the best pair is globally optimal.
        choices0 = np.flatnonzero(np.asarray(reference_symbols) == mobile_symbols[0])
        choices1 = np.flatnonzero(np.asarray(reference_symbols) == mobile_symbols[1])
        d = np.abs(np.linalg.norm(Q[choices0, None] - Q[choices1], axis=2)
                   - np.linalg.norm(P[0] - P[1]))
        if mobile_symbols[0] == mobile_symbols[1]:
            d[choices0[:, None] == choices1[None, :]] = np.inf
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        mapping = np.array([choices0[i], choices1[j]], dtype=np.int64)
        rotation, translation, rmsd = kabsch(P, Q[mapping])
        return rotation, translation, mapping, rmsd

    rotations, translations = _anchor_seed_transforms(
        P, Q, mobile_symbols, reference_symbols, pool=max(candidates, 32))
    bounds = _transform_lower_bounds(P, Q, rotations, translations, blocks)
    indices = np.argsort(bounds, kind="stable")[:min(candidates, len(bounds))]
    best = None
    for idx in indices:
        result = _refine(P, Q, blocks, rotations[idx], translations[idx])
        if best is None or result[3] < best[3]:
            best = result
            if result[3] <= 1e-12:
                break
    return best


def _selection(selection, n: int, name: str) -> np.ndarray:
    if selection is None:
        return np.arange(n, dtype=np.int64)
    if isinstance(selection, slice):
        return np.arange(n, dtype=np.int64)[selection]
    raw = np.asarray(selection)
    if raw.dtype == bool:
        if raw.ndim != 1 or len(raw) != n:
            raise ValueError("%s boolean mask must have length %d" % (name, n))
        out = np.flatnonzero(raw)
    else:
        if raw.ndim != 1:
            raise ValueError("%s must be a one-dimensional index sequence" % name)
        out = raw.astype(np.int64)
        if not np.all(raw == out):
            raise ValueError("%s contains a non-integer index" % name)
    if len(out) == 0:
        raise ValueError("%s must select at least one atom" % name)
    if np.any(out < 0) or np.any(out >= n):
        raise IndexError("%s index is out of range" % name)
    if len(np.unique(out)) != len(out):
        raise ValueError("%s contains duplicate indices" % name)
    return np.ascontiguousarray(out, dtype=np.int64)


def _topology_subset_indices(mobile: Molecule, reference: Molecule,
                             ref_select, max_matches: int = 512,
                             allow_boundary_truncation: bool = False
                             ) -> Optional[np.ndarray]:
    """Return a bond-preserving mobile match for a connected reference pick.

    Pure element/geometry matching can assemble a deceptively low-RMSD set
    from unrelated atoms in a large molecule. This bounded graph search uses
    the picked induced bond graph plus each atom's heavy-neighbor signature.
    Chemical graphs have tiny degree, so peptide/backbone selections complete
    in effectively linear time. Disconnected manual selections return ``None``
    and retain the general geometric fallback.
    """
    ref_indices = _selection(ref_select, reference.n_atoms, "ref_select")
    k = len(ref_indices)
    if k < 3:
        return None

    def adjacency(molecule: Molecule):
        out = [set() for _ in range(molecule.n_atoms)]
        for a, b, _order in (molecule.bonds or perceive_bonds(molecule)):
            out[a].add(b)
            out[b].add(a)
        return out

    radj = adjacency(reference)
    madj = adjacency(mobile)
    ref_local = {int(atom): i for i, atom in enumerate(ref_indices)}
    pattern = [set() for _ in range(k)]
    for i, atom in enumerate(ref_indices):
        pattern[i].update(ref_local[j] for j in radj[int(atom)] if j in ref_local)

    # Only a connected selection carries enough topology to improve on the
    # geometric matcher. Isolated/manual point clouds deliberately fall back.
    seen = {0}
    stack = [0]
    while stack:
        i = stack.pop()
        for j in pattern[i]:
            if j not in seen:
                seen.add(j)
                stack.append(j)
    if len(seen) != k:
        return None

    def heavy_signature(molecule: Molecule, adj, atom: int):
        counts = Counter(molecule.symbols[j] for j in adj[atom]
                         if molecule.symbols[j] != "H")
        return counts

    mobile_signatures = [heavy_signature(mobile, madj, i)
                         for i in range(mobile.n_atoms)]
    candidates = []
    reference_signatures = []
    for i, atom in enumerate(ref_indices):
        required = heavy_signature(reference, radj, int(atom))
        reference_signatures.append(required)
        if allow_boundary_truncation and len(pattern[i]) <= 1:
            # A maximal shared segment may end in the middle of the larger
            # reference chain. Require only the neighbor that remains inside
            # the segment; the smaller mobile legitimately lacks what was cut.
            required = Counter(reference.symbols[int(ref_indices[j])]
                               for j in pattern[i])
        symbol = reference.symbols[int(atom)]
        legal = [j for j, candidate_symbol in enumerate(mobile.symbols)
                 if candidate_symbol == symbol
                 and len(madj[j]) >= len(pattern[i])
                 and all(mobile_signatures[j][element] >= count
                         for element, count in required.items())]
        if not legal:
            return None
        candidates.append(legal)

    # Backbone selections are paths. Walk only bonded neighbors instead of
    # repeatedly scanning every same-element atom in the molecule; this turns
    # the common peptide case from combinatorial backtracking into a tiny,
    # degree-bounded traversal.
    edge_count = sum(map(len, pattern)) // 2
    if edge_count == k - 1 and max(map(len, pattern)) <= 2:
        endpoint = next(i for i, adjacent in enumerate(pattern) if len(adjacent) == 1)
        path = []
        previous = -1
        current = endpoint
        while current >= 0:
            path.append(current)
            following = [j for j in pattern[current] if j != previous]
            previous, current = current, (following[0] if following else -1)
        strict_candidates = [
            [atom for atom in items
             if mobile_signatures[atom] == reference_signatures[i]]
            for i, items in enumerate(candidates)]

        def search_path(path_candidates):
            candidate_sets = [set(items) for items in path_candidates]
            assigned = np.full(k, -1, dtype=np.int64)
            used = set()
            best_indices = None
            best_rmsd = np.inf
            completed = 0

            def walk(depth: int, atom: int) -> None:
                nonlocal best_indices, best_rmsd, completed
                node = path[depth]
                assigned[node] = atom
                used.add(atom)
                if depth + 1 == k:
                    completed += 1
                    _r, _t, rmsd = kabsch(
                        mobile.positions[assigned], reference.positions[ref_indices])
                    if rmsd < best_rmsd:
                        best_rmsd = rmsd
                        best_indices = assigned.copy()
                elif completed < max_matches:
                    next_node = path[depth + 1]
                    legal_next = candidate_sets[next_node]
                    for neighbor in madj[atom]:
                        if neighbor not in used and neighbor in legal_next:
                            walk(depth + 1, neighbor)
                            if completed >= max_matches:
                                break
                used.remove(atom)
                assigned[node] = -1

            for start in path_candidates[endpoint]:
                walk(0, start)
                if completed >= max_matches:
                    break
            return best_indices

        # Exact local chemistry normally identifies chain termini immediately.
        # Fall back to signature containment for fragments embedded in a
        # larger chain, where a terminal atom legitimately gains a neighbor.
        if all(strict_candidates):
            strict_match = search_path(strict_candidates)
            if strict_match is not None:
                return strict_match
        return search_path(candidates)

    assigned = np.full(k, -1, dtype=np.int64)
    used = set()
    best_indices = None
    best_rmsd = np.inf
    completed = 0

    def visit() -> None:
        nonlocal best_indices, best_rmsd, completed
        if completed >= max_matches:
            return
        unassigned = [i for i in range(k) if assigned[i] < 0]
        if not unassigned:
            completed += 1
            mobile_indices = assigned.copy()
            _r, _t, rmsd = kabsch(
                mobile.positions[mobile_indices], reference.positions[ref_indices])
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_indices = mobile_indices
            return

        # Grow outward from already mapped atoms. Candidate scarcity breaks
        # ties and keeps even large proteins comfortably below the match cap.
        node = min(
            unassigned,
            key=lambda i: (-sum(assigned[j] >= 0 for j in pattern[i]),
                           len(candidates[i]), -len(pattern[i]), i))
        mapped_neighbors = [j for j in pattern[node] if assigned[j] >= 0]
        for atom in candidates[node]:
            if atom in used:
                continue
            if any(int(assigned[j]) not in madj[atom] for j in mapped_neighbors):
                continue
            # Cheap forward check: every not-yet-mapped bonded pattern atom
            # must still have at least one legal neighbor available.
            viable = True
            for neighbor in pattern[node]:
                if assigned[neighbor] >= 0:
                    continue
                if not any(candidate not in used and candidate != atom
                           and candidate in madj[atom]
                           for candidate in candidates[neighbor]):
                    viable = False
                    break
            if not viable:
                continue
            assigned[node] = atom
            used.add(atom)
            visit()
            used.remove(atom)
            assigned[node] = -1
            if completed >= max_matches:
                break

    visit()
    return best_indices


def _largest_topology_subset(mobile: Molecule, reference: Molecule,
                             ref_select, minimum: int = 3
                             ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Largest connected path segment of ``ref_select`` found in ``mobile``.

    Used when a selected peptide backbone is longer than an overlay molecule.
    The longest viable bond-preserving segment wins; RMSD breaks ties between
    equally long locations. Returns ``(mobile_indices, reference_indices)``.
    """
    ref_indices = _selection(ref_select, reference.n_atoms, "ref_select")
    if len(ref_indices) < minimum:
        return None
    adjacency = [set() for _ in range(reference.n_atoms)]
    for a, b, _order in (reference.bonds or perceive_bonds(reference)):
        adjacency[a].add(b)
        adjacency[b].add(a)
    local = {int(atom): i for i, atom in enumerate(ref_indices)}
    pattern = [set(local[j] for j in adjacency[int(atom)] if j in local)
               for atom in ref_indices]
    if sum(map(len, pattern)) // 2 != len(ref_indices) - 1:
        return None
    if max(map(len, pattern), default=0) > 2:
        return None
    endpoints = [i for i, neighbors in enumerate(pattern) if len(neighbors) == 1]
    if len(endpoints) != 2:
        return None
    order = []
    previous = -1
    current = endpoints[0]
    while current >= 0:
        order.append(current)
        following = [j for j in pattern[current] if j != previous]
        previous, current = current, (following[0] if following else -1)
    path = ref_indices[np.asarray(order, dtype=np.int64)]

    available = Counter(mobile.symbols)
    max_length = min(len(path), mobile.n_atoms)
    for length in range(max_length, minimum - 1, -1):
        best = None
        best_rmsd = np.inf
        for start in range(0, len(path) - length + 1):
            segment = np.ascontiguousarray(path[start:start + length])
            needed = Counter(reference.symbols[int(i)] for i in segment)
            if any(count > available[element] for element, count in needed.items()):
                continue
            mobile_indices = _topology_subset_indices(
                mobile, reference, segment, max_matches=128,
                allow_boundary_truncation=True)
            if mobile_indices is None:
                continue
            _rotation, _translation, rmsd = kabsch(
                mobile.positions[mobile_indices], reference.positions[segment])
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best = (mobile_indices, segment)
        if best is not None:
            return best
    return None


def superpose(mobile: Molecule, reference: Molecule, *, select=None,
              ref_select=None, permute: bool = False, subset: bool = False,
              trials: int = 2000, candidates: int = 64,
              permute_max_atoms: Optional[int] = 300) -> AlignmentResult:
    """Align ``mobile`` onto ``reference`` and return a reproducible result.

    ``select``/``ref_select`` provide known index correspondence and are the
    fastest path.  ``permute=True`` discovers correspondence for equal-size
    molecules.  ``subset=True`` discovers which atoms of a larger reference
    match the complete mobile molecule; it implies permutation matching.
    """
    if not isinstance(mobile, Molecule) or not isinstance(reference, Molecule):
        raise TypeError("mobile and reference must be Molecule instances")
    P_all = _points(mobile.positions, "mobile.positions")
    Q_all = _points(reference.positions, "reference.positions")

    explicit = select is not None or ref_select is not None
    if explicit:
        psel = _selection(select, len(P_all), "select")
        qsel = _selection(ref_select if ref_select is not None else select,
                          len(Q_all), "ref_select")
        if len(psel) != len(qsel):
            raise ValueError("select and ref_select must have the same length")
        psymbols = [mobile.symbols[i] for i in psel]
        qsymbols = [reference.symbols[i] for i in qsel]
        if permute or subset:
            if Counter(psymbols) != Counter(qsymbols):
                raise ValueError("selected atoms must have identical element counts")
            rotation, translation, local_map, rmsd = permutation_search(
                P_all[psel], Q_all[qsel], psymbols, trials, candidates, qsymbols)
            paired_q = qsel[local_map]
            method = "permute"
        else:
            if psymbols != qsymbols:
                raise ValueError("corresponding selected atoms must have identical elements")
            rotation, translation, rmsd = kabsch(P_all[psel], Q_all[qsel])
            paired_q = qsel
            method = "subset"
        mapping = np.full(len(P_all), -1, dtype=np.int64)
        mapping[psel] = paired_q
        return AlignmentResult(
            rmsd=rmsd, n_fitted=len(psel),
            transform=Transform(rotation=rotation, translation=translation),
            ref_label=reference.name, method=method, select=psel,
            ref_select=paired_q.copy(), mapping=mapping)

    if subset:
        if permute_max_atoms is not None and len(P_all) > permute_max_atoms:
            raise ValueError("subset permutation search is impractical above %d mobile atoms"
                             % permute_max_atoms)
        rotation, translation, mapping, rmsd = subset_search(
            P_all, Q_all, mobile.symbols, reference.symbols, candidates=candidates)
        method = "subset"
    elif permute:
        if len(P_all) != len(Q_all):
            raise ValueError("different atom counts require subset=True")
        if permute_max_atoms is not None and len(P_all) > permute_max_atoms:
            raise ValueError(
                "permutation search is impractical above %d atoms; use index "
                "correspondence or an explicit atom subset" % permute_max_atoms)
        rotation, translation, mapping, rmsd = permutation_search(
            P_all, Q_all, mobile.symbols, trials, candidates, reference.symbols)
        method = "permute"
    else:
        if mobile.symbols != reference.symbols:
            raise ValueError("index alignment requires identical atom symbols in the same order")
        rotation, translation, rmsd = kabsch(P_all, Q_all)
        mapping = np.arange(len(P_all), dtype=np.int64)
        method = "index"

    return AlignmentResult(
        rmsd=rmsd, n_fitted=len(P_all),
        transform=Transform(rotation=rotation, translation=translation),
        ref_label=reference.name, method=method, mapping=mapping)


def superpose_to_reference_subset(mobile: Molecule, reference: Molecule,
                                  ref_select, *, candidates: int = 64,
                                  permute_max_atoms: Optional[int] = 300
                                  ) -> AlignmentResult:
    """Align ``mobile`` by finding the atoms matching a picked reference subset.

    This is the interactive ``R`` operation.  The user only has to pick atoms
    on the untinted reference.  Corresponding atoms are discovered inside the
    mobile molecule, and the inverse of that fragment match moves the *whole*
    mobile molecule onto the reference.
    """
    if not isinstance(mobile, Molecule) or not isinstance(reference, Molecule):
        raise TypeError("mobile and reference must be Molecule instances")
    ref_indices = _selection(ref_select, reference.n_atoms, "ref_select")
    if permute_max_atoms is not None and len(ref_indices) > permute_max_atoms:
        raise ValueError("subset permutation search is impractical above %d selected atoms"
                         % permute_max_atoms)
    query = reference.positions[ref_indices]
    query_symbols = [reference.symbols[i] for i in ref_indices]
    # The search runs inverted here -- the picked reference atoms are the
    # thing being located, and the whole mobile is what must accommodate them
    # -- so the two sides are renamed to keep raised messages truthful.
    rotation_qm, translation_qm, ref_to_mobile, rmsd = subset_search(
        query, mobile.positions, query_symbols, mobile.symbols,
        candidates=candidates, need="the reference selection", have="mobile")
    query_to_mobile = Transform(rotation=rotation_qm, translation=translation_qm)
    mobile_to_reference = query_to_mobile.inverse()
    mapping = np.full(mobile.n_atoms, -1, dtype=np.int64)
    mapping[ref_to_mobile] = ref_indices
    return AlignmentResult(
        rmsd=rmsd, n_fitted=len(ref_indices), transform=mobile_to_reference,
        ref_label=reference.name, method="subset", select=ref_to_mobile.copy(),
        ref_select=ref_indices.copy(), mapping=mapping)


def superpose_between_subsets(mobile: Molecule, reference: Molecule,
                              mobile_select, ref_select, *, candidates: int = 64,
                              permute_max_atoms: Optional[int] = 300
                              ) -> AlignmentResult:
    """Fit a smaller mobile subset into a larger selected reference subset.

    This is the complementary direction to
    :func:`superpose_to_reference_subset`: it is used when the main-frame
    selection contains terminal atoms that do not exist in the mobile
    structure. The whole mobile molecule still receives the fitted transform.
    """
    if not isinstance(mobile, Molecule) or not isinstance(reference, Molecule):
        raise TypeError("mobile and reference must be Molecule instances")
    mobile_indices = _selection(mobile_select, mobile.n_atoms, "mobile_select")
    ref_indices = _selection(ref_select, reference.n_atoms, "ref_select")
    if len(mobile_indices) > len(ref_indices):
        raise ValueError("mobile subset must be no larger than reference subset")
    if permute_max_atoms is not None and len(mobile_indices) > permute_max_atoms:
        raise ValueError("subset permutation search is impractical above %d selected atoms"
                         % permute_max_atoms)
    mobile_symbols = [mobile.symbols[i] for i in mobile_indices]
    reference_symbols = [reference.symbols[i] for i in ref_indices]
    rotation, translation, local_mapping, rmsd = subset_search(
        mobile.positions[mobile_indices], reference.positions[ref_indices],
        mobile_symbols, reference_symbols, candidates=candidates)
    paired_reference = ref_indices[local_mapping]
    mapping = np.full(mobile.n_atoms, -1, dtype=np.int64)
    mapping[mobile_indices] = paired_reference
    return AlignmentResult(
        rmsd=rmsd, n_fitted=len(mobile_indices),
        transform=Transform(rotation=rotation, translation=translation),
        ref_label=reference.name, method="subset", select=mobile_indices.copy(),
        ref_select=paired_reference.copy(), mapping=mapping)


__all__ = [
    "kabsch", "permutation_search", "subset_search", "superpose",
    "superpose_to_reference_subset", "superpose_between_subsets",
]
