"""Generate example molecule files with exact geometry (no external data)."""
import math
import os

HERE = os.path.dirname(__file__)
PKG_DATA = os.path.join(HERE, "..", "src", "vimol", "data")


def write_xyz(name, comment, atoms, dirs=(HERE,)):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, name)
        with open(path, "w") as f:
            f.write(f"{len(atoms)}\n{comment}\n")
            for sym, (x, y, z) in atoms:
                f.write(f"{sym:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")
        print("wrote", path, len(atoms), "atoms")


def water():
    r, half = 0.9584, math.radians(104.5 / 2)
    atoms = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (r * math.sin(half), -r * math.cos(half), 0.0)),
        ("H", (-r * math.sin(half), -r * math.cos(half), 0.0)),
    ]
    write_xyz("water.xyz", "water", atoms)


def methane():
    d = 1.087 / math.sqrt(3)
    dirs = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    atoms = [("C", (0.0, 0.0, 0.0))]
    atoms += [("H", (x * d, y * d, z * d)) for (x, y, z) in dirs]
    write_xyz("methane.xyz", "methane", atoms)


def benzene():
    rc, rh = 1.39, 1.39 + 1.09
    atoms = []
    for k in range(6):
        a = math.radians(60 * k)
        atoms.append(("C", (rc * math.cos(a), rc * math.sin(a), 0.0)))
    for k in range(6):
        a = math.radians(60 * k)
        atoms.append(("H", (rh * math.cos(a), rh * math.sin(a), 0.0)))
    write_xyz("benzene.xyz", "benzene", atoms)


def buckyball():
    phi = (1 + math.sqrt(5)) / 2
    bases = [
        (0.0, 1.0, 3 * phi),
        (2.0, 1 + 2 * phi, phi),
        (1.0, 2 + phi, 2 * phi),
    ]
    pts = set()
    for bx, by, bz in bases:
        vals = [bx, by, bz]
        # all sign combinations
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    t = (sx * vals[0], sy * vals[1], sz * vals[2])
                    # even (cyclic) permutations
                    for perm in ((0, 1, 2), (2, 0, 1), (1, 2, 0)):
                        p = (t[perm[0]], t[perm[1]], t[perm[2]])
                        pts.add(tuple(round(c, 6) for c in p))
    scale = 1.46 / 2.0  # edge length of construction is 2 A -> C-C ~1.46
    atoms = [("C", (x * scale, y * scale, z * scale)) for (x, y, z) in sorted(pts)]
    write_xyz("c60.xyz", "buckminsterfullerene C60", atoms, dirs=(HERE, PKG_DATA))


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a):
    n = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    return (a[0] / n, a[1] / n, a[2] / n)


def _perp(u):
    """Some unit vector perpendicular to *u*."""
    ref = (0.0, 0.0, 1.0) if abs(u[2]) < 0.9 else (1.0, 0.0, 0.0)
    return _norm(_cross(u, ref))


def _open_tetrahedral(taken):
    """Unit directions completing a tetrahedron around a centre whose bonds
    already point along *taken*. Returns 4 - len(taken) of them."""
    a = math.radians(109.4712)
    if not taken:
        d = 1.0 / math.sqrt(3)
        return [(d, d, d), (d, -d, -d), (-d, d, -d), (-d, -d, d)]
    if len(taken) == 1:
        u = _norm(taken[0])
        p, q = _perp(u), _norm(_cross(u, _perp(u)))
        out = []
        for k in range(3):
            t = 2 * math.pi * k / 3
            radial = _add(_scale(p, math.cos(t)), _scale(q, math.sin(t)))
            out.append(_norm(_add(_scale(u, math.cos(a)),
                                  _scale(radial, math.sin(a)))))
        return out
    if len(taken) == 2:
        u1, u2 = _norm(taken[0]), _norm(taken[1])
        bisect = _norm(_add(u1, u2))
        plane = _norm(_cross(u1, u2))
        half = a / 2
        return [_norm(_add(_scale(bisect, -math.cos(half)),
                           _scale(plane, s * math.sin(half))))
                for s in (1, -1)]
    total = taken[0]
    for t in taken[1:]:
        total = _add(total, t)
    return [_norm(_scale(total, -1))]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _volume(centre, p, q, r):
    """Signed volume of the three bonds centre->p, ->q, ->r. Its sign is the
    handedness of that centre, which is how the two stereocentres below are
    checked."""
    return _dot(_cross(_sub(p, centre), _sub(q, centre)), _sub(r, centre))


def _branch(centre, anchor, ref, length, angle_deg, phases):
    """Positions of atoms bonded to *centre*, each at *angle_deg* off the
    centre-anchor bond and spun about it by one of *phases* degrees, where
    phase 0 is anti-periplanar to *ref* (dihedral ref-anchor-centre-atom of
    180 degrees). Every rotor in the molecule is placed this way, so each one
    is pinned to a named atom rather than to whatever an arbitrary
    perpendicular happened to be."""
    u = _norm(_sub(anchor, centre))
    away = _sub(ref, anchor)
    radial_ref = _sub(away, _scale(u, _dot(away, u)))
    zero = _norm(_scale(radial_ref, -1))
    side = _norm(_cross(u, zero))
    a = math.radians(angle_deg)
    out = []
    for ph in phases:
        t = math.radians(ph)
        radial = _add(_scale(zero, math.cos(t)), _scale(side, math.sin(t)))
        d = _add(_scale(u, math.cos(a)), _scale(radial, math.sin(a)))
        out.append(_add(centre, _scale(_norm(d), length)))
    return out


# Threonine's bonds, in the order _threonine_at builds the atoms. Written out
# because the conformer search below scores contacts by how many bonds apart
# two atoms are, and a builder knows its own connectivity exactly.
_THR_BONDS = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7), (6, 8),
              (5, 9), (5, 10), (5, 11), (2, 12), (2, 13), (3, 14), (3, 15),
              (15, 16)]


def _threonine_at(chi1_phase):
    """L-threonine with the side chain at one of its three staggered rotamers.

    Both stereocentres are set by measurement rather than by assertion: the
    two signed volumes checked below are the ones the threonines of a real
    L-protein (chignolin, PDB 5AWL, residues 6 and 8) give, and each centre's
    two possible assignments are tried until the sign matches. Every rotor is
    pinned by dihedral to a named neighbour -- see :func:`_branch`."""
    CC, CN, CH, CO, OH, NH = 1.540, 1.469, 1.090, 1.427, 0.967, 1.014
    C_CARB, C_O_DOUBLE, C_O_SINGLE = 1.525, 1.213, 1.340
    TETRA = 109.4712

    atoms = []          # (symbol, position), in output order

    def put(sym, pos):
        atoms.append((sym, pos))
        return pos

    ca = put("C", (0.0, 0.0, 0.0))                       # 0  alpha carbon
    cb = put("C", (CC, 0.0, 0.0))                        # 1  beta carbon

    # Alpha carbon. The nitrogen takes the first open direction; the carboxyl
    # carbon and H-alpha take the other two, and which way round is exactly
    # the difference between L and D.
    open_ca = _open_tetrahedral([_sub(cb, ca)])
    n = _add(ca, _scale(open_ca[0], CN))
    i, j = 1, 2
    if _volume(ca, n, _add(ca, _scale(open_ca[i], C_CARB)), cb) < 0:
        i, j = 2, 1
    put("N", n)                                          # 2
    c_carb = put("C", _add(ca, _scale(open_ca[i], C_CARB)))   # 3
    put("H", _add(ca, _scale(open_ca[j], CH)))           # 4  H-alpha

    # Beta carbon, the second stereocentre. Its tripod is spun to *chi1_phase*
    # about the C-alpha bond, which is the side chain's chi-1 torsion; the
    # three substituents are then assigned in whichever of the two orders
    # gives the (3R) sign.
    tripod = [chi1_phase, chi1_phase + 120, chi1_phase + 240]
    og = _branch(cb, ca, n, CO, TETRA, tripod[:1])[0]
    cg_a, hb_a = _branch(cb, ca, n, CC, TETRA, tripod[1:2])[0], \
        _branch(cb, ca, n, CH, TETRA, tripod[2:3])[0]
    cg_b, hb_b = _branch(cb, ca, n, CC, TETRA, tripod[2:3])[0], \
        _branch(cb, ca, n, CH, TETRA, tripod[1:2])[0]
    cg, hb = (cg_a, hb_a) if _volume(cb, ca, og, cg_a) > 0 else (cg_b, hb_b)
    put("C", cg)                                         # 5  gamma carbon
    put("O", og)                                         # 6  hydroxyl oxygen
    put("H", hb)                                         # 7  H-beta

    # Side-chain hydroxyl, anti to the alpha carbon. This hydrogen is the one
    # demo C grows into a methyl.
    put("H", _branch(og, cb, ca, OH, 108.5, [0])[0])     # 8

    # Gamma methyl, one hydrogen anti to the hydroxyl oxygen (staggered).
    for p in _branch(cg, cb, og, CH, TETRA, [0, 120, 240]):
        put("H", p)                                      # 9, 10, 11

    # Amine. The lone pair takes the position anti to the beta carbon and the
    # two hydrogens flank it.
    for p in _branch(n, ca, cb, NH, TETRA, [120, 240]):
        put("H", p)                                      # 12, 13

    # Carboxyl: trigonal planar, carbonyl oxygen anti to the nitrogen. The
    # three angles at the carbon sum to 360, as a planar centre's must.
    o_double = put("O", _branch(c_carb, ca, n, C_O_DOUBLE, 125.3, [0])[0])    # 14
    o_single = put("O", _branch(c_carb, ca, n, C_O_SINGLE, 111.6, [180])[0])  # 15
    # Acid hydrogen, syn to the carbonyl -- the Z conformer every neutral
    # carboxylic acid adopts.
    put("H", _branch(o_single, c_carb, o_double, OH, 106.0, [180])[0])        # 16

    return atoms


def _closest_contact(atoms, bonds):
    """Shortest distance between two atoms three or more bonds apart -- the
    number that says whether a conformer has anything jammed into anything."""
    adj = {i: set() for i in range(len(atoms))}
    for i, j in bonds:
        adj[i].add(j)
        adj[j].add(i)
    worst = float("inf")
    for i in range(len(atoms)):
        near, front = {i}, {i}
        for _ in range(2):                    # atoms one or two bonds away
            front = {k for f in front for k in adj[f]} - near
            near |= front
        for j in range(i + 1, len(atoms)):
            if j not in near:
                worst = min(worst, math.dist(atoms[i][1], atoms[j][1]))
    return worst


def threonine():
    """L-threonine (2S,3R), in the roomiest of its three staggered side-chain
    rotamers -- 'roomiest' meaning the largest distance between any two atoms
    that are not within two bonds of each other, which is a rule rather than a
    judgement call and so gives the same file every time."""
    best = max((_threonine_at(p) for p in (0, 120, 240)),
               key=lambda atoms: _closest_contact(atoms, _THR_BONDS))
    write_xyz("threonine.xyz", "L-threonine", best)


if __name__ == "__main__":
    water()
    methane()
    benzene()
    buckyball()
    threonine()
