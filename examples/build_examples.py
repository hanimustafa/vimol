"""Generate example molecule files with exact geometry (no external data)."""
import math
import os

HERE = os.path.dirname(__file__)


def write_xyz(name, comment, atoms):
    path = os.path.join(HERE, name)
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
    write_xyz("c60.xyz", "buckminsterfullerene C60", atoms)


if __name__ == "__main__":
    water()
    methane()
    benzene()
    buckyball()
