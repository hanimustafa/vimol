# Multi-file CLI loading with auto-overlay (VIM-1, interactive scope)

Linear: [VIM-1](https://linear.app/vimol/issue/VIM-1/open-multiple-files-from-the-command-line),
project [Multi-file support](https://linear.app/vimol/project/multi-file-support-73ba5ed5e92c).
Builds on the approved data model in `docs/design/multi-structure.md` (§2, §3,
§4), specifically the "Consequences for app.py" list in §2 and the
`drawn_indices()` mark/overlay rule in §3.

## Summary

`vimol` currently accepts one file (`nargs="?"`). This change lets it accept
2+ files on the command line, loads all of them into one interactive session,
and automatically overlays them so the structures are visually comparable
without any extra keypresses. Each file may itself contain multiple models
(trajectory/NMR ensemble/multi-record SDF, already supported per-file since
VIM-9); the overlay defaults to showing the *first* model of each file, with
every other model still loaded and reachable.

**Also in this change, but unrelated to multi-file loading:** `--render`,
`--kitty`, and `--info` are removed from the CLI entirely (dead single-file
batch/embed/inspection flags — see "Removing `--render`/`--kitty`/`--info`"
below). This makes moot the design doc's §2 note about those flags
compositing the overlay for 2+ files; there is no longer a flag for that to
apply to.

**Out of scope for this change:** a `--active LABEL` flag; anything from
VIM-4 (alignment) or VIM-6 (cross-structure measurement) beyond what already
exists.

## CLI surface

`file` becomes `nargs="*"` (was `nargs="?"`).

- **Zero files** → today's bundled-C60 fallback, unchanged.
- **Exactly one file** → the existing code path runs completely unchanged:
  same error behavior (missing file: hard error, exit 2; unparseable file:
  hard error, exit 3), same handling for `--frame`.
- **2+ files, interactive mode** → the new path below.

## Removing `--render` / `--kitty` / `--info`

Requested independently of the multi-file work, in the same pass since it
touches the same file. All three are single-file-only batch/inspection
flags with no multi-file story of their own:

- `--render PNG` (still image to a file) and `--kitty` (one frame to stdout)
  are deleted, along with the `--size`/`--supersample` flags that existed
  only to configure them, and the now-dead `_parse_size` helper.
- `--info` (print structure info and exit) is deleted, along with
  `_print_info`.
- `--rotate` and `--frame` are unaffected — both are also used by the
  interactive path (initial camera orbit, initial active index) and stay.
- The `Scene.to_png()` / `Scene.to_kitty()` methods themselves, and the
  `vimol.Scene` library API documented in README's "Library usage" section,
  are untouched — this only removes the CLI flags, not the underlying
  capability for library callers (e.g. `examples/embed_demo.py`).
- README's `vimol protein.pdb --render out.png` example line is removed.
- The terminal-capability warning's "use --render out.png" suggestion is
  removed (`VIMOL_FORCE_KITTY=1` remains as the only suggested workaround).

## Loading 2+ files

For each file, in the order given:

1. If the path doesn't exist, or `load_all()` raises, or it parses to zero
   molecules: print `warning: skipping <path>: <reason>` to stderr and move
   on to the next file. This does not abort the session (VIM-1 acceptance
   criterion).
2. Otherwise, `ensure_bonds()` runs per model (respecting `--no-bonds` /
   `--bond-tolerance`, exactly as the single-file path does today), and every
   model from that file is appended to one shared `StructureSet`, in order.

If every file fails, `error: no molecules parsed`, exit 3 (same message the
single-file path already uses for an empty parse).

### Cross-file labels

Each file's label stem is its basename. If two files share a basename (e.g.
same filename from different directories), later ones get `~2`, `~3`, …
appended to the stem so labels stay unique. Within a file, multiple models
keep today's `#1`, `#2`, … suffixing. Example: `a.xyz` (3 models) and `b.xyz`
(1 model) produce labels `a.xyz#1`, `a.xyz#2`, `a.xyz#3`, `b.xyz`. Two files
both literally named `mol.xyz` produce `mol.xyz` and `mol.xyz~2`.

### `--frame`

Keeps working, now indexing into the whole loaded set in load order (a
strict superset of today's meaning, which only ever indexed one file's
models — unchanged for the single-file case). Clamped to
`[0, total_structures - 1]`. Default `0`, i.e. the first model of the first
file, consistent with `--frame`'s existing default.

## Auto-overlay

Once the `StructureSet` is built:

```python
viewer.structures.overlay = True
```

**Only the first-loaded structure of each file is marked** —
`entries[i].marked = True` for each file's first model, the rest of that
file's models are left unmarked. `StructureSet.drawn_indices()` already
resolves "overlay on + some entries marked" to *active + all marked*, so
this alone produces exactly "one structure per file, first model of each" in
the overlaid view — no changes needed to `drawn_indices()`, the composite
build, or the render path. Every other model stays loaded and reachable:
`n`/`p` or the digit keys cycle the active structure through the *entire*
set (all files, all models), the list strip shows every entry, and
`opt+click` on any row (including a non-first model of a file already in the
overlay) toggles it into/out of the overlay by the existing mechanism.

A single file with multiple internal models, opened alone, is unaffected:
`overlay` stays `False` and nothing gets auto-marked, exactly like today
(browse with `n`/`p`).

## Non-goals / explicit deferrals

- `--active LABEL` (readable multi-file equivalent of `--frame`).
- Any change to how a *single* multi-model file behaves when opened alone.

## Testing

- `app.make_parser()` accepts 2+ positional file args.
- `app.main([a, b])` with two valid single-model files: exits into the
  interactive path (non-tty test harness gets the existing exit-4 "needs a
  terminal" behavior, same as today's single-file interactive tests), and
  the `StructureSet` it builds (inspectable via a constructed `Viewer`, or by
  factoring the loader into a testable helper) has 2 entries, both labels
  distinct, `overlay is True`, and both entries `marked`.
- A file with 3 models + a single-model file: 4 entries total, labels
  `x.xyz#1/#2/#3` + `y.xyz`, only `x.xyz#1` and `y.xyz` marked.
- Two files sharing a basename: labels disambiguated with `~2`.
- One missing/unparseable file among 2+: the rest still load; stderr gets
  one `warning: skipping ...` line, and loading proceeds to the interactive
  path (same non-tty exit-4 guard as above) rather than aborting with an
  error exit code.
- All files missing/unparseable: exit 3, `error: no molecules parsed`.
- `--render`/`--kitty`/`--info` are no longer recognized flags:
  `make_parser().parse_args()` rejects them (argparse's own unrecognized-
  argument error).
- Single-file behavior (all existing tests) unchanged.
