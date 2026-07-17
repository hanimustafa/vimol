# Shared helpers for the release scripts. Sourced by build-pypi-artifacts and
# publish-pypi-artifacts -- not run directly.
#
# The version lives in ONE place: __version__ in src/vimol/__init__.py.
# pyproject.toml reads it dynamically (tool.setuptools.dynamic), so bumping
# only ever edits that single line and the two can never drift.
set -euo pipefail

# Callers set $here (their own scripts/ dir) before sourcing us; derive the repo
# root from it. Avoids depending on BASH_SOURCE, which isn't set when a non-bash
# shell sources this file.
: "${here:?lib.sh must be sourced after setting \$here to the scripts/ dir}"
ROOT="$(git -C "$here" rev-parse --show-toplevel)"
VERSION_FILE="$ROOT/src/vimol/__init__.py"
RELEASE_VENV="$ROOT/.venv-release"

# Current version string, read from the single source of truth.
read_version() {
    python3 - "$VERSION_FILE" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', s, re.M)
if not m:
    sys.exit("no __version__ assignment found in " + sys.argv[1])
print(m.group(1))
PY
}

# Rewrite the __version__ line in place. $1 = new version string.
set_version() {
    python3 - "$VERSION_FILE" "$1" <<'PY'
import re, sys
path, new = sys.argv[1], sys.argv[2]
s = open(path).read()
s2, n = re.subn(r'^(__version__\s*=\s*)["\'][^"\']*["\']',
                r'\g<1>"%s"' % new, s, count=1, flags=re.M)
if n != 1:
    sys.exit("could not rewrite __version__ in " + path)
open(path, "w").write(s2)
PY
}

# Increment a semantic version. $1 = "X.Y.Z", $2 = major|minor|patch. Echoes new.
bump_version() {
    local cur="$1" level="$2" a b c
    IFS=. read -r a b c <<<"$cur"
    case "$level" in
        major) a=$((a + 1)); b=0; c=0 ;;
        minor) b=$((b + 1)); c=0 ;;
        patch) c=$((c + 1)) ;;
        *) echo "unknown bump level '$level' (use major|minor|patch)" >&2; return 1 ;;
    esac
    echo "$a.$b.$c"
}

# Create the self-contained release venv (build + twine) if absent. It lives at
# .venv-release/ (gitignored) so releasing never touches your working env.
ensure_release_venv() {
    if [ ! -x "$RELEASE_VENV/bin/python" ]; then
        echo "==> creating release venv at $RELEASE_VENV (build + twine)"
        python3 -m venv "$RELEASE_VENV"
        "$RELEASE_VENV/bin/pip" install --quiet --upgrade pip build twine
    fi
}

# Exit 0 iff version $1 is already published on index $2 (pypi|testpypi).
published_on_index() {
    local ver="$1" idx="$2" host code
    case "$idx" in
        pypi) host="https://pypi.org" ;;
        testpypi) host="https://test.pypi.org" ;;
        *) echo "unknown index '$idx'" >&2; return 2 ;;
    esac
    # no -f: a 404 (not published) is a normal answer here, not a curl error.
    code="$(curl -s -o /dev/null -w '%{http_code}' "$host/pypi/vimol/$ver/json" || echo 000)"
    [ "$code" = "200" ]
}
