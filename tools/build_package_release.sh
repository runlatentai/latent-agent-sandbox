#!/usr/bin/env bash
# Build the package's release files into dist/ and run the release gate on them: the wheel as `python -m build`
# makes it, the sdist repacked with every member owned by uid/gid 0 and no user, group or extended-attribute
# headers (setuptools writes the building user's name into the tar otherwise). Uploads nothing.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
out="${1:-$here/dist}"
cd "$here"
rm -rf "$out" build ./*.egg-info
python -m build --outdir "$out" >/dev/null
rm -rf build ./*.egg-info
python - "$out" <<'PY'
import glob, io, os, sys, tarfile
out = sys.argv[1]
for src in glob.glob(os.path.join(out, "*.tar.gz")):
    tmp = src + ".tmp"
    with tarfile.open(src, "r:gz") as t_in, tarfile.open(tmp, "w:gz", format=tarfile.PAX_FORMAT, compresslevel=9) as t_out:
        for m in sorted(t_in.getmembers(), key=lambda m: m.name):
            data = t_in.extractfile(m).read() if m.isfile() else None
            m.uid = m.gid = 0
            m.uname = m.gname = ""
            m.pax_headers = {}
            t_out.addfile(m, io.BytesIO(data) if data is not None else None)
    os.replace(tmp, src)
    print("repacked", os.path.basename(src))
PY
# GATE_ARGS: extra gate flags as one string, e.g. GATE_ARGS="--git-ok --release-email research@runlatent.ai" in CI
archives=(); for f in "$out"/*; do archives+=(--archive "$f"); done
# shellcheck disable=SC2086
python tools/release_gate.py --tree "$here" ${GATE_ARGS:-} "${archives[@]}"
ls -l "$out"
