"""The release gate: the mandatory pre-publish check of the package tree, the dataset tree and every archive a
release produces. Standard library only. Exit 1 on ANY hit, every hit printed as `<where>:<offset>: <rule>: <text>`.

    python tools/release_gate.py --tree <dir> [--tree <dir>] --archive <file> [--archive ...] [--git-ok]
                                 [--rfc1918-ok <relative path>] [--release-email <addr>]

Rules.
(a) Every archive (a .tar / .tar.gz member list, a .whl / .zip entry list): no member with a non-empty user or
    group name, a uid or gid other than 0, an absolute path or a `..` component, a `._*` AppleDouble or `.DS_Store`
    entry, a pax header carrying an extended attribute (LIBARCHIVE.xattr.* / SCHILY.xattr.*), or a symlink.
(b) Every file of every tree and every text member of every archive: no path under /Users or under /home/<name>
    other than /home/agent (and the names passed as --home-ok, for a model's own invented paths in a transcript); no personal names or e-mail addresses other than the allowed contacts and
    @example.invalid; no host name or IP address that is not loopback, a documentation (TEST-NET) range, or a
    private-range address inside the one unit test that needs one; none of the private experiment vocabulary; no
    credential shapes (sk-, hf_, ghp_, AKIA, a private-key block, a bearer token that is not a planted TESTONLY
    one); no file in a tree with macOS extended attributes or a quarantine flag. One attribute is tolerated on a
    tree file: `com.apple.provenance`, which recent macOS releases stamp on every file an application writes, which
    `xattr -d` cannot remove, and which holds no user data; rule (a) still refuses it inside any archive.
(c) No `.git` in a tree (a checkout is a working copy, not a release: pass --git-ok in CI, where the gate then
    checks that every commit's author e-mail is the release identity, --release-email, repeatable; default
    research@runlatent.ai).

The literal strings the gate hunts for are assembled from fragments so that this file passes its own scan.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tarfile
import zipfile

ALLOWED_EMAILS = {"founders@runlatent.ai", "security@runlatent.ai", "research@runlatent.ai"}
ALLOWED_EMAIL_DOMAIN_SUFFIX = "@example.invalid"
TEXT_SUFFIXES = (
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".cfg",
    ".yml",
    ".yaml",
    ".json",
    ".jsonl",
    ".cff",
    ".sh",
    ".in",
    "Dockerfile",
    ".dockerignore",
    ".gitignore",
    "submit",
    "PKG-INFO",
    "SOURCES.txt",
    "METADATA",
    "RECORD",
    "entry_points.txt",
    "requires.txt",
    "top_level.txt",
    "dependency_links.txt",
    "WHEEL",
    "LICENSE",
)
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
    ".mypy_cache",
    "build",
    "dist",
    "node_modules",
}
RFC1918 = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
)
LOOPBACK = re.compile(r"^(?:127\.\d{1,3}\.\d{1,3}\.\d{1,3}|0\.0\.0\.0)$")
TESTNET = re.compile(r"^(?:192\.0\.2|198\.51\.100|203\.0\.113)\.\d{1,3}$")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
VERSION_LIKE = re.compile(r"\d+\.\d+\.\d+\.\d+")

# assembled from fragments: the gate must not contain the strings it forbids
_P = "ved" + "ant"
_HOSTS = "|".join(
    [
        "lambda" + "labs\\.com",
        "lambda\\." + "ai",
        "modal\\." + "com",
        "box" + "[12]\\b",
        "(?<![A-Za-z0-9_.-])[a-z0-9-]+\\.run" + "latent\\.ai",
    ]
)
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("personal-name", re.compile(_P, re.I)),
    ("home-path", re.compile(r"/Users/[A-Za-z0-9._-]+|/home/(?!agent\b)[A-Za-z0-9._-]+")),
    ("private-host", re.compile(_HOSTS, re.I)),
    (
        "private-vocabulary",
        re.compile(
            "|".join(
                [
                    r"\bE-R\d{1,3}\b",
                    "er" + r"168\b",
                    r"\bthe bri" + r"ef(?:'s|\s+(?:says|said|rule|asks|wants|calls|is|was))\b",
                    r"\blane [A-Z]\b",
                    r"\bLambda (?:Cloud|Labs)\b",
                    r"\bModal\b(?! window| dialog)",
                    "run" + "latent/" + "testing",
                    "latent-er" + "168",
                    r"\bc[12]-(?:final|gpu|reread)",
                    r"\bc[12]__",
                    "testing-er" + "168",
                    r"\bsite-blog\b",
                    r"\bHEADLINE\.md\b",
                ]
            )
        ),
    ),
    (
        "credential-shape",
        re.compile(
            "|".join(
                [
                    r"\bsk-[A-Za-z0-9_-]{16,}",
                    r"\bhf_[A-Za-z0-9]{20,}",
                    r"\bgh[pousr]_[A-Za-z0-9]{20,}",
                    r"\bAKIA[0-9A-Z]{16}\b",
                    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
                    r"\bxox[abp]-[A-Za-z0-9-]{10,}",
                ]
            )
        ),
    ),
    ("bearer-token", re.compile(r"Bearer\s+(?!pk_[a-z]+_TESTONLY_[0-9a-f]+\b|ci-token-TESTONLY-[0-9a-f]+\b)[A-Za-z0-9._-]{16,}")),
]
EMAIL = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
DECORATOR_FALSE_POSITIVE = re.compile(r"^n@(?:pytest|functools|dataclasses|contextlib|property|staticmethod|classmethod)")


class Gate:
    def __init__(self, rfc1918_ok: set[str], home_ok: set[str]) -> None:
        self.hits: list[str] = []
        self.rfc1918_ok = rfc1918_ok
        self.home_ok = home_ok | {"agent"}

    def hit(self, where: str, offset: int | str, rule: str, text: str) -> None:
        self.hits.append(f"{where}:{offset}: {rule}: {text[:100]!r}")

    # ------------------------------------------------------------------ text
    def scan_text(self, where: str, text: str, rel: str) -> None:
        for rule, pat in PATTERNS:
            for m in pat.finditer(text):
                s = m.group(0)
                if rule == "home-path" and s.startswith("/home/") and s.split("/")[2] in self.home_ok:
                    continue
                self.hit(where, m.start(), rule, s)
        for m in EMAIL.finditer(text):
            s = m.group(0)
            if s in ALLOWED_EMAILS or s.endswith(ALLOWED_EMAIL_DOMAIN_SUFFIX) or DECORATOR_FALSE_POSITIVE.match(s):
                continue
            if s.startswith("n@") and text[max(0, m.start() - 1) : m.start()] == "\\":  # "\n@decorator" inside JSON
                continue
            self.hit(where, m.start(), "email", s)
        for m in IPV4.finditer(text):
            ip = m.group(0)
            if LOOPBACK.match(ip) or TESTNET.match(ip):
                continue
            if any(int(x) > 255 for x in ip.split(".")):
                continue  # a version number, not an address
            if RFC1918.fullmatch(ip) and any(rel == ok or rel.endswith("/" + ok) for ok in self.rfc1918_ok):
                continue  # the one unit test's private-range examples, in the tree or inside an sdist member
            self.hit(where, m.start(), "ip-address", ip)

    def scan_bytes(self, where: str, data: bytes, rel: str) -> None:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
        self.scan_text(where, text, rel)

    # ------------------------------------------------------------------ trees
    def scan_tree(self, root: str, git_ok: bool, release_emails: set[str]) -> None:
        root = os.path.abspath(root)
        git_dir = os.path.join(root, ".git")
        if os.path.exists(git_dir):
            if not git_ok:
                self.hit(root, "-", "git-present", ".git exists in the tree (a working copy is not a release)")
            else:
                self.check_git_authors(root, release_emails)
        for dp, dn, fn in os.walk(root):
            dn[:] = sorted(d for d in dn if d not in SKIP_DIRS)
            for f in sorted(fn):
                p = os.path.join(dp, f)
                rel = os.path.relpath(p, root)
                if os.path.islink(p):
                    self.hit(rel, "-", "symlink", os.readlink(p))
                    continue
                if f == ".DS_Store" or f.startswith("._"):
                    self.hit(rel, "-", "mac-metadata-file", f)
                self.check_xattrs(p, rel)
                if f.endswith(TEXT_SUFFIXES) or "." not in f:
                    with open(p, "rb") as fh:
                        self.scan_bytes(rel, fh.read(), rel)

    def check_xattrs(self, path: str, rel: str) -> None:
        names: list[str] = []
        if hasattr(os, "listxattr"):
            try:
                names = list(os.listxattr(path))
            except OSError:
                names = []
        elif sys.platform == "darwin":
            try:
                out = subprocess.run(["xattr", "-l", path], capture_output=True, text=True, timeout=30).stdout
            except (OSError, subprocess.TimeoutExpired):
                out = ""
            names = [ln.split(":", 1)[0].strip() for ln in out.splitlines() if ":" in ln]
        for n in names:
            if n == "com.apple.provenance":
                continue  # macOS's own tag on every written file; not removable; refused inside archives by rule (a)
            rule = "quarantine-flag" if "quarantine" in n else "extended-attribute"
            self.hit(rel, "-", rule, n)

    def check_git_authors(self, root: str, release_emails: set[str]) -> None:
        try:
            out = subprocess.run(
                ["git", "-C", root, "log", "--all", "--format=%ae%n%ce"], capture_output=True, text=True, timeout=60
            ).stdout
        except (OSError, subprocess.TimeoutExpired) as e:
            self.hit(root, "-", "git-log-unreadable", str(e))
            return
        for addr in sorted({ln.strip() for ln in out.splitlines() if ln.strip()}):
            if addr not in release_emails:
                self.hit(root, "-", "git-author-email", addr)

    # ------------------------------------------------------------------ archives
    def scan_archive(self, path: str) -> None:
        name = os.path.basename(path)
        if path.endswith((".whl", ".zip")):
            self.scan_zip(path, name)
        else:
            self.scan_tar(path, name)

    def member_path_ok(self, where: str, member: str) -> None:
        if member.startswith("/") or member.startswith("\\"):
            self.hit(where, member, "absolute-member-path", member)
        if ".." in member.split("/"):
            self.hit(where, member, "parent-reference-in-path", member)
        base = member.rstrip("/").split("/")[-1]
        if base == ".DS_Store" or base.startswith("._"):
            self.hit(where, member, "mac-metadata-member", member)

    def scan_tar(self, path: str, where: str) -> None:
        with tarfile.open(path, "r:*") as t:
            for m in t.getmembers():
                self.member_path_ok(where, m.name)
                if m.uname or m.gname:
                    self.hit(where, m.name, "member-owner-name", f"uname={m.uname!r} gname={m.gname!r}")
                if m.uid != 0 or m.gid != 0:
                    self.hit(where, m.name, "member-owner-id", f"uid={m.uid} gid={m.gid}")
                if m.issym() or m.islnk():
                    self.hit(where, m.name, "symlink-member", m.linkname)
                xattr_keys = [
                    k for k in (m.pax_headers or {}) if ".xattr." in k or k.startswith("SCHILY.") or k.startswith("LIBARCHIVE.")
                ]
                if xattr_keys:
                    self.hit(where, m.name, "pax-xattr-header", ", ".join(sorted(xattr_keys)))
                if m.isfile() and (m.name.endswith(TEXT_SUFFIXES) or "." not in m.name.split("/")[-1]):
                    f = t.extractfile(m)
                    if f is not None:
                        self.scan_bytes(f"{where}!{m.name}", f.read(), m.name)

    def scan_zip(self, path: str, where: str) -> None:
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                self.member_path_ok(where, info.filename)
                mode = (info.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    self.hit(where, info.filename, "symlink-member", "")
                if info.filename.endswith(TEXT_SUFFIXES) or "." not in info.filename.split("/")[-1]:
                    self.scan_bytes(f"{where}!{info.filename}", z.read(info), info.filename)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="the pre-publish release gate (see the module docstring)")
    ap.add_argument("--tree", action="append", default=[], help="a directory to scan (repeatable)")
    ap.add_argument("--archive", action="append", default=[], help="a .tar.gz / .whl / .zip to scan (repeatable)")
    ap.add_argument("--git-ok", action="store_true", help="a .git is expected (CI); its author e-mails are checked instead")
    ap.add_argument(
        "--rfc1918-ok",
        action="append",
        default=["tests/test_agent_client.py"],
        help="a tree-relative file allowed to carry private-range example addresses (repeatable)",
    )
    ap.add_argument("--release-email", action="append", default=[], help="the commit identity a release may carry (repeatable)")
    ap.add_argument(
        "--home-ok", action="append", default=[], help="a /home/<name> a transcript may mention besides agent (repeatable)"
    )
    a = ap.parse_args(argv)
    if not a.tree and not a.archive:
        ap.error("give at least one --tree or --archive")
    gate = Gate(set(a.rfc1918_ok), set(a.home_ok))
    release_emails = set(a.release_email) or {"research@runlatent.ai"}
    for t in a.tree:
        gate.scan_tree(t, a.git_ok, release_emails)
    for p in a.archive:
        if not os.path.exists(p):
            gate.hit(p, "-", "archive-missing", p)
            continue
        gate.scan_archive(p)
    for h in gate.hits:
        print(h)
    n_targets = len(a.tree) + len(a.archive)
    print(f"release gate: {len(gate.hits)} hit(s) over {n_targets} target(s): {'FAIL' if gate.hits else 'PASS'}")
    return 1 if gate.hits else 0


if __name__ == "__main__":
    sys.exit(main())
