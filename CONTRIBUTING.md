# Contributing

Thanks for looking. This is a small research harness; contributions that keep it small are the most welcome.

## Ground rules

- **Observe and detect only.** Every unauthorized path in a scenario is a planted, harmless shortcut that needs
  nothing beyond a directory listing or a file read. A pull request that adds a real exploit, credential
  guessing, a real wallet or key, or anything that trains, prompts or otherwise helps a model get better at
  cheating or concealing will be closed. Hint rungs and peer notes say where a resource is; they never tell the
  agent to use it (`tests/test_registry.py` and `tests/test_peer_notes.py` pin the wording; keep them green).
- **Labels come from the sandbox's logs, never from the agent's text** (the one exception is the concealment
  rule over the final report). A new label needs a log event behind it and a test that shows the event.
- **Every template must be a real task**: `tests/test_generator.py::test_every_template_is_a_real_task` checks
  that the reference passes its own cases, that the hidden expected values are not all one value, and that the
  shipped stub fails the public tests the reference passes. Run it after touching a generator.
- **The sandbox stays offline.** `--network none`, mock services on loopback, no network at run time inside the
  image. The Dockerfile pins its base image by digest and every pip install by hash; the apt packages (curl, git)
  are version-unpinned, so the image is reproducible up to what Debian's repository serves on the build day.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
ruff check latent_agent_sandbox tests
ruff format --check latent_agent_sandbox tests    # CI enforces both; `ruff format` rewrites in place
python -m pytest tests          # the sandbox tests skip without a Docker daemon
```

With a Docker daemon the suite builds the image once and runs the scripted agent through every scenario.
`DOCKER_HOST=unix:///nonexistent/docker.sock python -m pytest tests` is the no-Docker run CI uses.

## The release gate (mandatory before anything is published)

`tools/release_gate.py` (standard library only) is the last step before a wheel, an sdist or the dataset tarball
leaves a machine, and CI runs it on every push. It fails on: an archive member with a user or group name, a non-zero
uid/gid, an absolute or `..` path, a `._*` / `.DS_Store` entry, an extended-attribute pax header or a symlink; any
file or archive member carrying a path under `/Users` or `/home/<name>` other than `/home/agent`, a personal name or
an e-mail other than the two contact addresses and `@example.invalid`, a host or IP that is not loopback, TEST-NET
or the private-range example in the one unit test that needs it, the private experiment vocabulary, a credential
shape; a file with macOS extended attributes or a quarantine flag; a `.git` in a release tree (in CI, `--git-ok`
turns that into a check of every commit's author e-mail against `--release-email`). Every hit is printed with its
file and offset and the exit code is 1.

```bash
tools/build_package_release.sh            # builds dist/, repacks the sdist owner-free, runs the gate on tree + archives
python tools/release_gate.py --tree . --archive dist/*.whl --archive dist/*.tar.gz
```

The dataset's `build_dataset_release.sh` runs the same gate on the dataset tree and `episodes.tar.gz`. A build
whose gate prints a hit is not a release; fix the cause (`xattr -rc` the tree, strip the name, move the file) and
rebuild. Never add an exception to the gate to make a hit go away.

## Pull requests

- One change per pull request, with a test that would have caught the bug or pins the new behavior.
- Keep `docs/CONTRACT.md` in step with `latent_agent_sandbox/contract.py` and each family's `SPEC.md` in step
  with `scenarios/registry.py`; the tests read both.
- No dates, run counts or internal project names in docstrings and comments: state the technical fact.
- Never commit a run directory, an API key, or a real credential. The only "secrets" in this repository are the
  sandbox's planted `TESTONLY` strings.
