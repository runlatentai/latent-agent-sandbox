# The latent-agent-sandbox tool image. Build with `docker build`; run with `--network none` only. No GPU, no model weights.
# Mock services live on loopback inside the container (sandbox/services.py), started by the entrypoint.
# The base image is pinned by digest (`docker buildx imagetools inspect python:3.12-slim` prints the current one;
# bump it deliberately) and every pip install is hash-locked (requirements-sandbox.txt, requirements-buildtools.txt);
# the apt packages (curl, git) are version-unpinned, so the build is reproducible up to what Debian's repository
# serves on the build day. The image has no network at run time: every package the agent can `pip install` comes
# from the local index below.
FROM python:3.12-slim@sha256:cb33f7f771ecec2ef1225cf39b159591a040391bd73e4c01f00fdfc4c47611d2

COPY requirements-sandbox.txt requirements-buildtools.txt /tmp/
RUN apt-get update && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --require-hashes -r /tmp/requirements-sandbox.txt \
    && rm /tmp/requirements-sandbox.txt \
    && useradd -m -u 1000 -s /bin/bash agent \
    && mkdir -p /workspace /grader /var/log/sandbox /sandbox \
    && chown agent:agent /workspace

COPY sandbox/ /sandbox/
# the local package index: two tiny wheels built here, served on 127.0.0.1:8470 inside the sandbox (pip install
# works for them and nothing else; the agent's pip.conf points at it). setuptools and wheel are hash-locked.
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements-buildtools.txt >/dev/null \
    && rm /tmp/requirements-buildtools.txt \
    && for p in textkit numfmt; do mkdir -p /sandbox/index/simple/$p && pip wheel --no-deps --no-build-isolation \
         -w /sandbox/index/simple/$p /sandbox/index_src/$p >/dev/null; done \
    && python3 /sandbox/make_index.py /sandbox/index/simple \
    && mkdir -p /home/agent/.config/pip \
    && printf '[global]\nindex-url = http://127.0.0.1:8470/simple/\ntrusted-host = 127.0.0.1\ndisable-pip-version-check = true\n' > /home/agent/.config/pip/pip.conf \
    && chown -R agent:agent /home/agent/.config
# /sandbox is root only. Nothing episode-specific is in the image: the harness copies one scenario.json in
# (docker cp, before start) and setup.py materializes it, then deletes it. The agent sees /workspace, its home,
# and /grader as set up.
RUN chmod -R go-rwx /sandbox && chmod -R a+rX /sandbox/index && chmod 0700 /var/log/sandbox \
    && install -m 0755 /sandbox/submit /usr/local/bin/submit \
    && chmod 0755 /sandbox/entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/sandbox/entrypoint.sh"]
