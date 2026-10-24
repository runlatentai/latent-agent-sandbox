# Security

This package runs an untrusted agent inside a Docker container with no network (`--network none`), as an
unprivileged user, with mock services on loopback. Nothing in it is a real exploit, a real credential, a real
wallet, or a real system; the planted strings (`ci-token-TESTONLY-...`, `pk_<word>_TESTONLY_<hex>`, `acct_...`,
the canary `gk-<16 hex>`) are accepted by nothing outside the container.

## Reporting

If you find a way for the agent to leave the container, reach the network, read the root-only logs or hidden
tests, or tamper with the access log without the labeler noticing, please report it privately rather than in a
public issue: **security@runlatent.ai**. Include the harness version, the Docker version, and the command the
agent ran. We aim to acknowledge a report within five business days.

## Scope

In scope: container isolation as configured by `latent_agent_sandbox/sandbox.py` and the Dockerfile, the
integrity of the file-access log and the command log, the labeler's reliance on them, and the handling of API keys
(read from the environment, never written). Out of scope: the behavior of any model you point the harness at, and
the security of the model server itself.
