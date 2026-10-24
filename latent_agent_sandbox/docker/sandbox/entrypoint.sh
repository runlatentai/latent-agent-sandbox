#!/bin/bash
# Container entrypoint: set the scenario up (/sandbox/scenario.json, copied in by the harness), start the watcher and the mock services on
# loopback, write /var/log/sandbox/ready, then sleep. Everything here runs as root; the agent runs as `agent`
# through `docker exec -u agent`.
set -eu
mkdir -p /var/log/sandbox/sync
chmod 0700 /var/log/sandbox
: > /var/log/sandbox/fs.jsonl
: > /var/log/sandbox/results.jsonl
: > /var/log/sandbox/payments.jsonl
python3 /sandbox/setup.py /sandbox/scenario.json   # writes /sandbox/watch_paths.json and /sandbox/services.json
python3 /sandbox/watch.py /sandbox/watch_paths.json /var/log/sandbox/fs.jsonl &
python3 /sandbox/services.py /sandbox/services.json &
# wait for the watcher's "watching" line and the services' "listening" lines
for i in $(seq 1 200); do
  if grep -q '"kind": "watching"' /var/log/sandbox/fs.jsonl 2>/dev/null \
     && [ -f /var/log/sandbox/services.ready ]; then break; fi
  sleep 0.05
done
touch /var/log/sandbox/ready
exec sleep infinity
