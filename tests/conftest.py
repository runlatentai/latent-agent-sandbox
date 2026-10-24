"""Fixtures. Docker is required for the sandbox tests (they skip without a daemon); the image is built once per
session; the fake-agent episodes run once per session and are shared by the label tests. The baselines tests use
a small synthetic dataset in the contract's shape (tests/_fixtures.py), no Docker and no model."""

import os

import pytest

from latent_agent_sandbox.sandbox import IMAGE, build_image, docker_available, image_exists

os.environ.pop("ANTHROPIC_API_KEY", None)  # the judge never runs in tests


@pytest.fixture(scope="session")
def docker_daemon():
    if not docker_available():
        pytest.skip("docker daemon not available")
    return True


@pytest.fixture(scope="session")
def image(docker_daemon):
    if not image_exists(IMAGE) or os.environ.get("LAS_REBUILD") == "1":
        build_image(IMAGE)
    return IMAGE


@pytest.fixture(scope="session")
def fake_runs(image, tmp_path_factory):
    """Every policy through every scenario once, labeled."""
    from latent_agent_sandbox.fake_agent import POLICIES, run_fake
    from latent_agent_sandbox.label import label_run
    from latent_agent_sandbox.scenarios import registry

    out = str(tmp_path_factory.mktemp("fake_runs"))
    manifests = {}
    for key in registry.all_keys():
        fam, var = key.split("/")
        for pol in POLICIES[:4]:
            sc = registry.resolve(fam, var, "rle" if fam != "unauthorized_spend" else None)
            m = run_fake(sc, pol, out, episode_id=f"{fam}-{var}-{pol}")
            manifests[(key, pol)] = m
    rows = label_run(out)
    labels = {
        (r["scenario_key"].rsplit("/", 1)[0] if r["scenario_key"].count("/") == 2 else r["scenario_key"], r["policy"]): r
        for r in rows
        if "label_error" not in r
    }
    return {"out": out, "manifests": manifests, "labels": labels, "rows": rows}


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory):
    from tests._fixtures import make_dataset

    d = tmp_path_factory.mktemp("synthetic-data")
    make_dataset(d, n_episodes=420, seed=0)
    return d


@pytest.fixture(scope="session")
def dataset(dataset_dir):
    from latent_agent_sandbox.contract import load_dataset

    return load_dataset(dataset_dir)
