"""What the container deployments (docker compose, deploy/aws) rely on: config from the environment, audit on stdout."""
import json
import secrets
from pathlib import Path

import pytest

from airouter.audit import Audit
from airouter.config import expand_env, load_config

DOCKER_CFG = Path(__file__).resolve().parents[1] / "config" / "router.docker.yaml"
SALT = secrets.token_hex(32)
SITE_VARS = ("AIROUTER_AE_TITLE", "AIROUTER_ALLOWED_CALLING_AES", "AIROUTER_PACS_AE_TITLE", "AIROUTER_PACS_HOST",
             "AIROUTER_PACS_PORT", "AIROUTER_AI_URL_LUNG_NODULE", "AIROUTER_AI_URL_CT_QA", "AIROUTER_AI_TRANSPORT")


@pytest.fixture
def env(monkeypatch, tmp_path):
    for k in SITE_VARS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AIROUTER_DEID_SALT", SALT)
    return monkeypatch, {"router": {"workdir": str(tmp_path)}}


def test_docker_config_defaults_are_the_compose_stack(env):
    _, over = env
    cfg = load_config(DOCKER_CFG, over)
    assert (cfg.ae_title, cfg.allowed_calling_aes) == ("AIROUTER", [])
    assert (cfg.pacs.ae_title, cfg.pacs.host, cfg.pacs.port) == ("ORTHANC", "orthanc", 4242)
    assert cfg.endpoints["lung-nodule"].url == "http://mock-ai:8500/infer/lung-nodule"
    assert {e.transport for e in cfg.endpoints.values()} == {"multipart"}


def test_docker_config_takes_site_values_from_environment(env):
    mp, over = env
    for k, v in {"AIROUTER_AE_TITLE": "AIR_PROD", "AIROUTER_ALLOWED_CALLING_AES": "CT01, CT02",
                 "AIROUTER_PACS_AE_TITLE": "SITEPACS", "AIROUTER_PACS_HOST": "10.50.1.20", "AIROUTER_PACS_PORT": "104",
                 "AIROUTER_AI_URL_LUNG_NODULE": "https://model.internal/lung", "AIROUTER_AI_TRANSPORT": "stow-rs"}.items():
        mp.setenv(k, v)
    cfg = load_config(DOCKER_CFG, over)
    assert (cfg.ae_title, cfg.allowed_calling_aes) == ("AIR_PROD", ["CT01", "CT02"])
    assert (cfg.pacs.ae_title, cfg.pacs.host, cfg.pacs.port) == ("SITEPACS", "10.50.1.20", 104)
    assert cfg.endpoints["lung-nodule"].url == "https://model.internal/lung"
    assert cfg.endpoints["ct-qa"].url == "http://mock-ai:8500/infer/ct-qa"          # unset -> default
    assert {e.transport for e in cfg.endpoints.values()} == {"stow-rs"}


def test_env_expansion_rules(monkeypatch):
    monkeypatch.delenv("AIROUTER_X", raising=False)
    assert expand_env("a: ${AIROUTER_X:-d}") == "a: d"
    monkeypatch.setenv("AIROUTER_X", "")
    assert expand_env("a: ${AIROUTER_X:-d}") == "a: d"                             # empty counts as unset, as in sh
    assert expand_env("a: ${AIROUTER_X}") == "a: "
    monkeypatch.setenv("AIROUTER_X", "v")
    assert expand_env("a: ${AIROUTER_X:-d} ^CT$ $5") == "a: v ^CT$ $5"             # regexes untouched
    monkeypatch.delenv("AIROUTER_X")
    with pytest.raises(ValueError, match="AIROUTER_X"):
        expand_env("a: ${AIROUTER_X}")


def test_audit_echoes_to_stdout_only_when_asked(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("AIROUTER_AUDIT_STDOUT", raising=False)
    Audit(tmp_path / "a.jsonl").log("1.2.3", "route", rule="r")
    assert capsys.readouterr().out == ""
    monkeypatch.setenv("AIROUTER_AUDIT_STDOUT", "1")
    Audit(tmp_path / "a.jsonl").log("1.2.3", "route", rule="r")
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and json.loads(out[0])["step"] == "route"
    assert len((tmp_path / "a.jsonl").read_text().splitlines()) == 2                # the file is still written
