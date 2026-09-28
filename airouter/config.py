"""Load and validate router.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import deid


@dataclass
class PacsCfg:
    ae_title: str
    host: str
    port: int
    retries: int = 3


@dataclass
class EndpointCfg:
    url: str
    timeout_s: float = 30
    retries: int = 3


@dataclass
class Rule:
    name: str
    model: str | None                 # None = recognised and deliberately NOT sent anywhere
    match: dict[str, str] = field(default_factory=dict)       # every condition must match
    match_any: dict[str, str] = field(default_factory=dict)   # at least one must match (if given)
    series: str = "all"               # which images the model gets: "all" series, or the "largest" one


@dataclass
class Config:
    ae_title: str
    port: int
    study_quiet_seconds: float
    workers: int
    workdir: Path
    pacs: PacsCfg
    endpoints: dict[str, EndpointCfg]
    rules: list[Rule]
    deid_salt: str
    reject_burned_in: bool = True
    bind_address: str = "0.0.0.0"
    allowed_calling_aes: list[str] = field(default_factory=list)   # empty = accept any calling AE
    raw: dict = field(default_factory=dict)

    # Folder layout under workdir
    @property
    def inbox(self) -> Path:
        return self.workdir / "inbox"

    @property
    def results_dir(self) -> Path:
        return self.workdir / "results"

    @property
    def db_path(self) -> Path:
        return self.workdir / "router.sqlite"

    @property
    def audit_path(self) -> Path:
        return self.workdir / "audit.jsonl"


def load_config(path: str | Path, overrides: dict | None = None, require_salt: bool = True) -> Config:
    """require_salt=False only for read-only commands (status) that never de-identify anything."""
    raw = yaml.safe_load(Path(path).read_text())
    if overrides:
        _deep_update(raw, overrides)
    r = raw["router"]
    cfg = Config(
        ae_title=r["ae_title"],
        port=int(r["port"]),
        study_quiet_seconds=float(r.get("study_quiet_seconds", 3)),
        workers=int(r.get("workers", 2)),
        workdir=Path(r.get("workdir", "./data")),
        pacs=PacsCfg(**raw["pacs"]),
        endpoints={k: EndpointCfg(**v) for k, v in raw["ai_endpoints"].items()},
        rules=[Rule(**x) for x in raw["rules"]],
        # The salt is a secret: take it from the environment, never from a committed file.
        deid_salt=os.environ.get("AIROUTER_DEID_SALT") or raw["deid"].get("salt", ""),
        reject_burned_in=raw["deid"].get("reject_burned_in_annotation", True),
        bind_address=r.get("bind_address", "0.0.0.0"),
        allowed_calling_aes=list(r.get("allowed_calling_aes") or []),
        raw=raw,
    )
    if require_salt:
        deid.check_salt(cfg.deid_salt)      # refuse to run with a missing / placeholder / short salt
    for rule in cfg.rules:
        if rule.model is not None and rule.model not in cfg.endpoints:
            raise ValueError(f"rule {rule.name!r} points at unknown model {rule.model!r}")
    for d in (cfg.workdir, cfg.inbox, cfg.results_dir):
        d.mkdir(parents=True, exist_ok=True)
    return cfg


def _deep_update(base: dict, upd: dict) -> None:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
