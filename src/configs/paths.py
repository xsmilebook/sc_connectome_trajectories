from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


class PathsConfigError(RuntimeError):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_scalar(raw: str) -> Any:
    s = raw.strip()
    if not s:
        return ""
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    low = s.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none", "~"}:
        return None
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


def load_simple_yaml(path: os.PathLike[str] | str) -> Dict[str, Any]:
    """
    Minimal YAML loader supporting nested mappings via indentation.

    Supported subset:
    - key: value
    - key: (starts a nested mapping)
    - indentation with spaces (2+)
    - comments starting with '#'

    This avoids an external PyYAML dependency for HPC environments.
    """

    p = Path(path)
    if not p.exists():
        raise PathsConfigError(f"Missing config file: {p}")

    root: Dict[str, Any] = {}
    stack: list[Tuple[int, Dict[str, Any]]] = [(-1, root)]

    for line_no, raw_line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip("\n").rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if "\t" in raw_line:
            raise PathsConfigError(f"Tabs are not supported in YAML: {p}:{line_no}")
        if ":" not in line:
            raise PathsConfigError(f"Invalid YAML (missing ':'): {p}:{line_no}")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise PathsConfigError(f"Invalid indentation: {p}:{line_no}")
        current = stack[-1][1]

        key_part, value_part = line.lstrip().split(":", 1)
        key = key_part.strip()
        if not key:
            raise PathsConfigError(f"Empty key: {p}:{line_no}")

        value_raw = value_part.strip()
        if value_raw == "":
            new_map: Dict[str, Any] = {}
            current[key] = new_map
            stack.append((indent, new_map))
        else:
            current[key] = _parse_scalar(value_raw)

    return root


def get_by_dotted_key(cfg: Dict[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise PathsConfigError(f"Missing config key: {dotted}")
        cur = cur[part]
    return cur


def resolve_repo_path(value: str) -> str:
    if value is None:
        raise PathsConfigError("Path value is None")
    if isinstance(value, (int, float, bool)):
        raise PathsConfigError(f"Expected string path, got: {type(value).__name__}")
    s = str(value)
    if s.startswith("/") or (len(s) >= 3 and s[1:3] == ":\\"):
        return s
    return str((_repo_root() / s).resolve())


def ensure_outputs_logs(config_path: str = "configs/paths.yaml") -> str:
    cfg = load_simple_yaml(_repo_root() / config_path)
    logs_rel = get_by_dotted_key(cfg, "local.outputs.logs")
    logs_path = Path(resolve_repo_path(str(logs_rel)))
    logs_path.mkdir(parents=True, exist_ok=True)
    return str(logs_path)


@dataclass(frozen=True)
class RenderSpec:
    var: str
    key: str


def parse_render_specs(items: Iterable[str]) -> list[RenderSpec]:
    specs: list[RenderSpec] = []
    for item in items:
        if "=" not in item:
            raise PathsConfigError(f"Invalid spec (expected VAR=key): {item}")
        var, key = item.split("=", 1)
        var = var.strip()
        key = key.strip()
        if not var or not key:
            raise PathsConfigError(f"Invalid spec (empty VAR/key): {item}")
        specs.append(RenderSpec(var=var, key=key))
    return specs


def to_bash_exports(kv: Dict[str, str]) -> str:
    parts = []
    for k, v in kv.items():
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        parts.append(f'export {k}="{escaped}"')
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/paths.yaml")
    p.add_argument("--format", choices=["bash"], default="bash")
    p.add_argument(
        "--set",
        nargs="+",
        metavar="VAR=key.path",
        help="Render config key(s) as environment variables.",
        required=True,
    )
    p.add_argument(
        "--resolve",
        action="store_true",
        help="Resolve repo-relative paths to absolute paths.",
    )
    args = p.parse_args(argv)

    cfg = load_simple_yaml(_repo_root() / args.config)
    specs = parse_render_specs(args.set)

    kv: Dict[str, str] = {}
    for spec in specs:
        val = get_by_dotted_key(cfg, spec.key)
        sval = "" if val is None else str(val)
        if args.resolve and sval:
            sval = resolve_repo_path(sval)
        kv[spec.var] = sval

    if args.format == "bash":
        print(to_bash_exports(kv))
        return 0
    raise PathsConfigError(f"Unsupported format: {args.format}")


if __name__ == "__main__":
    raise SystemExit(main())

