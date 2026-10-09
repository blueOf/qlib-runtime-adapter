"""Standalone FactorVector -> ModelScore -> Signal -> Strategy research entry.

Run from the source checkout with Python 3.12.14 and the locked dependencies:
    python adapter/run.py backtest --experiment EXP_ID --input input.json
        --config-root /path/to/local/configs --reports-root /path/to/local/runs

Factor definitions, inputs, and trained artifacts are supplied by the caller.
The four-layer service currently uses the deterministic execution engine.
The separate Qlib native adapter remains available as a library.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

STAGES = ("factor", "model", "signal", "strategy", "full")
COMMAND_STAGES = {
    "research": None,
    "backtest": "full",
    "factor-eval": "factor",
    "model-eval": "model",
    "signal-eval": "signal",
    "strategy-backtest": "strategy",
}


def _session_window(value):
    try:
        after, through = map(int, value.split(":"))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("session window must be AFTER:THROUGH") from error
    if not 0 <= after < through:
        raise argparse.ArgumentTypeError("session window requires 0 <= AFTER < THROUGH")
    return {"after": after, "through": through}


def main(argv=None):
    from quant_project.paths import config_root, runs_root

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, stage in COMMAND_STAGES.items():
        command = commands.add_parser(name, help="Run the configured research stages")
        command.add_argument("--experiment", required=True)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--config-root", type=Path, default=config_root())
        command.add_argument("--reports-root", type=Path, default=runs_root() / "experiments")
        command.add_argument("--run-id")
        command.add_argument("--dry-run", action="store_true")
        command.add_argument("--execution-engine", choices=("deterministic", "qlib"), default="deterministic")
        command.add_argument("--from", dest="window_start", help="Legacy V1 input only")
        command.add_argument("--to", dest="window_end", help="Legacy V1 input only")
        command.add_argument("--session-window", type=_session_window, help="Legacy V1 input only")
        if stage is None:
            command.add_argument("--stage", choices=STAGES, default="full")
    audit = commands.add_parser("model-artifact", help="Verify or restore sealed model artifacts")
    audit.add_argument("--run-root", type=Path, required=True)
    audit.add_argument("--roundtrip", action="store_true")
    audit.add_argument("--record-unavailable", action="store_true")
    args = parser.parse_args(argv)

    from quant_project.common import read_json
    from quant_project.modeling.contracts import CapabilityError, ConfigError, ModelingError

    stage = "model"
    failure_schema = "quant-project-model-failure-v1"
    try:
        if args.command == "model-artifact":
            if args.record_unavailable:
                from quant_project.model_retention import record_model_retention_downgrade

                result = record_model_retention_downgrade(args.run_root)
            else:
                from quant_project.evaluation.model_audit import audit_model_run

                result = audit_model_run(args.run_root, roundtrip=args.roundtrip)
        else:
            stage = COMMAND_STAGES[args.command] or args.stage
            failure_schema = "quant-project-research-failure-v1"
            try:
                payload = read_json(args.input)
            except (OSError, ValueError) as error:
                raise ConfigError(f"cannot read research input: {error}") from error
            if not isinstance(payload, dict):
                raise ConfigError("research input must be a JSON object")
            four_layer = payload.get("schema") == "quant-project-four-layer-input-v2"
            if four_layer:
                failure_schema = "quant-project-four-layer-failure-v2"
            elif stage == "model":
                failure_schema = "quant-project-model-failure-v1"
            overrides = args.window_start is not None or args.window_end is not None or args.session_window is not None
            if (four_layer or stage == "model") and overrides:
                raise CapabilityError("windows must be frozen in context and SplitPolicy")
            if overrides:
                context = dict(payload.get("context") or {})
                if args.window_start is not None:
                    context["start"] = args.window_start
                if args.window_end is not None:
                    context["end"] = args.window_end
                payload = dict(payload, context=context)
                if args.session_window is not None:
                    payload["session_window"] = args.session_window
            from quant_project.evaluation.research import run_experiment

            result = run_experiment(
                args.experiment, payload, config_root=args.config_root,
                reports_root=args.reports_root, stage=stage, run_id=args.run_id,
                execution_engine=args.execution_engine, dry_run=args.dry_run,
            )
    except ModelingError as error:
        print(json.dumps({"schema": failure_schema, "stage": stage, "status": "failed",
                          "code": error.code, "message": str(error), "details": error.details}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
