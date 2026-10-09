"""Private process entry; only repository-registered adapters can be invoked."""
from __future__ import annotations

import sys
import time
from pathlib import Path

from ..common import atomic_json, read_json
from .contracts import FitContext, ModelInputBatch, ModelSpec, ModelingError, PredictContext
from .runner import ModelRunner


def run(request):
    spec = ModelSpec.from_mapping(request["model"])
    fields = spec.input_factor_ids
    train = request.get("train_rows")
    train_batch = ModelInputBatch.from_rows(train, fields) if train is not None else None
    inference = ModelInputBatch.from_rows(request["inference_rows"], fields)
    runner = ModelRunner(artifact_store=request["artifact_store"])
    started = time.monotonic()
    fitted = runner.fit(spec, train_batch, FitContext(**request["fit_context"]))
    scores = runner.predict(fitted, inference, PredictContext(**request["predict_context"]))
    return {"status": "success", "scores": [score.to_dict() for score in scores],
            "artifact": dict(fitted.artifact), "wall_seconds": time.monotonic() - started}


def main():
    request_path, response_path = sys.argv[1:]
    try:
        result = run(read_json(request_path))
    except ModelingError as error:
        result = {"status": "failed", "code": error.code, "message": str(error), "details": error.details}
    except Exception as error:
        result = {"status": "failed", "code": "MODELING_ERROR", "message": str(error)}
    atomic_json(Path(response_path), result)


if __name__ == "__main__":
    main()
