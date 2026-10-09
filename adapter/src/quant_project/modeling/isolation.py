"""Run a registered fold in a bounded process, including artifact publication."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ..common import atomic_json, read_json
from .contracts import ModelingError, ResourceBudgetError


def run_isolated_fold(request, directory, *, max_wall_seconds, max_threads):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    request_path, response_path = directory / "request.json", directory / "response.json"
    atomic_json(request_path, request)
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = source_root
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = str(max_threads)
    command = [sys.executable, "-m", "quant_project.modeling.worker", str(request_path), str(response_path)]
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, creationflags=creation_flags) as process:
        try:
            stdout, stderr = process.communicate(timeout=max_wall_seconds)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.communicate()
            raise ResourceBudgetError("model fold exceeded max_wall_seconds; worker terminated") from error
        except BaseException:
            process.kill()
            process.communicate()
            raise
    if not response_path.is_file():
        raise ModelingError("model worker exited without a result", details={"returncode": process.returncode,
                            "stderr": stderr.decode("utf-8", errors="replace")[-2000:]})
    result = read_json(response_path)
    if result.get("status") != "success":
        from . import contracts
        error_type = next((getattr(contracts, name) for name in dir(contracts)
                           if isinstance(getattr(contracts, name), type)
                           and issubclass(getattr(contracts, name), ModelingError)
                           and getattr(getattr(contracts, name), "code", None) == result.get("code")), ModelingError)
        raise error_type(result.get("message", "model worker failed"), details=result.get("details"))
    return result
