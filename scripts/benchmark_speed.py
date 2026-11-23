"""Measure every run's latency under the same conditions.

    python -m scripts.benchmark_speed

The latency recorded during the experiment suite depends on whatever else the
machine was doing at that moment. This script loads every run, warms each one
up, then times them in interleaved rounds (A, B, C, ..., A, B, C, ...), so
background load and CPU clock changes affect all models alike. Run it on an
otherwise idle machine.

The measured call is the app's own: one image, Keras on the CPU, i.e. how long a
user waits for a prediction. For phone-style numbers see ``export_tflite``.
Updates ``speed`` in ``reports/results/<run>.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.benchmark_speed)
except ImportError:
    import _bootstrap  # noqa: F401  (python scripts/benchmark_speed.py)

import numpy as np

from plantdisease import config

METHOD = "interleaved benchmark, single image, Keras on CPU (scripts/benchmark_speed.py)"


def interleaved_timings(predict: dict, inputs: dict, rounds: int, calls: int, warmup: int = 5) -> dict[str, list[float]]:
    """Milliseconds per call for each name, measured in alternating rounds."""
    for name, fn in predict.items():
        for _ in range(warmup):
            fn(inputs[name])
    times: dict[str, list[float]] = {name: [] for name in predict}
    for _ in range(rounds):
        for name, fn in predict.items():
            for _ in range(calls):
                start = time.perf_counter()
                fn(inputs[name])
                times[name].append((time.perf_counter() - start) * 1000)
    return times


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=config.RUNS_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--calls", type=int, default=5, help="Calls per model per round.")
    args = parser.parse_args(argv)

    from plantdisease.inference import PlantDiseaseClassifier

    names = [p.stem for p in sorted(args.results_dir.glob("*.json"))
             if (args.runs_dir / p.stem / config.MODEL_FILENAME).exists()]
    if not names:
        print(f"No runs with results found ({args.runs_dir}, {args.results_dir})")
        return 1
    classifiers = {n: PlantDiseaseClassifier.load(args.runs_dir / n) for n in names}
    rng = np.random.default_rng(0)
    inputs = {n: rng.random((1, c.image_size, c.image_size, 3), dtype=np.float32) for n, c in classifiers.items()}
    print(f"Timing {len(names)} models: {args.rounds} rounds x {args.calls} calls each", flush=True)
    times = interleaved_timings({n: c.predict_batch for n, c in classifiers.items()}, inputs, args.rounds, args.calls)

    for name in names:
        path = args.results_dir / f"{name}.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        speed = result.setdefault("speed", {})
        speed.update({
            "latency_ms_median": round(float(np.median(times[name])), 2),
            "latency_ms_p90": round(float(np.percentile(times[name], 90)), 2),
            "latency_method": METHOD,
            "params": int(classifiers[name].model.count_params()),
            "model_mb": round((args.runs_dir / name / config.MODEL_FILENAME).stat().st_size / 1e6, 2),
        })
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"{name:28s} {speed['latency_ms_median']:7.1f} ms (p90 {speed['latency_ms_p90']:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
