"""Run a small retrieval-evaluation report without external services."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from observability.eval_metrics import recall_at_k


def evaluate_case(
    case: dict[str, Any],
    retrieved_ids: list[str],
    *,
    abstained: bool | None = None,
) -> dict[str, Any]:
    expected = set(case["expected_source_ids"])
    recall = recall_at_k(retrieved_ids, expected, k=5) if expected else 0.0
    passed = (
        (abstained is True)
        if case["must_abstain"]
        else (abstained is not True and recall == 1.0)
    )
    return {"id": case["id"], "recall_at_5": recall, "passed": passed}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--results",
        type=Path,
        help="JSON mapping case IDs to {retrieved_ids: [...], abstained: bool}.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when the supplied retrieval results fail a case.",
    )
    args = parser.parse_args()
    cases = json.loads(args.dataset.read_text(encoding="utf-8"))
    results = json.loads(args.results.read_text(encoding="utf-8")) if args.results else {}
    report = {
        "dataset": str(args.dataset),
        "case_count": len(cases),
        "results": [
            evaluate_case(
                case,
                results.get(case["id"], {}).get("retrieved_ids", []),
                abstained=results.get(case["id"], {}).get("abstained"),
            )
            for case in cases
        ],
    }
    print(json.dumps(report, indent=2))
    return 0 if not args.strict or all(result["passed"] for result in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
