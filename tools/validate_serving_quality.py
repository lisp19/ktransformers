#!/usr/bin/env python3
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class CaseResult:
    name: str
    passed: bool
    output_text: str
    reasoning_text: str
    failures: list[str]
    usage: dict[str, Any] | None


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def parse_json_fragment(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found")
    return json.loads(text[start : end + 1])


def run_case(base_url: str, model: str, case: dict[str, Any], timeout: int) -> CaseResult:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": case["messages"],
        "temperature": case.get("temperature", 0),
        "max_tokens": case.get("max_tokens", 128),
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=(10, timeout))
    resp.raise_for_status()
    body = resp.json()
    choice = body["choices"][0]
    message = choice.get("message", {})
    text = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""

    failures: list[str] = []
    normalized = normalize_text(text)

    if not normalized:
        failures.append("empty_output")

    min_chars = case.get("min_chars")
    if min_chars is not None and len(text) < min_chars:
        failures.append(f"min_chars<{min_chars}")

    max_chars = case.get("max_chars")
    if max_chars is not None and len(text) > max_chars:
        failures.append(f"max_chars>{max_chars}")

    for item in case.get("must_contain", []):
        if item not in text:
            failures.append(f"missing:{item}")

    for item in case.get("must_not_contain", []):
        if item in text:
            failures.append(f"forbidden:{item}")

    if "json_required_keys" in case or "json_expected" in case:
        try:
            parsed = parse_json_fragment(text)
        except Exception as exc:
            failures.append(f"json_parse:{exc}")
        else:
            for key in case.get("json_required_keys", []):
                if key not in parsed:
                    failures.append(f"json_missing_key:{key}")
            for key, expected in case.get("json_expected", {}).items():
                if parsed.get(key) != expected:
                    failures.append(f"json_value:{key}!={expected}")

    return CaseResult(
        name=case["name"],
        passed=not failures,
        output_text=text,
        reasoning_text=reasoning,
        failures=failures,
        usage=body.get("usage"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--write-reference", action="store_true")
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    results = [run_case(args.base_url, args.model, case, args.timeout) for case in cases]

    report = {
        "base_url": args.base_url,
        "model": args.model,
        "all_passed": all(r.passed for r in results),
        "results": [asdict(r) for r in results],
    }

    out_path = Path(args.output_json)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.write_reference:
        ref_path = out_path.with_suffix(".reference.json")
        reference = {
            r.name: {
                "output_text": r.output_text,
                "reasoning_text": r.reasoning_text,
            }
            for r in results
        }
        ref_path.write_text(
            json.dumps(reference, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.name}")
        if result.failures:
            print("  " + ", ".join(result.failures))

    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
