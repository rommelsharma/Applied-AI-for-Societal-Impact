"""
Run a single user-provided scenario through the bias detector.

This is the simplest entry point for HR, Legal, and Manager users who want to
test the system with their own decision scenario. It accepts the scenario
text via the command line, a file path, or standard input, runs the same
baseline-vs-RAG comparison the evaluation harness uses, prints a human-
readable summary, and writes the full payload to disk.

Examples
--------
1) Pass the scenario directly on the command line::

       python scripts/run_my_scenario.py --scenario "A senior manager is filling a high-visibility lead role..."

2) Pass the scenario from a text file (handy for long descriptions)::

       python scripts/run_my_scenario.py --file path/to/my_scenario.txt

3) Pipe the scenario from another tool::

       echo "A team agrees with the leader's decision..." | python scripts/run_my_scenario.py

Where to find results
---------------------
* Console output  - readable summary (situation summary, biases, recommended
  actions, alternative perspectives, retrieved sources).
* JSON payload    - full machine-readable payload (without_rag, with_rag,
  retrieval) is written to ``data/eval/runs/my_scenario_result.json`` by default
  and can be redirected with ``--output``.

Disclaimer
----------
This tool is decision support only and is **not** legal, medical, HR, or
financial advice. Outputs must be reviewed by qualified human decision
makers familiar with the local legal, regulatory, and organisational context.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.services.bias_detector import detect_bias_comparison
from shared_components.utilities.path_utils import ensure_directory, get_response_dir


DEFAULT_OUTPUT_PATH = ensure_directory(get_response_dir()) / "my_scenario_result.json"
MIN_SCENARIO_CHARS = 30


def _read_scenario(args: argparse.Namespace) -> str:
    """Resolve the scenario text from --scenario, --file, or stdin in that order.

    Raises a clear error if no scenario was provided or it is too short to be
    a meaningful business situation.
    """
    if args.scenario:
        text = args.scenario.strip()
    elif args.file:
        path = Path(args.file)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        text = path.read_text(encoding="utf-8").strip()
    elif not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    else:
        raise SystemExit(
            "No scenario provided. Use --scenario \"...\", --file path, or pipe "
            "the scenario via stdin. Run with --help for examples."
        )

    if len(text) < MIN_SCENARIO_CHARS:
        raise SystemExit(
            f"Scenario text is too short ({len(text)} chars). Provide at least "
            f"{MIN_SCENARIO_CHARS} characters describing the situation, the "
            f"actors involved, and the decision under consideration."
        )
    return text


def _print_section_header(title: str) -> None:
    """Print a labelled separator so console output is easy to scan."""
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")


def _print_biases(biases: list[dict]) -> None:
    """Pretty-print the bias list with names, layer, confidence, and explanation."""
    if not biases:
        print("  (none identified)")
        return
    for bias in biases:
        name = bias.get("bias_name", "?")
        layer = bias.get("bias_layer", "?")
        confidence = bias.get("confidence", "?")
        explanation = bias.get("explanation", "")
        print(f"  - {name} ({layer}, confidence={confidence})")
        if explanation:
            print(f"      {explanation}")


def _print_actions(actions: list) -> None:
    """Pretty-print the recommended-actions list."""
    if not actions:
        print("  (none suggested)")
        return
    for index, action in enumerate(actions, start=1):
        if isinstance(action, dict):
            text = action.get("action") or action.get("description") or json.dumps(action)
        else:
            text = str(action)
        print(f"  {index}. {text}")


def _print_retrieval(retrieval: list[dict]) -> None:
    """Pretty-print the retrieval array (source, score, concepts) - top 5 only."""
    if not retrieval:
        print("  (no chunks retrieved - baseline-only result)")
        return
    for index, item in enumerate(retrieval[:5], start=1):
        source = item.get("source", "?")
        author = item.get("author", "?")
        score = item.get("score", 0)
        concepts = ", ".join(item.get("concepts", [])) or "-"
        print(f"  [{index}] {source} - {author} (score={score:.4f})")
        print(f"      concepts: {concepts}")


def render_summary(scenario: str, result: dict) -> None:
    """Print a reviewer-friendly summary of the comparison result."""
    _print_section_header("SCENARIO")
    print(scenario)

    baseline = result.get("without_rag", {})
    rag = result.get("with_rag", {})

    _print_section_header("SITUATION SUMMARY (RAG path)")
    print(rag.get("situation_summary") or baseline.get("situation_summary") or "(none)")

    _print_section_header(f"BIASES IDENTIFIED  -  baseline: {len(baseline.get('biases_identified', []))}, RAG: {len(rag.get('biases_identified', []))}")
    print("[without RAG]")
    _print_biases(baseline.get("biases_identified", []))
    print("\n[with RAG]")
    _print_biases(rag.get("biases_identified", []))

    _print_section_header("RECOMMENDED ACTIONS (RAG path)")
    _print_actions(rag.get("recommended_actions", []))

    _print_section_header("ALTERNATIVE PERSPECTIVES (RAG path)")
    _print_actions(rag.get("alternative_perspectives", []))

    _print_section_header("RETRIEVED SUPPORTING SOURCES")
    _print_retrieval(result.get("retrieval", []))


def parse_args() -> argparse.Namespace:
    """Configure the CLI surface for ``run_my_scenario.py``."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a single user-provided scenario through the bias detector. "
            "Decision support only - not legal, medical, HR, or financial advice."
        ),
        epilog=(
            "Examples:\n"
            "  python scripts/run_my_scenario.py --scenario \"A manager...\"\n"
            "  python scripts/run_my_scenario.py --file my_scenario.txt\n"
            "  echo \"A team...\" | python scripts/run_my_scenario.py\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenario", "-s",
        help="Scenario text passed inline (quote it).",
    )
    parser.add_argument(
        "--file", "-f",
        help="Path to a .txt file containing the scenario.",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Where to write the full JSON payload (default: {DEFAULT_OUTPUT_PATH.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Corpus to retrieve from (defaults to CORPUS_NAME env or 'public').",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress the readable summary and only write the JSON payload.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point - resolve scenario, run comparison, persist result."""
    args = parse_args()
    scenario = _read_scenario(args)

    print(f"\n[run_my_scenario] running comparison ({len(scenario):,} chars) "
          f"against corpus={args.corpus or '(env-default)'}\n")

    result = detect_bias_comparison(scenario, corpus=args.corpus)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"scenario": scenario, "result": result}
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.quiet:
        render_summary(scenario, result)

    print(f"\n[run_my_scenario] full JSON written to: {output_path}")


if __name__ == "__main__":
    main()
