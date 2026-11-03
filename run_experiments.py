import argparse
import copy
import csv
import json
from pathlib import Path

import joblib

from evaluate import apply_rationale_review, evaluate_run
from predict import run_predictions
from rag import load_config, read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--review")
    args = parser.parse_args()
    cfg = load_config(args.config)
    questions, gold, corpus = [read_jsonl(path) for path in [args.questions, args.gold, args.corpus]]
    index = joblib.load(Path(args.index) / "index.joblib")
    if index["build_settings"] != {"chunk_chars": cfg["chunk_chars"]}:
        raise ValueError("Индекс собран с другими настройками чанкинга")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    settings = [("no_context", False, False), ("baseline", False, False),
                ("final", True, True), ("no_char", False, True), ("no_neighbors", True, False)]
    rows = []
    for name, char_search, neighbors in settings:
        current = copy.deepcopy(cfg)
        current.update(name=name, char_search=char_search, neighbors=neighbors)
        folder = output / name
        folder.mkdir(exist_ok=True)
        answers, traces = run_predictions(index, questions, folder / "answers.jsonl", folder / "trace.jsonl", current)
        metrics = evaluate_run(questions, gold, answers, traces, corpus, no_context=name == "no_context")
        if args.review:
            review = json.loads(Path(args.review).read_text(encoding="utf-8"))
            metrics = apply_rationale_review(metrics, review[name], folder / "answers.jsonl")
        (folder / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (folder / "settings.json").write_text(json.dumps({k: v for k, v in current.items() if k != "tokenizer_dir"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows.append({"name": name, **{field: metrics[field] for field in ["answer_accuracy", "Q", "mean_input_tokens", "mean_output_tokens", "mean_llm_calls", "mean_cost", "median_latency_ms"]}})
        with (output / "experiments.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(name, json.dumps(rows[-1], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
