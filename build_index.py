import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import joblib

from rag import create_index, load_config, read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    cfg = load_config(args.config)
    blocks = read_jsonl(args.corpus)
    index = create_index(blocks, cfg)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, output / "index.joblib", compress=3)
    stats = {"blocks": len(blocks), "chunks": len(index["chunks"]), "char_features": len(index["vectorizer"].vocabulary_),
             "seconds": time.perf_counter() - start, "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
             "index_bytes": (output / "index.joblib").stat().st_size,
             "corpus_sha256": hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest(), "settings": index["build_settings"]}
    (output / "build_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
