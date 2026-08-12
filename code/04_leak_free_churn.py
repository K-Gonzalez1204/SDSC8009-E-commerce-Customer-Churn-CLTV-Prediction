import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from leak_free.main import run_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    manifest = run_pipeline(limit=args.limit)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
