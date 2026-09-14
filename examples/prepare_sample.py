from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.preprocessing import preprocess_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one [C,T] EEG array")
    parser.add_argument("--input", required=True)
    parser.add_argument("--channels", required=True, help="JSON list of channel names")
    parser.add_argument("--sample-rate", required=True, type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    signal = np.load(args.input, allow_pickle=False)
    channels = json.loads(Path(args.channels).read_text(encoding="utf-8"))
    windows = preprocess_record(signal, args.sample_rate, channels)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, windows, allow_pickle=False)
    print(json.dumps({"output": str(output), "shape": list(windows.shape)}))


if __name__ == "__main__":
    main()

