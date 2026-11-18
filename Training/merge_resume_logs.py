import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


EPOCH_PATTERN = re.compile(r"Epoch:\s*\[(\d+)\]")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge consecutive training logs and upload their statistics to Weights & Biases."
    )
    parser.add_argument("--initial_log", required=True, help="Log produced before the interruption.")
    parser.add_argument("--resumed_log", required=True, help="Log produced after resuming training.")
    parser.add_argument("--output_log", required=True, help="Merged log path.")
    parser.add_argument("--output_csv", required=True, help="CSV path passed to upload.py.")
    parser.add_argument("--wandb_entity", required=True)
    parser.add_argument("--wandb_project", required=True)
    return parser.parse_args()


def epoch_range(log_path: Path):
    epochs = set()
    with log_path.open("r", errors="replace") as log_file:
        for line in log_file:
            epochs.update(int(epoch) for epoch in EPOCH_PATTERN.findall(line))
    if not epochs:
        raise ValueError(f"No epoch markers found in {log_path}")
    return min(epochs), max(epochs)


def append_file(source: Path, destination):
    with source.open("rb") as source_file:
        shutil.copyfileobj(source_file, destination)


def merge_logs(initial_log: Path, resumed_log: Path, output_log: Path):
    initial_min, initial_max = epoch_range(initial_log)
    resumed_min, resumed_max = epoch_range(resumed_log)
    if resumed_min != initial_max + 1:
        raise ValueError(
            "Logs are not consecutive: "
            f"{initial_log} covers epochs {initial_min}-{initial_max}, while "
            f"{resumed_log} starts at epoch {resumed_min}."
        )

    output_log.parent.mkdir(parents=True, exist_ok=True)
    temporary_log = output_log.with_suffix(output_log.suffix + ".tmp")
    try:
        with temporary_log.open("wb") as merged_file:
            append_file(initial_log, merged_file)
            merged_file.write(b"\n")
            append_file(resumed_log, merged_file)
        temporary_log.replace(output_log)
    finally:
        temporary_log.unlink(missing_ok=True)

    print(
        f"Merged {initial_log} (epochs {initial_min}-{initial_max}) and "
        f"{resumed_log} (epochs {resumed_min}-{resumed_max}) into {output_log}"
    )


def main():
    args = parse_args()
    initial_log = Path(args.initial_log)
    resumed_log = Path(args.resumed_log)
    output_log = Path(args.output_log)
    output_csv = Path(args.output_csv)

    merge_logs(initial_log, resumed_log, output_log)
    subprocess.run(
        [
            sys.executable,
            "upload.py",
            "--log_file",
            str(output_log),
            "--output_csv",
            str(output_csv),
            "--wandb_entity",
            args.wandb_entity,
            "--wandb_project",
            args.wandb_project,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
