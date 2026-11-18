import argparse
import re
import csv
import copy
from pathlib import Path

import neptune
import wandb
# project = neptune.init_run(project="russo.alessandro199/SGG-VG", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNDcyMGQ5Mi05YjAzLTRjN2QtYjQ0Yi1kOTM1ZDFiZWI3ODQifQ==")

DEFAULT_WANDB_ENTITY = "shumingsi-politecnico-di-torino"
DEFAULT_WANDB_PROJECT = "SGGVG"


def parse_args():
    parser = argparse.ArgumentParser(description="Extract training stats from a log file and upload them to wandb.")
    parser.add_argument("--log_file", required=True, help="Path to the log file to parse.")
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Path to the output CSV. Defaults to the log path with a .csv suffix.",
    )
    parser.add_argument("--wandb_entity", default=DEFAULT_WANDB_ENTITY, help="Wandb entity name.")
    parser.add_argument("--wandb_project", default=DEFAULT_WANDB_PROJECT, help="Wandb project name.")
    parser.add_argument("--disable_wandb", action="store_true", help="Write CSV without uploading to wandb.")
    return parser.parse_args()


def log_to_wandb(data, log_file, wandb_entity, wandb_project):
    run_name = Path(log_file).name
    with wandb.init(entity=wandb_entity, project=wandb_project, name=run_name) as run:
        for d in data:
            for k in d.keys():
                if not isinstance(d[k], dict):
                    # project[k].append(d[k])
                    run.log({k: d[k]})
                else:
                    for k1 in d[k].keys():
                        run.log({f"{k}/{k1}": d[k][k1]})


def write_csv(data, output_csv):
    keys = sorted(set(key for entry in data for key in entry.keys()))
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)


def extract_stats(log_file, output_csv, wandb_entity=DEFAULT_WANDB_ENTITY, wandb_project=DEFAULT_WANDB_PROJECT, use_wandb=True):
    """
    Extracts specific statistics from a model log file and saves them to a CSV.

    :param log_file: Path to the log file.
    :param output_csv: Path to the CSV output file.
    """
    # Regular expression patterns
    r_pattern = re.compile(r"R@20: ([0-9.]+)|R@50: ([0-9.]+)|R@100: ([0-9.]+)")
    averaged_stats_pattern = re.compile(r"\b([a-zA-Z_0-9]+)\s*: (-?[0-9.]+)")
    ap_ar_pattern = re.compile(
        r"(Average (?:Precision|Recall))\s+\(\w+\)\s+@\[\s+IoU=([0-9.]+(?::[0-9.]+)?)\s+\|\s+area=\s*([a-z\s]+)\s+\|\s+maxDets=\s*(\d+)\s+\]\s+=\s+([0-9.]+)"
    )
    epoch_marker = "Epoch:"

    data = []
    current_epoch = 0
    current_entry = {"train": {}, "val": {}, "Epoch": -1}

    with open(log_file, 'r') as file:
        go = False
        go_train = False
        save = False
        for line in file:
            if re.search(r"Epoch:\s\[[0-9]+\]\s\s\[", line):
                current_epoch = int(re.match(r"Epoch:\s\[([0-9]+)\]\s\s\[", line).group(1))
                if current_epoch > current_entry["Epoch"]:
                    current_entry["Epoch"] = current_epoch
                else:
                    continue
            if '] Total time: ' in line:
                go_train = True
                continue
            if go_train and 'Test:  [' in line:
                go_train = False
            if '=sgdet=' in line:
            # if 'defaultdict' in line:
                go = True
                continue
            elif go and 'Epoch: [' in line:
                go = False
                data.append(copy.deepcopy(current_entry))
                continue
            if go or go_train:
                # Extract "R@" values
                r_match = r_pattern.findall(line)
                if r_match:
                    for match in r_match:
                        if match[0]:
                            current_entry["R@20"] = float(match[0])
                        elif match[1]:
                            current_entry["R@50"] = float(match[1])
                        elif match[2]:
                            current_entry["R@100"] = float(match[2])
                    continue

                # Extract averaged stats
                averaged_stats = averaged_stats_pattern.findall(line)

                if averaged_stats:
                    for key, value in averaged_stats:
                        if go_train:
                            current_entry["train"][key] = float(value)
                        elif go:
                            current_entry["val"][key] = float(value)
                    continue

                # Extract Average Precision and Recall
                ap_ar_match = ap_ar_pattern.findall(line)
                if ap_ar_match:
                    for match in ap_ar_match:
                        stat_type = "AP" if "Precision" in match[0] else "AR" # "Average Precision" or "Average Recall"
                        area = match[2].strip()
                        value = float(match[4])
                        key = f"IoU={match[1]} area={area} MaxDets={match[3]}"
                        if stat_type not in current_entry.keys():
                            current_entry[stat_type] = {}
                        current_entry[stat_type][key] = value

            # Add the last entry

    # Write results to CSV
    if data:
        if use_wandb:
            log_to_wandb(data, log_file, wandb_entity, wandb_project)
        write_csv(data, output_csv)
    print(f"Statistics extracted to {output_csv}")


def main():
    args = parse_args()
    output_csv = args.output_csv or str(Path(args.log_file).with_suffix(".csv"))
    extract_stats(
        log_file=args.log_file,
        output_csv=output_csv,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        use_wandb=not args.disable_wandb,
    )


if __name__ == "__main__":
    main()
