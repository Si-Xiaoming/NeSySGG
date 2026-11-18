import csv
import json
from pathlib import Path


def load_mapping_dict(json_path: str) -> dict:
    """
    Load a JSON file containing mapping dictionaries for entities and relations,
    then merge them into a single flat dictionary.

    Expected JSON examples:
    {
        "entity": {"car": "automobile", "person": "human"},
        "rel": {"on": "sopra", "under": "sotto"}
    }

    or

    {
        "entities": {"car": "automobile", "person": "human"},
        "relations": {"on": "sopra", "under": "sotto"}
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entity_dict = (
        data.get("entity")
        or data.get("entities")
        or data.get("entity_dict")
        or data.get("label_to_idx")
        or {}
    )
    rel_dict = (
        data.get("rel")
        or data.get("rels")
        or data.get("relation")
        or data.get("relations")
        or data.get("rel_dict")
        or data.get("predicate_to_idx")
        or {}
    )

    if not isinstance(entity_dict, dict):
        raise ValueError("Entity mapping in JSON must be a dictionary.")
    if not isinstance(rel_dict, dict):
        raise ValueError("Relation mapping in JSON must be a dictionary.")

    # Merge into one flat mapping dict
    mapping = {}
    mapping.update(entity_dict)
    mapping.update(rel_dict)

    return mapping


def translate_triplets_csv(
    input_csv_path: str,
    input_json_path: str,
    output_csv_path: str,
    keep_original_if_missing: bool = True,
) -> None:
    """
    Read a CSV with rows structured as:
        entity1, rel, entity2

    Apply mapping from JSON to each field and write the translated triplets
    to a new CSV with the same 3-column structure.

    If a value is missing from the mapping:
    - keep_original_if_missing=True  -> keep original value
    - keep_original_if_missing=False -> raise KeyError
    """
    mapping = load_mapping_dict(input_json_path)

    with open(input_csv_path, "r", encoding="utf-8", newline="") as f_in, \
         open(output_csv_path, "w", encoding="utf-8", newline="") as f_out:

        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        for row_idx, row in enumerate(reader, start=1):
            if len(row) != 3:
                raise ValueError(
                    f"Row {row_idx} has {len(row)} columns, expected exactly 3: {row}"
                )

            e1, rel, e2 = [cell.strip() for cell in row]

            if keep_original_if_missing:
                new_e1 = mapping.get(e1, e1)
                new_rel = mapping.get(rel, rel)
                new_e2 = mapping.get(e2, e2)
            else:
                try:
                    new_e1 = mapping[e1]
                    new_rel = mapping[rel]
                    new_e2 = mapping[e2]
                except KeyError as exc:
                    raise KeyError(
                        f"Missing mapping for value {exc.args[0]!r} at row {row_idx}"
                    ) from exc

            writer.writerow([new_e1, new_rel, new_e2])


if __name__ == "__main__":
    # Change these paths
    input_csv = "outputs_new/qwen_triplet_outputs5_pos_final_human.csv"
    input_json = "visual_genome_format.json"
    output_csv = "outputs_new_index/qwen_triplet_outputs5_pos_final.csv"

    translate_triplets_csv(
        input_csv_path=input_csv,
        input_json_path=input_json,
        output_csv_path=output_csv,
        keep_original_if_missing=True,
    )

    print(f"Done. Output written to: {Path(output_csv).resolve()}")