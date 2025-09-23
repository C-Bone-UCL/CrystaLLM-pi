import argparse
import os
import random
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _utils import extract_formula_nonreduced, extract_space_group_symbol


def atoms_in_formula(formula):
    tokens = re.findall(r"([A-Z][a-z]?)", formula)
    return set(tokens)


def derive_template(formula):
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    letter_map = {}
    next_letter = "A"
    template = ""
    for elem, count_str in tokens:
        if elem not in letter_map:
            letter_map[elem] = next_letter
            next_letter = chr(ord(next_letter) + 1)
        letter = letter_map[elem]
        count = int(count_str) if count_str else 1
        template += letter + (str(count) if count > 1 else "")
    return template


def process_row(args):
    idx, cif_text, reduced_formula = args
    if not isinstance(cif_text, str):
        cif_text = ""
    if not isinstance(reduced_formula, str):
        reduced_formula = ""
    form_nonred = extract_formula_nonreduced(cif_text)
    space_group = extract_space_group_symbol(cif_text)
    elements = atoms_in_formula(reduced_formula)
    template = derive_template(reduced_formula)
    return idx, form_nonred, space_group, elements, template


def select_in_bin(indices, elements_list, templates_list, need):
    if len(indices) <= need:
        return indices
    selected = []
    selected_atoms = set()
    selected_templates = set()
    remaining = set(indices)
    while len(selected) < need:
        best_idx = None
        best_atoms = -1
        best_new_template = -1
        for idx in remaining:
            new_atoms = len(elements_list[idx] - selected_atoms)
            new_template = 0 if templates_list[idx] in selected_templates else 1
            if new_atoms > best_atoms or (new_atoms == best_atoms and new_template > best_new_template):
                best_idx = idx
                best_atoms = new_atoms
                best_new_template = new_template
        if best_idx is None:
            break
        selected.append(best_idx)
        selected_atoms.update(elements_list[best_idx])
        selected_templates.add(templates_list[best_idx])
        remaining.remove(best_idx)
    if len(selected) < need:
        extra = random.sample(list(remaining), need - len(selected))
        selected.extend(extra)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet")
    parser.add_argument("--output_parquet")
    parser.add_argument("--target_size", type=int)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    random.seed(args.seed)

    df = pd.read_parquet(args.input_parquet)

    print(f"Original size: {len(df)}")
    print("Removing NaN values for density")

    df = df.dropna(subset=["Density (g/cm^3)"])
    # df = df[df["Density (g/cm^3)"] > 0.5]

    # randomly sample until we reach the target size
    if args.target_size > 0:
        df = df.sample(n=args.target_size, random_state=args.seed)

    print(f"New size: {len(df)}")

    df.to_parquet(args.output_parquet, index=False)


if __name__ == "__main__":
    main()