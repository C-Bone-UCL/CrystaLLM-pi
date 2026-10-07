"""Data utilities for CIF tokenization, filtering, and condition processing.
"""

import numpy as np
import ast
import torch

np.random.seed(1)

def validate_condition_values(tokenized_dataset, dataset_with_idx, parsed_condition_columns):
    """Check that condition values survive tokenisation, comparing one row against its source."""
    print("\nCondition Values Check")
    if not tokenized_dataset["train"]:
        print("Train split is empty, skipping condition check")
        return

    row_idx_raw = int(tokenized_dataset["train"][0]["__raw_idx"])

    original_conditions = []
    for col_name in parsed_condition_columns:
        original_value = dataset_with_idx["train"][col_name][row_idx_raw]
        if col_name == "Condition Vector" or col_name == "condition_vector":
            if isinstance(original_value, str):
                try:
                    parsed_list = ast.literal_eval(original_value)
                    rounded_list = [round(float(v), 4) for v in parsed_list]
                    original_conditions.extend(rounded_list)
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Could not parse 'Condition Vector' for row {row_idx_raw}: {e}")
                    original_conditions.extend([-100.0])
        elif isinstance(original_value, (int, float)):
            original_conditions.append(round(float(original_value), 4))
        elif isinstance(original_value, str):
            try:
                original_conditions.append(round(float(original_value), 4))
            except ValueError:
                print(f"Warning: Could not convert to float for column '{col_name}' row {row_idx_raw}")
                original_conditions.append(-100.0)
        else:
            print(f"Warning: Unhandled type for column '{col_name}' row {row_idx_raw}")
            original_conditions.append(-100.0)

    tokenized_conditions = tokenized_dataset["train"][0]["condition_values"]
    if isinstance(tokenized_conditions, torch.Tensor):
        tokenized_conditions = tokenized_conditions.tolist()

    print(f"Row {row_idx_raw} Original: {original_conditions}, Tokenized: {tokenized_conditions}")
    if len(original_conditions) == len(tokenized_conditions):
        if all(abs(o - t) < 1e-5 for o, t in zip(original_conditions, tokenized_conditions)):
            print("Condition values match")
        else:
            print("Condition values MISMATCH!")
    else:
        print("Condition values length MISMATCH!")


def get_token_length_stats(tokenized_dataset, split="train"):
    """Calculate and print token-length statistics for each row in a dataset split."""
    dataset_split = tokenized_dataset[split]
    lengths = []
    for i in range(len(dataset_split)):
        example = dataset_split[i]
        lengths.append(len(example["input_ids"]))

    mean_length = np.mean(lengths)
    min_length = np.min(lengths)
    max_length = np.max(lengths)
    std_length = np.std(lengths)

    print(f"Split: {split}")
    print("Mean length of tokens:", mean_length)
    print("Min length of tokens:", min_length)
    print("Max length of tokens:", max_length)
    print("Standard Deviation of token lengths:", std_length)
    print("Amount of tokens above 1024:", len([l for l in lengths if l > 1024]))
    print("Amount of tokens above 2048:", len([l for l in lengths if l > 2048]))


def filter_long_CIFs(tokenized_dataset, context_length):
    """Filter out entries where token length exceeds context length."""
    
    def filter_long(example):
        return len(example["input_ids"]) <= context_length
    tokenized_dataset = tokenized_dataset.filter(filter_long, num_proc=4)
    print(f"Removed entries with token length exceeding {context_length}")
    return tokenized_dataset


def filter_CIFs_with_unk(tokenized_dataset, tokenizer):
    """Remove CIFs with unknown tokens and report removals per split."""
    unk_id = tokenizer.unk_token_id
    if unk_id is None:  # Skip filtering when no unknown-token ID is defined.
        raise ValueError("filter_CIFs_with_unk needs the slow CustomCIFTokenizer, whose unk id is 370")

    sizes_before = {split: len(rows) for split, rows in tokenized_dataset.items()}
    tokenized_dataset = tokenized_dataset.filter(lambda example: unk_id not in example["input_ids"])

    for split, rows in tokenized_dataset.items():
        print(f"Removed {sizes_before[split] - len(rows)} of {sizes_before[split]} {split} entries with unknown tokens")
    return tokenized_dataset


def create_fixed_format_mask(text, tokenizer, full_length):
    """Generate a binary mask for the CIF text.

    Tokens outside variable brackets are 1 (fixed), tokens inside are 0 (variable), and the bracket
    tokens "[" and "]" themselves are 1.
    """
    
    return _fixed_mask_from_ids(tokenizer(text, truncation=False)["input_ids"], tokenizer)


def _fixed_mask_from_ids(input_ids, tokenizer):
    """Mask encoded IDs: 1 on and outside brackets, 0 inside."""
    open_id, close_id = tokenizer.convert_tokens_to_ids(["[", "]"])

    mask = []
    inside_variable = False
    for token_id in input_ids:
        if token_id == open_id:
            inside_variable = True
            mask.append(1)
        elif token_id == close_id:
            inside_variable = False
            mask.append(1)
        else:
            mask.append(0 if inside_variable else 1)
    return mask

# Helper functions for tokenization with conditions

def _validate_inputs(condition_columns, mode):
    """Quick validation of critical inputs."""
    if mode == "conditional" and condition_columns is None:
        raise ValueError(f"condition_columns required for mode '{mode}'")

    if mode == "conditional" and condition_columns is not None:
        parsed = ast.literal_eval(str(condition_columns)) if isinstance(condition_columns, str) else condition_columns
        if not isinstance(parsed, list):
            raise ValueError("condition_columns must be a list")
        return parsed
    return None

def _parse_condition_value(raw_value):
    """Parse condition value to float list."""
    if isinstance(raw_value, str):
        try:
            parsed = ast.literal_eval(raw_value)
            if isinstance(parsed, list):
                return [float(v) for v in parsed]
            else:
                return [float(parsed)]
        except (ValueError, SyntaxError):
            return [float(raw_value)]
    
    if isinstance(raw_value, (int, float)):
        return [float(raw_value)]
    
    if isinstance(raw_value, list):
        return [float(v) for v in raw_value]
    
    return [float(raw_value)]

def _process_conditions_for_numeric(examples, condition_columns, num_examples):
    """Process conditions and return as numeric values for conditioning."""
    batch_condition_values = []
    
    for i in range(num_examples):
        example_conditions = []
        for column_name in condition_columns:
            raw_value = examples[column_name][i]
            float_values = _parse_condition_value(raw_value)
            float_values = [round(v, 4) for v in float_values]
            example_conditions.extend(float_values)
        batch_condition_values.append(example_conditions)
    
    return batch_condition_values

def tokenize_function(examples, tokenizer, condition_columns=None, mode="unconditional"):
    """Encode each CIF once, with optional conditioning."""
    if mode not in ["unconditional", "conditional"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'unconditional' or 'conditional'")

    num_examples = len(examples["CIF"])
    parsed_condition_columns = _validate_inputs(condition_columns, mode)

    texts = [f"{tokenizer.bos_token}\n{example}\n{tokenizer.eos_token}" for example in examples["CIF"]]
    all_ids = tokenizer(texts, truncation=False)["input_ids"]

    # The collator fills special_tokens_mask with zeros.
    tokenized_output = {
        "input_ids": all_ids,
        "attention_mask": [[1] * len(ids) for ids in all_ids],
        "fixed_mask": [_fixed_mask_from_ids(ids, tokenizer) for ids in all_ids],
    }

    if mode == "conditional":
        batch_condition_values = _process_conditions_for_numeric(examples, parsed_condition_columns, num_examples)
        tokenized_output["condition_values"] = batch_condition_values

    return tokenized_output
