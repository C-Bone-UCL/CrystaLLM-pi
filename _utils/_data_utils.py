"""
Data utilities for CIF tokenization, filtering, and condition processing.
"""

import numpy as np
import ast
import torch

np.random.seed(1)

def validate_condition_values(tokenized_dataset, dataset_with_idx, parsed_condition_columns):
    """
    Helper function to validate condition values between source and tokenized datasets.
    """
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

    print(f"Row {row_idx_raw} — Original: {original_conditions}, Tokenized: {tokenized_conditions}")
    if len(original_conditions) == len(tokenized_conditions):
        if all(abs(o - t) < 1e-5 for o, t in zip(original_conditions, tokenized_conditions)):
            print("Condition values match")
        else:
            print("Condition values MISMATCH!")
    else:
        print("Condition values length MISMATCH!")


def get_token_length_stats(tokenized_dataset, split="train"):
    """Gathers the length (number of tokens) for each row in a given split 
    and prints stats."""
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
    """Remove examples with unknown tokens from dataset."""
    
    def filter_no_unk(example):
        return tokenizer.unk_token_id not in example["input_ids"]
    tokenized_dataset = tokenized_dataset.filter(filter_no_unk)
    print(f"Removed entries with unknown tokens")
    return tokenized_dataset


def create_fixed_format_mask(text, tokenizer, full_length):
    """
    Generate a binary mask for the CIF text:
      - Tokens not within variable brackets are 1 (fixed).
      - Tokens inside brackets are 0 (variable).
      - The bracket tokens "[" and "]" themselves are 1 (fixed).
    """
    
    tokenized = tokenizer(text, truncation=False)
    tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"])
    
    mask = []
    inside_variable = False
    for token in tokens:
        if token == "[":
            mask.append(1)
            inside_variable = True
        elif token == "]":
            mask.append(1)
            inside_variable = False
        else:
            mask.append(0 if inside_variable else 1)
    return mask

# Helper functions for tokenization with conditions

def _validate_inputs(condition_columns, mode):
    """Quick validation of critical inputs."""
    if mode in ["conditional", "raw"] and condition_columns is None:
        raise ValueError(f"condition_columns required for mode '{mode}'")
    
    if mode in ["conditional", "raw"] and condition_columns is not None:
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

def _process_conditions_for_text(examples, condition_columns, num_examples):
    """Process conditions and return as formatted text strings."""
    condition_strings = []
    
    for i in range(num_examples):
        condition_strs = []
        for column_name in condition_columns:
            raw_value = examples[column_name][i]
            float_values = _parse_condition_value(raw_value)
            float_strings = [f"{v:.4f}" for v in float_values]
            condition_strs.extend(float_strings)
        
        condition_strings.append(f"[{' '.join(condition_strs)}]")
    
    return condition_strings

def tokenize_function(examples, tokenizer, condition_columns=None, mode="unconditional"):
    """Tokenize CIF examples with optional conditioning support."""
    if mode not in ["unconditional", "conditional", "raw"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'unconditional', 'conditional', or 'raw'")
    
    num_examples = len(examples["CIF"])
    parsed_condition_columns = _validate_inputs(condition_columns, mode)
    
    # Prepare texts based on mode
    if mode in ["unconditional", "conditional"]:
        texts = [f"{tokenizer.bos_token}\n{example}\n{tokenizer.eos_token}" for example in examples["CIF"]]
    else:  # mode == "raw"
        condition_strings = _process_conditions_for_text(examples, parsed_condition_columns, num_examples)
        texts = [f"{tokenizer.bos_token}\n{condition_strings[i]}\n{examples['CIF'][i]}\n{tokenizer.eos_token}" 
                for i in range(num_examples)]
    
    tokenized_output = tokenizer(
        texts,
        truncation=False,
        return_special_tokens_mask=True,
        return_attention_mask=True
    )
    
    if mode == "conditional":
        batch_condition_values = _process_conditions_for_numeric(examples, parsed_condition_columns, num_examples)
        tokenized_output["condition_values"] = batch_condition_values
    
    # Create fixed masks
    masks = [create_fixed_format_mask(texts[i], tokenizer, len(tokenized_output["input_ids"][i])) 
             for i in range(num_examples)]
    
    tokenized_output["fixed_mask"] = masks
    return tokenized_output
