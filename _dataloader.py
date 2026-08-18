"""Load, tokenize and collate CIF datasets for training and evaluation.

Sequences are packed round-robin so batches fill the context window instead of padding it away. Optional filters drop CIFs longer than the context length and rows whose condition values are unusable. Both checks happen once, here, instead of per batch in the training loop.
"""

import numpy as np
import torch
import ast
import datasets

from _utils import (
    get_token_length_stats,
    filter_long_CIFs,
    filter_CIFs_with_unk,
    tokenize_function,
    validate_condition_values
)

NUM_PROC_TOK = 4
MISSING_CONDITION_VALUE = -100.0

np.random.seed(1)


def _pack_condition_values(values_list: list) -> torch.Tensor:
    """Tensorize per-sample conditions (ported from CrystaLLM-graph).

    A scalar condition is one row per sample so a plain stack works. A continuous XRD condition is a (1000, 2) table per sample, and uneven row counts make np.array build an object array that torch cannot use. Pad short tables up to the tallest, at the front with MISSING_CONDITION_VALUE where the models expect filler.
    """
    first_cond = values_list[0]

    is_nested = hasattr(first_cond, "__len__") and (
        (isinstance(first_cond, list) and isinstance(first_cond[0], list)) or
        (isinstance(first_cond, (np.ndarray, torch.Tensor)) and first_cond.ndim >= 2)
    )

    if is_nested:
        cond_tensors = [torch.tensor(cond, dtype=torch.float32) for cond in values_list]
        max_rows = max(t.shape[0] for t in cond_tensors)
        padded = []
        for cond_tensor in cond_tensors:
            n_rows = cond_tensor.shape[0]
            if n_rows < max_rows:
                padding = torch.full(
                    (max_rows - n_rows, cond_tensor.shape[1]), MISSING_CONDITION_VALUE, dtype=torch.float32
                )
                cond_tensor = torch.cat([padding, cond_tensor], dim=0)
            padded.append(cond_tensor)
        return torch.stack(padded)

    packed = torch.as_tensor(np.array(values_list), dtype=torch.float)
    if packed.dim() == 1:
        packed = packed.unsqueeze(-1)
    return packed


class CustomCIFDataCollator:
    """Collate tokenized CIF rows into fixed-length training batches.

    Every sequence in a batch is exactly `context_length` tokens. A CIF longer than that is sliced from the beginning; a shorter one is packed with further CIFs taken round-robin from the rest of the batch until the window is full. Packing rather than padding is what keeps the context window carrying real tokens instead of filler.

    Conditional mode is detected from the features themselves, by whether `condition_values` is present, so the same collator serves both training modes.
    """
    def __init__(self, tokenizer: "CustomCIFTokenizer", context_length: int) -> None:
        self.tokenizer = tokenizer
        self.context_length = context_length

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        """Pack a list of tokenized features into one batch.

        Conditional mode is detected from the first feature carrying condition_values. Raises ValueError if a packed sequence misses context_length, which would mean the packing loop left the batch ragged.

        Args:
            features: tokenized rows with input_ids, fixed_mask, optionally attention_mask, special_tokens_mask, condition_values

        Returns:
            dict of [B, context_length] tensors: input_ids, labels, fixed_mask, attention_mask, special_tokens_mask, plus condition_values when conditional. Labels are input_ids with pad set to -100
        """

        # Auto-detect conditional mode based on presence of condition_values
        is_conditional = "condition_values" in features[0]
        
        # Prepare batch containers
        batch_input_ids = []
        batch_fixed_mask = []
        batch_attention_mask = []
        batch_special_tokens_mask = []
        batch_condition_values = [] if is_conditional else None

        def slice_from_beginning(input_ids, fixed_mask, special_tokens_mask, attention_mask):
            if len(input_ids) > self.context_length:
                input_ids = input_ids[:self.context_length]
                fixed_mask = fixed_mask[:self.context_length]
                if special_tokens_mask is not None:
                    special_tokens_mask = special_tokens_mask[:self.context_length]
                attention_mask = attention_mask[:self.context_length]
            return input_ids, fixed_mask, special_tokens_mask, attention_mask

        # Convert to Python lists for indexing
        for feature in features:
            feature["input_ids"] = list(feature["input_ids"])
            feature["fixed_mask"] = list(feature["fixed_mask"])
            feature["attention_mask"] = list(feature.get("attention_mask", [1]*len(feature["input_ids"])))
            feature["special_tokens_mask"] = list(feature["special_tokens_mask"]) if "special_tokens_mask" in feature else None

        # Pack sequences
        for i in range(len(features)):
            if is_conditional:
                current_condition_values = features[i]["condition_values"]

            input_ids_i = features[i]["input_ids"]
            fixed_mask_i = features[i]["fixed_mask"]
            special_mask_i = features[i]["special_tokens_mask"]
            attention_mask_i = features[i]["attention_mask"]

            # Single long CIF - slice from beginning
            if len(input_ids_i) > self.context_length:
                packed_input_ids, packed_fixed_mask, packed_special_tokens_mask, packed_attention_mask = (
                    slice_from_beginning(input_ids_i, fixed_mask_i, special_mask_i, attention_mask_i)
                )
            else:
                # Pack multiple CIFs
                packed_input_ids = []
                packed_fixed_mask = []
                packed_special_tokens_mask = []
                packed_attention_mask = []

                current_idx = i
                while len(packed_input_ids) < self.context_length:
                    block_input_ids = features[current_idx]["input_ids"]
                    block_fixed_mask = features[current_idx]["fixed_mask"]
                    block_special_mask = features[current_idx]["special_tokens_mask"]
                    block_attention = features[current_idx]["attention_mask"]

                    # Trim block if it would exceed context_length
                    new_total_length = len(packed_input_ids) + len(block_input_ids)
                    if new_total_length > self.context_length:
                        space_left = self.context_length - len(packed_input_ids)
                        block_input_ids = block_input_ids[:space_left]
                        block_fixed_mask = block_fixed_mask[:space_left]
                        if block_special_mask is not None:
                            block_special_mask = block_special_mask[:space_left]
                        block_attention = block_attention[:space_left]

                    # Fill missing special_tokens_mask with zeros
                    if block_special_mask is None:
                        block_special_mask = [0] * len(block_input_ids)

                    # Extend packed sequences
                    packed_input_ids.extend(block_input_ids)
                    packed_fixed_mask.extend(block_fixed_mask)
                    packed_special_tokens_mask.extend(block_special_mask)
                    packed_attention_mask.extend(block_attention)

                    # Break if we've reached context_length
                    if len(packed_input_ids) >= self.context_length:
                        packed_input_ids = packed_input_ids[:self.context_length]
                        packed_fixed_mask = packed_fixed_mask[:self.context_length]
                        packed_special_tokens_mask = packed_special_tokens_mask[:self.context_length]
                        packed_attention_mask = packed_attention_mask[:self.context_length]
                        break

                    # Move to next feature in round-robin
                    current_idx = (current_idx + 1) % len(features)

                # Ensure special_tokens_mask exists
                if len(packed_special_tokens_mask) == 0:
                    packed_special_tokens_mask = [0] * len(packed_input_ids)

            # validation
            if len(packed_input_ids) != self.context_length:
                raise ValueError("Packed sequence length mismatch")

            # Add to batch
            batch_input_ids.append(packed_input_ids)
            batch_fixed_mask.append(packed_fixed_mask)
            batch_attention_mask.append(packed_attention_mask)
            batch_special_tokens_mask.append(packed_special_tokens_mask)
            
            if is_conditional:
                batch_condition_values.append(current_condition_values)

        # Convert to tensors
        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        batch_fixed_mask = torch.tensor(batch_fixed_mask, dtype=torch.long)
        batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        batch_special_tokens_mask = torch.tensor(batch_special_tokens_mask, dtype=torch.long)

        # Labels
        labels = batch_input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Build output batch
        batch = {
            "input_ids": batch_input_ids,
            "labels": labels,
            "fixed_mask": batch_fixed_mask,
            "attention_mask": batch_attention_mask,
            "special_tokens_mask": batch_special_tokens_mask
        }

        # Add condition values for conditional mode
        if is_conditional:
            batch["condition_values"] = _pack_condition_values(batch_condition_values)

        return batch

def load_data(
    tokenizer: "CustomCIFTokenizer", 
    dataset: object, 
    context_length: int, 
    mode: str="unconditional",
    condition_columns: list[str] | None=None,
    remove_CIFs_above_context: bool=False, 
    remove_CIFs_with_unk: bool=False,
    show_token_stats: bool=False,
    validate_conditions: bool=False
) -> tuple:
    """Tokenize a CIF dataset and build the collator that batches it.

    Args:
        tokenizer: CIF tokenizer, used for encoding and the pad token id
        dataset: HF dataset with a CIF text column, plus condition columns when conditional
        context_length: tokens per training sequence, packed to exactly this
        mode: "unconditional" or "conditional"
        condition_columns: columns holding the conditioning values, required when conditional
        remove_CIFs_above_context: drop over-length CIFs instead of slicing them
        remove_CIFs_with_unk: drop CIFs with unknown tokens, usually a vocabulary gap
        show_token_stats: print token length statistics
        validate_conditions: check condition values before training

    Returns:
        (tokenized_dataset, data_collator) ready for the HF Trainer
    """
    
    # Validate inputs
    if mode == "conditional" and condition_columns is None:
        raise ValueError(f"condition_columns must be provided for mode='{mode}'")

    # Parse condition columns for conditional mode
    parsed_condition_columns = None
    if mode == "conditional":
        try:
            parsed_condition_columns = ast.literal_eval(str(condition_columns))
        except Exception as e:
            raise ValueError(f"Error parsing condition_columns: {condition_columns}. {e}")

    # Create tokenizer function based on mode
    if mode == "unconditional":
        def tokenize_fn(examples):
            return tokenize_function(examples, tokenizer, context_length, mode="unconditional")
    elif mode == "conditional":
        def tokenize_fn(examples):
            return tokenize_function(examples, tokenizer, condition_columns, mode="conditional")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Handle dataset indexing for conditional mode
    if mode == "conditional" and validate_conditions:
        dataset_with_idx = dataset.map(
            lambda ex, idx: {"__raw_idx": idx},
            with_indices=True,
            batched=False
        )
    else:
        dataset_with_idx = dataset

    # Determine columns to keep based on mode
    if mode == "conditional":
        columns_to_keep = [
            "input_ids",
            "attention_mask",
            "fixed_mask",
            "special_tokens_mask",
            "condition_values",
        ]
        if validate_conditions:
            columns_to_keep.append("__raw_idx")
    else:  # unconditional
        columns_to_keep = [
            "input_ids",
            "token_type_ids",
            "attention_mask",
            "fixed_mask",
            "special_tokens_mask",
        ]

    # Calculate columns to remove (reduce memory usage)
    original_columns = dataset_with_idx["train"].column_names
    if mode == "conditional":
        columns_to_remove = [
            col for col in original_columns 
            if col not in columns_to_keep and col not in parsed_condition_columns and col != "CIF"
        ]
    else:
        columns_to_remove = [col for col in original_columns if col not in columns_to_keep]

    # Tokenize dataset
    tokenized_dataset = dataset_with_idx.map(
        tokenize_fn,
        batched=True,
        remove_columns=columns_to_remove,
        num_proc=NUM_PROC_TOK
    )


    # Few filters/validations/stats
    if remove_CIFs_above_context:
        print("\n******* Tokenization Filters Stats *******")
        print(f"Removing CIFs with token length exceeding {context_length}")
        tokenized_dataset = filter_long_CIFs(tokenized_dataset, context_length)

    if remove_CIFs_with_unk:
        print("Removing CIFs with unknown tokens")
        tokenized_dataset = filter_CIFs_with_unk(tokenized_dataset, tokenizer)

    if "fixed_mask" not in tokenized_dataset["train"].features:
        raise ValueError("Fixed mask not present in tokenized dataset")

    if show_token_stats:
        print("\n******* Token Length Stats *******")
        get_token_length_stats(tokenized_dataset)

    if mode == "conditional" and validate_conditions:
        validate_condition_values(tokenized_dataset, dataset_with_idx, parsed_condition_columns)

    data_collator = CustomCIFDataCollator(tokenizer, context_length)
    return tokenized_dataset, data_collator