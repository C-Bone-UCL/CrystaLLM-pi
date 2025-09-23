import logging
import _tokenizer
from importlib import reload

reload(_tokenizer)

logger = logging.getLogger(__name__)

def tokenizer_ID_check(args, tokenized_dataset, tokenizer):
    print("\nChecking dataset and vocab tokens are consistent")
    # Check for out-of-range token IDs
    all_ids = []
    for example in tokenized_dataset["train"]:
        all_ids.extend(example["input_ids"])
    max_id = max(all_ids)
    if max_id >= len(tokenizer):
        raise ValueError(
            f"Found out-of-range token ID ({max_id}) in the dataset. Tokenizer vocab size is {len(tokenizer)}."
        )
    else:
        print("No out-of-range token IDs found. Continuing...")