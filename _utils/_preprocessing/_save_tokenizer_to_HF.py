"""
Script to save a custom CIF tokenizer to a local directory and optionally push it to the Hugging Face Hub.
"""

import logging
from typing import Optional
from _tokenizer import CustomCIFTokenizer
from huggingface_hub import login
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _utils import load_api_keys

logger = logging.getLogger(__name__)

# Test CIF data for tokenizer validation
TEST_CIF_DATA = '''<bos>
data_[Zr8Ti8Au8]
loop_
 _atom_type_symbol
 _atom_type_electronegativity
 _atom_type_radius
 _atom_type_ionic_radius
[
  Zr  1.3300  1.5500  0.8600
  Ti  1.5400  1.4000  0.8517
  Au  2.5400  1.3500  1.0700
]
_symmetry_space_group_name_H-M [Amm2]
_cell_length_a [8.7641]
_cell_length_b [5.5973]
_cell_length_c [9.4061]
_cell_angle_alpha [90.0000]
_cell_angle_beta [90.0000]
_cell_angle_gamma [90.0000]
_symmetry_Int_Tables_number [38]
_chemical_formula_structural [ZrTiAu]
_chemical_formula_sum '[Zr8 Ti8 Au8]'
_cell_volume [461.4175]
_cell_formula_units_Z [8]
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
[
  Zr  Zr0  4  0.1830  0.0000  0.3375  1
  Zr  Zr1  4  0.3123  0.0000  0.6770  1
  Ti  Ti2  4  0.0000  0.2402  0.5867  1
  Ti  Ti3  2  0.0000  0.0000  0.8197  1
  Ti  Ti4  2  0.5000  0.0000  0.1591  1
  Au  Au5  4  0.2430  0.0000  0.0062  1
  Au  Au6  4  0.5000  0.2425  0.4033  1
]
<eos>
'''


def test_tokenizer(tokenizer: CustomCIFTokenizer) -> None:
    """Test the tokenizer with sample CIF data to verify functionality."""
    print('\nTesting tokenizer')
    
    # Tokenize and decode, show individual tokens in quotes
    print(TEST_CIF_DATA)
    print("tokenized text", tokenizer.tokenize(TEST_CIF_DATA))
    print("what those tokens correspond to as strings:", tokenizer.convert_tokens_to_string(tokenizer.tokenize(TEST_CIF_DATA)))

    encoded = tokenizer.encode(TEST_CIF_DATA)
    decoded = tokenizer.decode(encoded)
    print("Encoded:", encoded)
    print("Decoded:", decoded)


def save_HFtokenizer_locally(
        vocab_file: str = "_utils/_tokenizer_utils/vocabulary.json",
        spacegroups_file: str = "_utils/_tokenizer_utils/spacegroups.txt",
        path: str = "HF_tokenizer",
        hub_path: str = "c-bone/cif-tokenizer",
        API_key: Optional[str] = None,
        push_to_hub: bool = False,
        testing: bool = True) -> None:

    """Save a CIF tokenizer locally and optionally push to Hugging Face Hub."""
    tokenizer = CustomCIFTokenizer(
        vocab_file=vocab_file,
        spacegroups_file=spacegroups_file
    )

    if testing:
        test_tokenizer(tokenizer)

    print('\nSaving tokenizer')
    tokenizer.save_pretrained(path)

    if push_to_hub:
        print('Pushing to hub')
        login(API_key)
        tokenizer.push_to_hub(hub_path)
    else:
        print('Not pushing to hub')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Save the CIF tokenizer to a local directory.")
    parser.add_argument("--vocab_file", type=str, default="_utils/_tokenizer_utils/vocabulary.json", help="Path to vocabulary file.")
    parser.add_argument("--spacegroups_file", type=str, default="_utils/_tokenizer_utils/spacegroups.txt", help="Path to spacegroups file.")
    parser.add_argument("--path", type=str, default="HF_tokenizer", help="Local save path.")
    parser.add_argument("--hub_path", type=str, default="c-bone/cif-tokenizer", help="Hugging Face Hub path.")
    parser.add_argument("--push_to_hub", action="store_true", help="Push to Hugging Face Hub.")
    parser.add_argument("--testing", action="store_true", help="Test the tokenizer.")
    args = parser.parse_args()

    # Load API key if needed for hub push
    api_key = None
    if args.push_to_hub:
        api_keys = load_api_keys()
        api_key = api_keys.get("HF_key")

    save_HFtokenizer_locally(
        vocab_file=args.vocab_file,
        spacegroups_file=args.spacegroups_file,
        path=args.path,
        hub_path=args.hub_path,
        API_key=api_key,
        push_to_hub=args.push_to_hub,
        testing=args.testing
    )

    print("\nDone.")
