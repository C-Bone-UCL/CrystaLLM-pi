"""
Create vocabulary file for the tokenizer. CIF specific vocab and syntax.

If you want to add new tokens, you can put therm in the 
respective sections in the code below.

Then you run the code to get new vocabulary file., which you should plug into 
_utils/_preprocessing/_save_tokenizer_to_HF.py
"""

import json

INPUT_SPACE_GROUPS_FILE = "_utils/_tokenizer_utils/spacegroups.txt"
OUTPUT_VOCAB_FILE = "_utils/_tokenizer_utils/vocabulary.json"

# generates the vocabulary for the tokenizer
def generate_vocabulary():
    # Step 1: Generate cif_tokens.txt
    tokens = []

    # Add Atoms
    tokens.extend(["Si", "C", "Pb", "I", "Br", "Cl", "Eu", "O", "Fe", "Sb", "In", "S", "N", "U", "Mn", "Lu", "Se", "Tl", "Hf",
            "Ir", "Ca", "Ta", "Cr", "K", "Pm", "Mg", "Zn", "Cu", "Sn", "Ti", "B", "W", "P", "H", "Pd", "As", "Co", "Np",
            "Tc", "Hg", "Pu", "Al", "Tm", "Tb", "Ho", "Nb", "Ge", "Zr", "Cd", "V", "Sr", "Ni", "Rh", "Th", "Na", "Ru",
            "La", "Re", "Y", "Er", "Ce", "Pt", "Ga", "Li", "Cs", "F", "Ba", "Te", "Mo", "Gd", "Pr", "Bi", "Sc", "Ag", "Rb",
            "Dy", "Yb", "Nd", "Au", "Os", "Pa", "Sm", "Be", "Ac", "Xe", "Kr", "He", "Ne", "Ar"])

    # Add Digits
    tokens.extend([str(d) for d in list(range(10))])

    # Add Keywords
    tokens.extend([
        "_cell_length_b",
        "_atom_site_occupancy",
        "_atom_site_attached_hydrogens",
        "_cell_length_a",
        "_cell_angle_beta",
        "_symmetry_equiv_pos_as_xyz",
        "_cell_angle_gamma",
        "_atom_site_fract_x",
        "_symmetry_space_group_name_H-M",
        "_symmetry_Int_Tables_number",
        "_chemical_formula_structural",
        "_chemical_name_systematic",
        "_atom_site_fract_y",
        "_atom_site_symmetry_multiplicity",
        "_chemical_formula_sum",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_cell_length_c",
        "_atom_site_B_iso_or_equiv",
        "_symmetry_equiv_pos_site_id",
        "_cell_volume",
        "_atom_site_fract_z",
        "_cell_angle_alpha",
        "_cell_formula_units_Z",
        "loop_",
        "data_"
    ])

    # Add Extended Keywords
    tokens.extend([
        "_atom_type_symbol",
        "_atom_type_electronegativity",
        "_atom_type_radius",  
        "_atom_type_ionic_radius",  
        "_atom_type_oxidation_number"
    ])

    # symbols
    tokens.extend(["x", "y", "z", ".", "(", ")", "+", "-", "/", "'", ",", " ", "\n"])

    # Add Space Groups by loading from utils/space_groups.txt
    space_groups = []
    with open(INPUT_SPACE_GROUPS_FILE, "r") as f:
        for line in f:
            space_groups.append(line.strip())
    tokens.extend([sg+'_sg' for sg in space_groups])

    vocab = {token: i for i, token in enumerate(tokens)}

    # Write to file in a json format with mapping from token to index
    with open(OUTPUT_VOCAB_FILE, "w") as f:
        json.dump(vocab, f)

    return

if __name__ == "__main__":
    # generate the vocabulary
    generate_vocabulary()
    print(f"Vocabulary generated successfully and saved to {OUTPUT_VOCAB_FILE}")

