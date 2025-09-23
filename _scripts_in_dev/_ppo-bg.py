#############################
# This script is under development and may not be fully functional.
# Testing PPO with ehull, unique, novel, valid rewards?
#############################

import torch
from tqdm import tqdm
import pandas as pd
import os
import argparse
import collections
import shutil
import json
import sys

tqdm.pandas()

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, set_seed

from _tokenizer import CustomCIFTokenizer
from _utils._evaluation_og.postprocess import postprocess
from _utils._evaluation_conditional.orb_ehull import make_ehull_reward_fn
from _utils import (
    is_formula_consistent,
    is_atom_site_multiplicity_consistent,
    is_space_group_consistent,
    bond_length_reasonableness_score,
    extract_formula_nonreduced,
)
from pymatgen.core import Composition, Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator


def is_valid(cif_str, bond_length_acceptability_cutoff=1.0) -> bool:
    """Checks if a CIF string represents a valid crystal structure based on several criteria."""
    try:
        if not is_formula_consistent(cif_str):
            # print("Debug: Failed formula consistency")
            return False
        if not is_atom_site_multiplicity_consistent(cif_str):
            # print("Debug: Failed multiplicity consistency")
            return False
        bond_length_score = bond_length_reasonableness_score(cif_str)
        if bond_length_score < bond_length_acceptability_cutoff:
            # print(f"Debug: Failed bond length score: {bond_length_score}")
            return False
        if not is_space_group_consistent(cif_str):
            # print("Debug: Failed space group consistency")
            return False
        return True
    except Exception as e:
        # print(f"Debug: Validity check failed with exception: {e}")
        return False

# Uniqueness
def initialize_structure_matcher(l_tol=0.2, s_tol=0.3, angle_tol=5):
    """Initializes the Pymatgen StructureMatcher with error handling."""
    try:
        # Standard initialization
        return StructureMatcher(ltol=l_tol, stol=s_tol, angle_tol=angle_tol)
    except TypeError as e:
        print(f"Warning: Pymatgen StructureMatcher initialization failed with TypeError: {e}")
        print("Trying older initialization fallback (may occur in older pymatgen versions)")
        try:
            return StructureMatcher(ltol=l_tol, stol=s_tol, angle_tol=angle_tol, primitive_cell=True)
        except Exception as init_e:
            print(f"Error: Could not initialize StructureMatcher. Pymatgen issue? Error: {init_e}")
            exit(1) # Exit if matcher cannot be initialized
    except Exception as e:
        print(f"Error: Could not initialize StructureMatcher. Error: {e}")
        exit(1) # Exit if matcher cannot be initialized

# persistent_unique_structures = collections.defaultdict(list)

# def is_unique(cif_string: str, struct_matcher: StructureMatcher) -> bool:
#     """
#     Checks if a generated CIF structure is unique within its composition
#     compared to previously validated unique structures in the current run.
#     Updates the global 'persistent_unique_structures' dictionary.
#     """
#     global persistent_unique_structures

#     parsed_structure = None
#     try:
#         # Attempt to parse the CIF string into a Pymatgen Structure object
#         parsed_structure = Structure.from_str(cif_string, fmt="cif")
#         # Generate a standardized formula string for dictionary keying
#         composition_str = parsed_structure.composition.formula.replace(" ", "")
#     except Exception as e:
#         # If parsing fails, it cannot be considered unique or valid in this context
#         # print(f"Debug: Structure parsing failed for uniqueness check: {e}")
#         return False

#     # Retrieve the list of known unique structures for this composition
#     existing_structures_for_composition = persistent_unique_structures[composition_str]

#     is_unique_structure = True
#     if not existing_structures_for_composition:
#         # If no structures exist for this composition yet, it's unique by default
#         is_unique_structure = True
#     else:
#         # Compare against each existing structure for this composition
#         for existing_struct in existing_structures_for_composition:
#             try:
#                 # Use the structure matcher to check if the new structure matches an existing one
#                 if struct_matcher.fit(parsed_structure, existing_struct):
#                     # Match found, structure is not unique
#                     is_unique_structure = False
#                     # print(f"Debug: Structure matched existing one for composition {composition_str}")
#                     break # No need to check further
#             except Exception as e:
#                 # Handle potential errors during structure matching
#                 # print(f"Warning: StructureMatcher comparison error for {composition_str}. Error: {e}")
#                 # Treat match errors cautiously; consider it not unique to avoid adding problematic structures
#                 is_unique_structure = False
#                 break

#     # If the structure was determined to be unique, add it to the dictionary
#     if is_unique_structure:
#         persistent_unique_structures[composition_str].append(parsed_structure)
#         # print(f"Debug: Added unique structure for composition {composition_str}")
#         return True
#     else:
#         return False

def process_chunk(pairs, device_id, ref_data, inner_batch):
    device = f"cuda:{device_id}"
    orbff = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-high")
    calc = ORBCalculator(orbff, device=device)
    batch_hull = make_ehull_reward_fn(calc, ref_data, batch=inner_batch, n_jobs=1)

    indices, cif_strings = zip(*pairs)
    ehull_vals = batch_hull(list(cif_strings))
    return list(zip(indices, ehull_vals))


def main(args):
    """Runs the PPO training process."""

    print("loading reference data from alexandrias ehull file (≈45 s)")
    ref_path = "_utils/_evaluation_conditional/alignn_model_ckpts/convex_hull_pbe.json"
    with open(ref_path, "r") as f:
        ref_data = json.load(f)

    print("Initializing PPOConfig...")
    config = PPOConfig(
        model_name=args.model_checkpoint_path,
        learning_rate=args.learning_rate,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=False,
        kl_penalty="kl",
        seed=args.seed,
        steps=args.steps,
        tracker_project_name=args.tracker_project_name,
        log_with=args.log_with,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        init_kl_coef=args.kl_coef,
        cliprange=args.cliprange,
        cliprange_value=args.cliprange_value,
    )
    # Set seed for reproducibility
    set_seed(config.seed)

    # Model and Tokenizer Loading
    print(f"Loading model from: {args.model_checkpoint_path}")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_checkpoint_path)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_checkpoint_path)

    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = CustomCIFTokenizer.from_pretrained(
        pretrained_dir=args.tokenizer_path,
        pad_token="<pad>"
    )

    # Ensure tokenizer pad token ID matches model config if necessary
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token # Or define a specific pad token
        print(f"Warning: pad_token_id not set, setting to eos_token_id: {tokenizer.pad_token_id}")

    # Set pad_token_id in model config if needed by generate
    model.config.pad_token_id = tokenizer.pad_token_id
    ref_model.config.pad_token_id = tokenizer.pad_token_id

    # Generation args
    generation_kwargs = {
        "min_length": -1,
        "max_length": args.max_length,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # prompt
    query_txt = "<bos>\ndata_["

    query_tensor = tokenizer.encode(query_txt, return_tensors="pt").squeeze(0)
    print(f"Query text: '{query_txt}'")
    print(f"Tokenized query tensor shape: {query_tensor.shape}")


    # initialise PPOTrainer
    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=None,
        data_collator=None,
    )
    print("PPOTrainer initialized.")

    device = ppo_trainer.accelerator.device
    print(f"Using device: {device}")

    # Move fixed query tensor to device once
    query_tensor = query_tensor.to(device)

    # Load the ORB calculator
    orbff = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-high")
    calc = ORBCalculator(orbff, device=device)
    print("ORB calculator initialized.")

    # Load the alignn model for bg
    pex_path = "/home/cyprien/CrystaLLMv2_PKV/_utils/_evaluation_conditional/alignn_pex/alignn_predictor.pex"
    sys.path.append(pex_path)
    from predictor import get_figshare_model, make_reward_bg
    alignn_model = get_figshare_model(model_name="mp_gappbe_alignn")
    print("ALIGNN model initialized.")

    # For uniqueness
    L_TOL = 0.2
    S_TOL = 0.3
    ANGLE_TOL = 5
    struct_matcher = initialize_structure_matcher(l_tol=L_TOL, s_tol=S_TOL, angle_tol=ANGLE_TOL)
    print("StructureMatcher initialized.")

    print(f"Starting PPO training for {config.steps} steps...")
    global_step = 0

    # for uniqueness
    # persistent_unique_structures.clear()

    for step in tqdm(range(config.steps), desc="PPO Steps"):
        global_step += 1
        batch = {}

        # Prepare batch of query tensors
        queries_list = [query_tensor for _ in range(config.batch_size)]
        batch["query"] = [query_txt] * config.batch_size

        # response generation
        response_tensors_list = ppo_trainer.generate(
            queries_list,
            return_prompt=False,
            length_sampler=None,
            **generation_kwargs,
        )

        # process response and make reward
        batch["response"] = []
        rewards = []
        valid_count = 0
        unique_count = 0
        error_count = 0
        stable_count = 0

        processed_responses_for_step = []

        # make an empty structure list for uniqueness (this will reset every batch)
        unique_processed_structures = []
        unique_formulas = []

        for i in range(config.batch_size):
            # Get the tensor for the current response
            response_tensor = response_tensors_list[i]

            # Truncate at the first EOS token
            eos_indices = (response_tensor == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_indices) > 0:
                response_tensor = response_tensor[:eos_indices[0]]
                # print(f"Debug: Response {i} truncated at EOS index {eos_indices[0].item()}")

            # Keep the processed tensor for the PPO step
            processed_responses_for_step.append(response_tensor.to(device))

            # Decode for validation and logging
            # Move tensor to CPU for decoding if tokenizer expects CPU tensors
            response_str_decoded = tokenizer.decode(response_tensor.cpu(), skip_special_tokens=True)

            # Reconstruct full CIF for validation
            full_cif_str = "data_" + response_str_decoded

            # Calculate Reward
            reward_value = 0.0
            is_valid_flag = False
            is_unique_flag = False

            try:
                # Apply postprocessing first
                processed_cif_str = postprocess(full_cif_str)
                batch["response"].append(processed_cif_str)

                # Check validity using the dedicated function
                is_valid_flag = is_valid(processed_cif_str)

                if is_valid_flag:
                    reward_value = 1.0
                    valid_count += 1
                    print(f"Debug: Valid CIF for {i}: {processed_cif_str}")

                    # Check uniqueness
                    parsed_structure = Structure.from_str(processed_cif_str, fmt="cif")
                    composition_str = parsed_structure.composition.formula.replace(" ", "")
                    if composition_str not in unique_formulas:
                        unique_formulas.append(composition_str)
                        unique_processed_structures.append(parsed_structure)
                        is_unique_flag = True
                    else:
                        if struct_matcher.fit(parsed_structure, unique_processed_structures[unique_formulas.index(composition_str)]):
                            is_unique_flag = False
                        else:
                            is_unique_flag = True
                            unique_processed_structures.append(parsed_structure)

                    if is_unique_flag:
                        unique_count += 1
                        reward_value += 1.0
                        print(f"Debug: Unique CIF for {i}: {processed_cif_str}")
                    else:
                        reward_value -= 0.5   
                        print(f"Debug: Non-unique CIF for {i}: {processed_cif_str}")                     

                    # make a reward for sability
                    e_hull = make_ehull_reward_fn(calc, ref_data, batch=1, n_jobs=1)
                    ehull_val = e_hull([processed_cif_str])[0]
                    if ehull_val < 0.1:
                        reward_value += 1.0
                        stable_count += 1
                        print(f"Debug: Stable CIF for {i}: {processed_cif_str}")
                    else:
                        reward_value -= 0.5
                        print(f"Debug: Unstable CIF for {i}: {processed_cif_str}")

                    # make a reward for bandgap
                    bandgap_value = make_reward_bg(processed_cif_str, alignn_model)
                    if bandgap_value > 1.5:
                        reward_value += 1.0
                        print(f"Debug: High bandgap for {i}: {processed_cif_str}")
                    elif bandgap_value < 0.5:
                        reward_value -= 0.5
                        print(f"Debug: Low bandgap for {i}: {processed_cif_str}")

                else:
                    reward_value = -1.0


            except Exception as e:
                # Catch errors during postprocessing or validation
                print(f"Warning: Error processing generated CIF (Index {i}): {e}")
                # Ensure a response string is added even if processing fails
                if len(batch["response"]) == i:
                     batch["response"].append("[ERROR DURING PROCESSING]")
                reward_value = -1.0
                error_count += 1

            rewards.append(torch.tensor(reward_value, device=device)) 

        # PPO step
        try:
            # Prepare lists of tensors
            # Ensure query tensors are lists of tensors, not a single stacked tensor
            queries_list_tensors = [q.to(device) for q in queries_list]

            # Run PPO optimization step
            stats = ppo_trainer.step(queries_list_tensors, processed_responses_for_step, rewards)

            # Log stats (rewards are passed again here for logging, ensure it's a list of tensors)
            ppo_trainer.log_stats(stats, batch, rewards)


        except Exception as e:
            print(f"\nError during PPO step {global_step}: {e}")
            import traceback
            traceback.print_exc()
            print("Skipping PPO step {global_step} due to error.")
            continue

        # Print summary stats
        avg_reward = torch.mean(torch.stack(rewards)).item()
        print(f"Step {global_step}/{config.steps} | Avg Reward: {avg_reward:.2f} | Valid: {valid_count}/{config.batch_size} \
                | Unique: {unique_count}/{config.batch_size} | Errors: {error_count}/{config.batch_size}")


        # Save model checkpoint every N steps
        if global_step % args.save_every == 0 and global_step > 0:
            print(f"\nSaving model checkpoint at step {global_step}...")
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            try:
                # Make sure the directory exists
                os.makedirs(save_path, exist_ok=True)
                # Save model using ppo_trainer's method which saves both model and value head
                ppo_trainer.save_pretrained(save_path)
                # Save tokenizer separately
                tokenizer.save_pretrained(save_path)
                print(f"Model and tokenizer saved to {save_path}")

                # List all checkpoint directories in the output folder
                all_checkpoints = sorted(
                    [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, d))],
                    key=lambda x: int(x.split("-")[-1]) # Sort by step number
                )
                # Keep only latest 2 checkpoints
                checkpoints_to_keep = 2
                while len(all_checkpoints) > checkpoints_to_keep:
                    oldest = all_checkpoints.pop(0)
                    oldest_path = os.path.join(args.output_dir, oldest)
                    try:
                        shutil.rmtree(oldest_path)
                        print(f"Removed old checkpoint: {oldest_path}")
                    except OSError as e:
                        print(f"Error removing old checkpoint {oldest_path}: {e}")

            except Exception as save_e:
                print(f"Error saving checkpoint at step {global_step}: {save_e}")


    print("\nPPO training finished.")

    # Final save
    final_save_path = os.path.join(args.output_dir, "final_model")
    print(f"Saving final tuned model to {final_save_path}...")
    try:
        os.makedirs(final_save_path, exist_ok=True)
        ppo_trainer.save_pretrained(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        print(f"Final model and tokenizer saved successfully to {final_save_path}")
    except Exception as final_save_e:
        print(f"Error saving final model: {final_save_e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training for CIF Generation")

    # Paths
    parser.add_argument("--model_checkpoint_path", type=str, required=True, help="Path to the pretrained base model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the pretrained tokenizer")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save trained models and logs")

    # PPO Hyperparameters - Updated defaults reflecting recommendations
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for PPO optimizer (default: 5e-6)")
    parser.add_argument("--ppo_epochs", type=int, default=3, help="Number of optimization epochs per PPO step (default: 1)")
    parser.add_argument("--target_kl", type=float, default=0.02, help="Target KL divergence for penalty (default: 0.02)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping (default: 1.0)")
    parser.add_argument("--vf_coef", type=float, default=0.1, help="Value function coefficient (default: 0.1)")
    parser.add_argument("--kl_coef", type=float, default=0.1, help="KL coefficient (default: 0.1)")
    parser.add_argument("--cliprange", type=float, default=0.2, help="PPO clip range (default: 0.2)")
    parser.add_argument("--cliprange_value", type=float, default=0.2, help="PPO clip range for value function (default: 0.2)")
    parser.add_argument("--max_length", type=int, default=1015, help="Max length for generation (default: 1015)")
    
    # Other PPO Hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for generation and PPO")
    parser.add_argument("--mini_batch_size", type=int, default=4, help="Mini-batch size for PPO updates (ensure batch_size % mini_batch_size == 0)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps for gradient accumulation")
    parser.add_argument("--steps", type=int, default=2000, help="Total number of PPO training steps")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")

    # Logging and Saving
    parser.add_argument("--save_every", type=int, default=250, help="Save model checkpoint every N steps") # Kept original freq
    parser.add_argument("--tracker_project_name", type=str, default="ppo_cif_generation_stable", help="Project name for tracking (e.g., WandB)")
    parser.add_argument("--log_with", type=str, default='wandb', help="Logging integration ('wandb', 'tensorboard', or None)") # Defaulting to wandb as used before

    args = parser.parse_args()

    # Basic validation
    if args.batch_size % args.mini_batch_size != 0:
        print(f"Warning: batch_size ({args.batch_size}) should be divisible by mini_batch_size ({args.mini_batch_size})")
        args.mini_batch_size = args.batch_size

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print("Starting PPO training with the following arguments:")
    for k, v in vars(args).items():
        print(f"- {k}: {v}")

    main(args)