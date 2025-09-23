"""
CIF generation script for conditional and unconditional crystal structure generation using transformer models.
Supports PKV, Prepend, Slider, and Raw conditioning (can be done on multiple GPUs in parallel).

Pass a .jsonc config file with required arguments:
    python generate_CIFs.py --config your_config.jsonc

Required config arguments:
    - model_ckpt_dir: Path to trained model checkpoint
    - input_parquet: Input prompts/conditions file
    - output_parquet: Output CIFs file
    - pretrained_tokenizer_dir: CIF tokenizer directory (usually HF-cif-tokenizer)

Optional config arguments:
    - activate_conditionality: PKV/Prepend/Slider/Raw (default: None for unconditional)
    - gen_max_length: Max generation length (default: 1024)
    - do_sample: True/False/beam (default: True)
    - num_return_sequences: Sequences per GPU pass (default: 1)
    - num_repeats: Repeat generations for memory management (default: 1)
"""

import os
import multiprocessing as mp
import sys
import json
import time
import re
import warnings
import gc
from tqdm import tqdm
import torch
import pandas as pd
from transformers import GPT2LMHeadModel

# Global constants
DEFAULT_MAX_LENGTH = 1024
TOKENIZER_PAD_TOKEN = "<pad>"
CONDITIONAL_MODELS = ["PKV", "Prepend", "Slider"]
DEFAULT_TOKENIZER_DIR = "HF-cif-tokenizer"

# Global warning filters
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _tokenizer import CustomCIFTokenizer
from _models import PKVGPT, PrependGPT, SliderGPT
from _args import parse_args
from _utils import find_checkpoint_from_dir

global_model = None
global_tokenizer = None

def init_tokenizer(pretrained_tokenizer_dir):
    """Initialize tokenizer with standard config."""
    tokenizer = CustomCIFTokenizer.from_pretrained(
        pretrained_dir=pretrained_tokenizer_dir,
        pad_token=TOKENIZER_PAD_TOKEN
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return tokenizer

def get_model_class(conditionality_type):
    """Return the appropriate model class based on the conditionality type."""
    if conditionality_type == "PKV":
        return PKVGPT
    elif conditionality_type == "Prepend":
        return PrependGPT
    elif conditionality_type == "Slider":
        return SliderGPT
    else:
        # Default to GPT2LMHeadModel for unconditional or raw generation
        return GPT2LMHeadModel

def get_model_max_length(model_ckpt_dir, activate_conditionality):
    """Get model's max length from config by loading temp model."""
    model_class = get_model_class(activate_conditionality)
    
    # Handle Raw and unconditional cases
    if activate_conditionality == "Raw" or activate_conditionality is None:
        model_class = GPT2LMHeadModel
    elif model_class == GPT2LMHeadModel and activate_conditionality not in [None, "Raw"]:
         print(f"Warning: Unsupported conditionality {activate_conditionality}, using default max_length")
         return DEFAULT_MAX_LENGTH

    try:
        temp_model = model_class.from_pretrained(model_ckpt_dir, low_cpu_mem_usage=True)
        n_positions = temp_model.config.n_positions
        del temp_model
        return n_positions
    except Exception as e:
        print(f"Error loading model to get max length: {e}")
        # Fallback or error handling
        return DEFAULT_MAX_LENGTH


def build_generation_kwargs(args, tokenizer, max_length):
    """Build generation kwargs based on args and tokenizer."""
    base_kwargs = {
        "max_length": min(args.gen_max_length, max_length),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "renormalize_logits": True,
        "remove_invalid_values": True,
    }
    
    do_sample_str = str(args.do_sample).lower()
    
    if "true" in do_sample_str:
        base_kwargs.update({
            "num_return_sequences": args.num_return_sequences,
            "do_sample": True,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
        })
    elif "false" in do_sample_str:
        base_kwargs.update({
            "num_return_sequences": 1,
            "do_sample": False,
            "top_k": 0,
            "top_p": 1.0,
            "temperature": 1.0,
        })
    elif args.do_sample == "beam":
        base_kwargs.update({
            "num_return_sequences": args.num_return_sequences,
            "do_sample": False,
            "num_beams": args.num_return_sequences,
            "top_k": 0,
            "top_p": 1.0,
            "temperature": 1.0,
        })
    
    return base_kwargs

def remove_conditionality(cif_str: str) -> str:
    """Remove comments preceding the 'data_' block in CIF string for Raw conditioning."""
    match = re.search(r'(data_.*)', cif_str, re.DOTALL)
    return match.group(1) if match else cif_str

def setup_device(gpu_id):
    """Setup device and return appropriate torch device."""
    device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device

def parse_condition_vector(condition_vector):
    """Parse condition vector string into tensor values."""
    if condition_vector in (None, "None"):
        return None
    
    if "," in str(condition_vector):
        return [float(x.strip()) for x in str(condition_vector).split(",")]
    else:
        return [float(condition_vector)]

def determine_material_id(row, generated_count, global_offset=0):
    """Determine material ID from row data."""
    if "Material ID" in row:
        return row["Material ID"]
    elif "Formula" in row:
        return row["Formula"]
    else:
        return f"Generated_{generated_count + global_offset + 1}"

def init_worker(model_ckpt_dir, pretrained_tokenizer_dir, activate_conditionality):
    global global_model, global_tokenizer
    
    global_tokenizer = init_tokenizer(pretrained_tokenizer_dir)
    
    # Load models based on conditionality type
    if activate_conditionality == "PKV":
        global_model = PKVGPT.from_pretrained(model_ckpt_dir).eval()
        print("Loading model with PKV conditionality...")
    elif activate_conditionality == "Prepend":
        global_model = PrependGPT.from_pretrained(model_ckpt_dir).eval()
        print("Loading model with Prepend conditionality...")
    elif activate_conditionality == "Slider":
        global_model = SliderGPT.from_pretrained(model_ckpt_dir).eval()
        print("Loading model with Slider conditionality...")
    elif activate_conditionality == "Raw" or activate_conditionality is None:
        global_model = GPT2LMHeadModel.from_pretrained(model_ckpt_dir).eval()
        print("Loading standard model...")
    else:
        # Fallback to standard GPT2 model for unknown conditionality types
        global_model = GPT2LMHeadModel.from_pretrained(model_ckpt_dir).eval()
        print(f"Unknown conditionality '{activate_conditionality}', loading standard model...")
    
    global_model.resize_token_embeddings(len(global_tokenizer))

def progress_listener(queue, total):
    pbar = tqdm(total=total, desc="Generating CIFs...")
    while True:
        message = queue.get()
        if message == "kill":
            break
        pbar.update(message)
    pbar.close()

def generate_on_gpu(
    gpu_id,
    prompts,
    generation_kwargs,
    queue,
    start_idx,
    end_idx,
    model_ckpt_dir,
    activate_conditionality,
    num_repeats,
    global_offset=0,
    max_attempts=5,
):
    global global_model, global_tokenizer
    
    device = setup_device(gpu_id)
    model = global_model.to(device)
    tokenizer = global_tokenizer
    generated = []
    
    torch.cuda.manual_seed_all(int(time.time()) + os.getpid())
    
    for rep in range(num_repeats):
        for idx in range(start_idx, end_idx):
            row = prompts.iloc[idx]
            input_ids = tokenizer.encode(row["Prompt"], return_tensors="pt").to(device)
            
            # Handle different conditionality types
            if activate_conditionality in ["PKV", "Prepend", "Slider"]:
                # Parse condition vector for conditional models
                condition_tensor = None
                if row["condition_vector"] not in (None, "None"):
                    values = parse_condition_vector(row["condition_vector"])
                    if values:
                        condition_tensor = torch.tensor([values], device=device)
                
                # Generate with conditional models
                required = generation_kwargs.get("num_return_sequences", 1)
                produced = 0
                attempts = 0
                
                while produced < required and attempts < max_attempts:
                    attempts += 1
                    try:
                        with torch.no_grad():
                            outputs = model.generate(
                                input_ids=input_ids,
                                condition_values=condition_tensor,
                                **generation_kwargs,
                            )
                    except RuntimeError as e:
                        if device.type == "cuda" and "device-side assert" in str(e):
                            print(f"CUDA error on GPU {gpu_id}: {e}")
                            device = torch.device("cpu")
                            # Reload model on CPU
                            model_class = get_model_class(activate_conditionality)
                            model = model_class.from_pretrained(model_ckpt_dir).eval().to(device)
                            model.resize_token_embeddings(len(tokenizer))
                            input_ids = input_ids.cpu()
                            if condition_tensor is not None:
                                condition_tensor = condition_tensor.cpu()
                            continue
                    
                    # Process outputs with validation
                    for output in outputs:
                        if torch.isnan(output).any() or torch.isinf(output).any():
                            continue
                        
                        eos_idx = (output == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                        if eos_idx.numel() > 0:
                            output = output[: int(eos_idx[0])]
                        
                        cif_txt = tokenizer.decode(output, skip_special_tokens=True).replace("\n\n", "\n")
                        mid = determine_material_id(row, len(generated), global_offset)
                        
                        generated.append({
                            "Material ID": mid,
                            "Prompt": row["Prompt"],
                            "Generated CIF": cif_txt,
                            "condition_vector": row.get("condition_vector", "None"),
                        })
                        queue.put(1)
                        produced += 1
                        
            else:
                # Handle Raw or unconditional generation
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        **generation_kwargs,
                    )
                
                for output in outputs:
                    eos_idx = (output == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                    if eos_idx.numel() > 0:
                        output = output[: int(eos_idx[0])]
                    
                    cif_txt = tokenizer.decode(output, skip_special_tokens=True).replace("\n\n", "\n")
                    
                    # Apply remove_conditionality for Raw conditioning
                    if activate_conditionality == "Raw":
                        cif_txt = remove_conditionality(cif_txt)
                    
                    mid = determine_material_id(row, len(generated), global_offset)
                    
                    generated.append({
                        "Material ID": mid,
                        "Prompt": row["Prompt"],
                        "Generated CIF": cif_txt,
                        "condition_vector": row.get("condition_vector", "None") if "condition_vector" in row else "None",
                    })
                    queue.put(1)
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return generated

def create_output_dataframe(generated_data, args, df_prompts):
    """Create and format the output DataFrame."""
    df = pd.DataFrame(generated_data)
    if args.input_parquet and 'True CIF' in df_prompts.columns:
        # Ensure 'Material ID' is of a compatible type for merging
        df_prompts['Material ID'] = df_prompts['Material ID'].astype(str)
        df['Material ID'] = df['Material ID'].astype(str)
        df = df.merge(df_prompts[['Material ID', 'True CIF']], on='Material ID', how='left')
    return df

def main():
    args = parse_args()
    
    # Minimum required arguments
    required_args = ['model_ckpt_dir', 'input_parquet', 'output_parquet']
    for req_arg in required_args:
        if not getattr(args, req_arg):
            sys.exit(f"ERROR: {req_arg} argument must be set.")
    
    # Set defaults
    args.num_repeats = getattr(args, 'num_repeats', 1)

    print("Environment info")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("\nGeneration settings")
    print(f"Total sequences per prompt-condition pair: {args.num_return_sequences * args.num_repeats}")
    print(f"Will save generated CIFs to {args.output_parquet}")
    

    # Find checkpoint if needed
    if 'checkpoint' not in args.model_ckpt_dir:
        args.model_ckpt_dir = find_checkpoint_from_dir(args.model_ckpt_dir)
    
    # Display model checkpoint info
    losses_file = os.path.join(args.model_ckpt_dir, "losses.json")
    if os.path.exists(losses_file):
        with open(losses_file, "r") as f:
            losses = json.load(f)
        print("\nModel checkpoint info")
        print(f"Most Recent Train Loss: {losses['training_losses'][-1]:.4f}")
        print(f"Most Recent Validation Loss: {losses['validation_losses'][-1]:.4f}")
    

    # Get model's max length from config
    n_positions = get_model_max_length(args.model_ckpt_dir, args.activate_conditionality)
    print(f"Model's max_length: {n_positions}")
    if n_positions < args.gen_max_length:
        print(f"WARNING: The model's max_length is {n_positions}, adjusting generation max_length")
    

    # Initialize tokenizer and build generation kwargs
    tokenizer = init_tokenizer(DEFAULT_TOKENIZER_DIR)
    generation_kwargs = build_generation_kwargs(args, tokenizer, n_positions)
    print(f"Generation kwargs: {generation_kwargs}")
    

    # Load and prepare data
    df_prompts = pd.read_parquet(args.input_parquet)
    if args.max_samples:
        df_prompts = df_prompts.sample(n=int(args.max_samples), random_state=1)
    
    # Optimize for single prompt case: duplicate prompt and halve repeats for better GPU utilization
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    single_prompt_optimization = False
    if len(df_prompts) == 1 and num_gpus > 1 and args.num_repeats > 1:
        single_prompt_optimization = True
        original_repeats = args.num_repeats
        args.num_repeats = args.num_repeats // num_gpus
        # Duplicate the single prompt for each GPU
        df_prompts = pd.concat([df_prompts] * num_gpus, ignore_index=True)
    
    total_samples = len(df_prompts)
    print(f"\nGeneration")
    print(f"Number of condition-prompt pairs: {total_samples}")
    

    # Setup multiprocessing
    manager = mp.Manager()
    queue = manager.Queue()
    total_generations = total_samples * args.num_return_sequences * args.num_repeats
    samples_per_gpu = total_samples // num_gpus
    
    results = []
    # Use spawn context for CUDA safety
    ctx = mp.get_context('spawn')
    with ctx.Pool(num_gpus + 1,
                 initializer=init_worker,
                 initargs=(args.model_ckpt_dir, DEFAULT_TOKENIZER_DIR, args.activate_conditionality)) as pool:
        
        pool.apply_async(progress_listener, (queue, total_generations))
        
        try:
            # Handle case with 0 GPUs
            if num_gpus == 0:
                results.append(pool.apply_async(
                    generate_on_gpu,
                    (0, df_prompts, generation_kwargs, queue, 0, total_samples,
                    args.model_ckpt_dir, args.activate_conditionality, args.num_repeats, 0)
                ))
            else:
                for gpu_id in range(num_gpus):
                    start = gpu_id * samples_per_gpu
                    end = (gpu_id + 1) * samples_per_gpu if gpu_id != num_gpus - 1 else total_samples
                    # Calculate global offset for material ID generation
                    if single_prompt_optimization:
                        # For single prompt case, each GPU should have unique material ID ranges
                        global_offset = gpu_id * args.num_return_sequences * args.num_repeats
                    else:
                        global_offset = 0
                    results.append(pool.apply_async(
                        generate_on_gpu,
                        (gpu_id, df_prompts, generation_kwargs, queue, start, end,
                        args.model_ckpt_dir, args.activate_conditionality, args.num_repeats, global_offset)
                    ))
        except Exception as e:
            print(f"(Did you set the correct activate conditionality??. Generation error: {e}")
            pool.terminate()
            manager.shutdown()
            sys.exit(1)
        
        # Collect results
        generated_data = []
        for res in results:
            generated_data.extend(res.get())
        queue.put("kill")
    
    # Create and save output
    df = create_output_dataframe(generated_data, args, df_prompts)

    # if there is a directory in the output path, create it
    output_dir = os.path.dirname(args.output_parquet)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df.to_parquet(args.output_parquet, index=False)
    print(f"\nSaved {len(df)} generated CIFs to {args.output_parquet}")
    

    # Cleanup
    manager.shutdown()
    gc.collect()

if __name__ == "__main__":
    # Set start method to 'spawn' for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    main()
