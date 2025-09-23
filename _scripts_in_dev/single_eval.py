import os
import json
import torch
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _tokenizer import CustomCIFTokenizer
from _args import parse_args
from _models import PKVGPT, PrefixGPT, SliderGPT
from _utils import find_checkpoint_from_dir_gen
from transformers import StoppingCriteria

class StopOnEOSToken(StoppingCriteria):
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, scores, **kwargs):
        # Stop generation if any sequence has generated <eos>
        return any(self.eos_token_id in seq for seq in input_ids.tolist())


def main():
    # Parse arguments
    args = parse_args()

    # Load Custom Tokenizer
    tokenizer = CustomCIFTokenizer.from_pretrained(
        pretrained_dir=args.pretrained_tokenizer_dir,
        pad_token="<pad>"
    )
    # If no pad token, make it the eos token
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load Model
    # to handle if user provides a directory instead of a checkpoint file
    if 'checkpoint' not in args.model_ckpt_dir:
        args = find_checkpoint_from_dir_gen(args)
    
    if args.activate_conditionality == "PKV":
        print("Loading PKVGPT model...")
        model = PKVGPT.from_pretrained(args.model_ckpt_dir)
    elif args.activate_conditionality == "Prepend":
        print("Loading PrefixGPT model...")
        model = PrefixGPT.from_pretrained(args.model_ckpt_dir)
    elif args.activate_conditionality == "Slider":
        print("Loading SliderGPT model...")
        model = SliderGPT.from_pretrained(args.model_ckpt_dir)
    else:
        raise ValueError(f"Unknown model type: {args.activate_conditionality}")

    
    model.eval()
    model.to(device)
    # Resize token embeddings in case the tokenizer has a different size
    model.resize_token_embeddings(len(tokenizer))

    # Print the model's configuration
    print("-" * 80)
    print("Model Configuration:")
    print(model.config)
    print("-" * 80)

    # Double-check the model’s max context size (often 1024 for GPT-2)
    max_positions = model.config.n_positions
    print(f"Max Context Size: {max_positions}, so generation will be limited to this length.")

    # Load the losses.json file
    losses_file = os.path.join(args.pretrained_tokenizer_dir, "losses.json")
    if os.path.exists(losses_file):
        with open(losses_file, "r") as f:
            losses = json.load(f)
        # Print the most recent train and validation loss, rounded to 4 decimal places
        print(f"Most Recent Train Loss: {losses['training_losses'][-1]:.4f}")
        print(f"Most Recent Validation Loss: {losses['validation_losses'][-1]:.4f}")

    # Turn prompt into token IDs
    print("Generating text using the custom generation method...\n")
    input_ids = tokenizer.encode(args.prompt, return_tensors='pt')
    input_ids = input_ids.to(device)

    # Convert the bandgap value (passed as args.vector) into a tensor.
    # Ensure it has shape [1, 1] as expected by the model (batch size 1, n_input_vector = 1).
    # vector is a string of the form "0.000, 1.000", its input as a string, but pass this as a vector so it can handle 1 or more values
    
    # Convert condition vector string into tensor
    if args.condition_vector is not None and args.condition_vector != "None":
        try:
            # Split by comma if multiple values, otherwise use single value
            if ',' in str(args.condition_vector):
                values = [float(x.strip()) for x in str(args.condition_vector).split(',')]
            else:
                values = [float(args.condition_vector)]
            condition_tensor = torch.tensor([values], device=device)
            print(f"Condition tensor shape: {condition_tensor.shape}, values: {values}")
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not convert condition vector {args.condition_vector} to float: {e}")
            condition_tensor = None
    else:
        condition_tensor = None

    # Set Generation parameters
    generation_kwargs = {
        "max_length": min(args.gen_max_length, max_positions),
        "num_return_sequences": args.num_return_sequences,
        "do_sample": args.do_sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "renormalize_logits": True,
        "temperature": args.temperature,
        "suppress_tokens": [tokenizer.pad_token_id],
        "use_cache": True
    }

    # Generate the output tokens while passing condition_values
    with torch.no_grad():
        outputs_custom = model.generate(
            input_ids=input_ids,
            condition_values=condition_tensor,
            **generation_kwargs
        )

    # Decode the output tokens into text
    decoded_outputs_custom = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs_custom]

    # Print the decoded outputs
    for i, output in enumerate(decoded_outputs_custom):
        print(f"Generated Text {i+1} (Custom Generation):")
        print(output)
        print("-" * 80)


if __name__ == "__main__":
    main()
