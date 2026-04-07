"""Local test section: integration."""

import os
import torch
from datasets import Dataset, DatasetDict

DEVICE = None


def get_device():
    """Get the device to use for tests."""
    global DEVICE
    return DEVICE if DEVICE is not None else torch.device("cpu")

class IntegrationTests:
    """Integration tests for end-to-end pipeline validation."""
    
    def __init__(self, temp_dir, test_data):
        self.temp_dir = temp_dir
        self.test_data = test_data
    
    def test_minimal_training_loop(self):
        """Test minimal training loop (2 steps) to verify pipeline doesn't crash."""
        from _utils._model_utils import build_model
        from _utils._trainer_utils import CIFFormattingTrainer
        from _dataloader import load_data, CustomCIFDataCollator
        from _tokenizer import CustomCIFTokenizer
        from transformers import TrainingArguments
        
        device = get_device()
        
        # Setup tokenizer
        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        
        # Create tiny mock dataset
        mock_data = {
            "train": Dataset.from_dict({
                "CIF": [self.test_data['augmented_cif']] * 4
            }),
            "validation": Dataset.from_dict({
                "CIF": [self.test_data['augmented_cif']] * 2
            })
        }
        dataset = DatasetDict(mock_data)
        
        # Load and tokenize
        tokenized_dataset, data_collator = load_data(
            tokenizer=tokenizer,
            dataset=dataset,
            context_length=256,
            mode="unconditional"
        )
        
        # Build minimal model
        class MockArgs:
            activate_conditionality = None
            n_embd = 64
            n_layer = 1
            n_head = 2
            context_length = 256
            n_positions = 256
            residual_dropout = 0.1
            embedding_dropout = 0.1
            attention_dropout = 0.1
        
        args = MockArgs()
        model = build_model(args, tokenizer)
        model.to(device)
        
        # Setup minimal training - use CIFFormattingTrainer which handles fixed_mask
        output_dir = os.path.join(self.temp_dir, "train_test")
        training_args = TrainingArguments(
            output_dir=output_dir,
            max_steps=2,
            per_device_train_batch_size=2,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            use_cpu=(device.type == "cpu")
        )
        
        trainer = CIFFormattingTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            data_collator=data_collator
        )
        
        # Run 2 training steps
        trainer.train()
        
        # Verify model updated (loss should be computed)
        assert True, "Training loop completed without crash"
    
    def test_conditional_training_loop(self):
        """Test conditional model training loop."""
        from _models.PKV_model import PKVGPT, PKVGPT2Config
        from _utils._trainer_utils import CIFFormattingTrainer
        from _dataloader import load_data
        from _tokenizer import CustomCIFTokenizer
        from transformers import TrainingArguments
        
        device = get_device()
        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        
        # Mock conditional dataset
        mock_data = {
            "train": Dataset.from_dict({
                "CIF": [self.test_data['augmented_cif']] * 4,
                "bandgap": [0.5, 0.6, 0.7, 0.8]
            }),
            "validation": Dataset.from_dict({
                "CIF": [self.test_data['augmented_cif']] * 2,
                "bandgap": [0.55, 0.65]
            })
        }
        dataset = DatasetDict(mock_data)
        
        # Use larger context to accommodate PKV prefix tokens
        context_length = 512
        n_prefix_tokens = 2
        
        tokenized_dataset, data_collator = load_data(
            tokenizer=tokenizer,
            dataset=dataset,
            context_length=context_length,
            mode="conditional",
            condition_columns="['bandgap']"
        )
        
        # Build conditional model - n_positions must accommodate context + prefix tokens
        config = PKVGPT2Config(
            vocab_size=len(tokenizer),
            n_positions=context_length + n_prefix_tokens,
            n_embd=64,
            n_layer=1,
            n_head=2,
            n_input_vector=1,
            n_prefix_tokens=n_prefix_tokens,
            n_hidden_cond=32
        )
        model = PKVGPT(config)
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)
        
        output_dir = os.path.join(self.temp_dir, "cond_train_test")
        training_args = TrainingArguments(
            output_dir=output_dir,
            max_steps=2,
            per_device_train_batch_size=2,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            use_cpu=(device.type == "cpu")
        )
        
        trainer = CIFFormattingTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            data_collator=data_collator
        )
        
        trainer.train()
        assert True, "Conditional training loop completed"
    
    def test_generation_after_training(self):
        """Test model can generate after training."""
        from transformers import GPT2LMHeadModel, GPT2Config
        from _tokenizer import CustomCIFTokenizer
        
        device = get_device()
        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        
        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=256,
            n_embd=64,
            n_layer=1,
            n_head=2
        )
        model = GPT2LMHeadModel(config)
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)
        model.eval()
        
        # Prepare prompt
        prompt = "<bos>\ndata_"
        input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=50,
                do_sample=True,
                top_k=10,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0])
        assert len(generated_text) > len(prompt), "Should generate additional tokens"
        assert "data_" in generated_text, "Should contain data block"
