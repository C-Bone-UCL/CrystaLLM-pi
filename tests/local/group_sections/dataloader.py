"""Local test section: dataloader."""

from datasets import Dataset, DatasetDict

class DataLoaderTests:
    """Test data loading and collation components."""
    
    def __init__(self, temp_dir, test_data):
        self.temp_dir = temp_dir
        self.test_data = test_data
    
    def test_data_collator_unconditional(self):
        """Test CustomCIFDataCollator in unconditional mode."""
        from _dataloader import CustomCIFDataCollator
        from _tokenizer import CustomCIFTokenizer
        
        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        collator = CustomCIFDataCollator(tokenizer, context_length=256)
        
        # Create mock features (unconditional - no condition_values)
        cif_text = self.test_data['augmented_cif']
        encoded = tokenizer.encode(cif_text)
        
        features = [
            {
                "input_ids": encoded[:100],
                "fixed_mask": [1] * 100,
                "attention_mask": [1] * 100,
                "special_tokens_mask": [0] * 100
            },
            {
                "input_ids": encoded[:80],
                "fixed_mask": [1] * 80,
                "attention_mask": [1] * 80,
                "special_tokens_mask": [0] * 80
            }
        ]
        
        batch = collator(features)
        
        assert "input_ids" in batch, "Batch should have input_ids"
        assert "labels" in batch, "Batch should have labels"
        assert "attention_mask" in batch, "Batch should have attention_mask"
        assert batch["input_ids"].shape[1] == 256, "Should pack to context_length"
        assert "condition_values" not in batch, "Unconditional should not have condition_values"
    
    def test_data_collator_conditional(self):
        """Test CustomCIFDataCollator in conditional mode."""
        from _dataloader import CustomCIFDataCollator
        from _tokenizer import CustomCIFTokenizer
        
        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        collator = CustomCIFDataCollator(tokenizer, context_length=256)
        
        cif_text = self.test_data['augmented_cif']
        encoded = tokenizer.encode(cif_text)
        
        # Create mock features with condition_values
        features = [
            {
                "input_ids": encoded[:100],
                "fixed_mask": [1] * 100,
                "attention_mask": [1] * 100,
                "special_tokens_mask": [0] * 100,
                "condition_values": [0.5, 0.3]
            },
            {
                "input_ids": encoded[:80],
                "fixed_mask": [1] * 80,
                "attention_mask": [1] * 80,
                "special_tokens_mask": [0] * 80,
                "condition_values": [0.7, 0.2]
            }
        ]
        
        batch = collator(features)
        
        assert "condition_values" in batch, "Conditional batch should have condition_values"
        assert batch["condition_values"].shape == (2, 2), "Condition values shape should be (batch, n_conditions)"
        assert batch["input_ids"].shape[1] == 256, "Should pack to context_length"
    
    def test_data_collator_round_robin(self):
        """Test round-robin packing when CIFs are short."""
        from _dataloader import CustomCIFDataCollator
        from _tokenizer import CustomCIFTokenizer
        
        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        collator = CustomCIFDataCollator(tokenizer, context_length=256)
        
        # Create short sequences that need packing
        features = [
            {
                "input_ids": [1, 2, 3] * 20,  # 60 tokens
                "fixed_mask": [1] * 60,
                "attention_mask": [1] * 60,
                "special_tokens_mask": [0] * 60
            },
            {
                "input_ids": [4, 5, 6] * 25,  # 75 tokens
                "fixed_mask": [1] * 75,
                "attention_mask": [1] * 75,
                "special_tokens_mask": [0] * 75
            }
        ]
        
        batch = collator(features)
        
        # Each sequence should be packed to exactly context_length
        assert batch["input_ids"].shape == (2, 256), "Should pack to (batch, context_length)"
        assert batch["fixed_mask"].shape == (2, 256), "Fixed mask should match"
    
    def test_data_collator_long_cif_slicing(self):
        """Test that CIFs longer than context_length get sliced."""
        from _dataloader import CustomCIFDataCollator
        from _tokenizer import CustomCIFTokenizer
        
        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        context_length = 128
        collator = CustomCIFDataCollator(tokenizer, context_length=context_length)
        
        # Create a sequence longer than context_length
        long_seq = list(range(1, 300))  # 299 tokens
        features = [
            {
                "input_ids": long_seq,
                "fixed_mask": [1] * len(long_seq),
                "attention_mask": [1] * len(long_seq),
                "special_tokens_mask": [0] * len(long_seq)
            }
        ]
        
        batch = collator(features)
        
        assert batch["input_ids"].shape[1] == context_length, "Should slice to context_length"
        # Should keep beginning of sequence
        assert batch["input_ids"][0, 0].item() == 1, "Should keep start of sequence"
    
    def test_load_data_unconditional(self):
        """Test load_data function in unconditional mode."""
        from _dataloader import load_data
        from _tokenizer import CustomCIFTokenizer
        
        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        
        # Create a minimal mock dataset
        mock_data = {
            "train": Dataset.from_dict({
                "CIF": [self.test_data['augmented_cif']] * 5
            }),
            "validation": Dataset.from_dict({
                "CIF": [self.test_data['augmented_cif']] * 2
            })
        }
        dataset = DatasetDict(mock_data)
        
        tokenized_dataset, data_collator = load_data(
            tokenizer=tokenizer,
            dataset=dataset,
            context_length=512,
            mode="unconditional",
            remove_CIFs_above_context=False,
            remove_CIFs_with_unk=False
        )
        
        assert "train" in tokenized_dataset, "Should have train split"
        assert "input_ids_variants" in tokenized_dataset["train"].features, "Should have input_ids_variants"
        assert data_collator is not None, "Should return data collator"
    
    def test_load_data_conditional(self):
        """Test load_data function in conditional mode."""
        from _dataloader import load_data
        from _tokenizer import CustomCIFTokenizer
        
        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        
        # Create mock dataset with condition columns
        mock_data = {
            "train": Dataset.from_dict({
                "CIF": [self.test_data['augmented_cif']] * 5,
                "bandgap": [0.5, 0.6, 0.7, 0.8, 0.9],
                "density": [0.1, 0.2, 0.3, 0.4, 0.5]
            }),
            "validation": Dataset.from_dict({
                "CIF": [self.test_data['augmented_cif']] * 2,
                "bandgap": [0.55, 0.65],
                "density": [0.15, 0.25]
            })
        }
        dataset = DatasetDict(mock_data)
        
        tokenized_dataset, data_collator = load_data(
            tokenizer=tokenizer,
            dataset=dataset,
            context_length=512,
            mode="conditional",
            condition_columns="['bandgap', 'density']",
            remove_CIFs_above_context=False,
            remove_CIFs_with_unk=False
        )
        
        assert "condition_values" in tokenized_dataset["train"].features, "Should have condition_values"
        # Check first example has correct number of conditions
        first_example = tokenized_dataset["train"][0]
        assert len(first_example["condition_values"]) == 2, "Should have 2 condition values"

    def test_load_data_train_uses_all_cif_variants(self):
        """Train split should pre-tokenize all CIF variants and collator should sample one per row."""
        from _dataloader import load_data
        from _tokenizer import CustomCIFTokenizer

        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")

        base_cif = self.test_data['augmented_cif']
        mock_data = {
            "train": Dataset.from_dict({
                "CIF": [base_cif] * 4,
                "CIF_SUPERCELL_1": [base_cif] * 4,
                "CIF_SUPERCELL_2": [base_cif] * 4,
            }),
            "validation": Dataset.from_dict({
                "CIF": [base_cif] * 2,
                "CIF_SUPERCELL_1": [""] * 2,
                "CIF_SUPERCELL_2": [""] * 2,
            }),
        }
        dataset = DatasetDict(mock_data)

        tokenized_dataset, data_collator = load_data(
            tokenizer=tokenizer,
            dataset=dataset,
            context_length=512,
            mode="unconditional",
            remove_CIFs_above_context=False,
            remove_CIFs_with_unk=False,
        )

        train_example = tokenized_dataset["train"][0]
        assert "input_ids_variants" in train_example
        assert len(train_example["input_ids_variants"]) == 3

        batch = data_collator([tokenized_dataset["train"][0], tokenized_dataset["train"][1]])
        assert batch["input_ids"].shape[0] == 2
        assert batch["input_ids"].shape[1] == 512

    def test_data_collator_samples_unique_variant_content(self):
        """Variant sampling should deduplicate equivalent variants before random choice."""
        from _dataloader import CustomCIFDataCollator
        from _tokenizer import CustomCIFTokenizer

        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        collator = CustomCIFDataCollator(tokenizer, context_length=4, data_seed=7)

        feature = {
            "input_ids_variants": [
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [9, 9, 9, 9],
            ],
            "fixed_mask_variants": [[1, 1, 1, 1]] * 4,
            "attention_mask_variants": [[1, 1, 1, 1]] * 4,
            "special_tokens_mask_variants": [[0, 0, 0, 0]] * 4,
        }

        unique_hits = 0
        draws = 200
        for _ in range(draws):
            batch = collator([feature])
            if int(batch["input_ids"][0, 0].item()) == 9:
                unique_hits += 1

        unique_rate = unique_hits / float(draws)
        assert 0.35 <= unique_rate <= 0.65
