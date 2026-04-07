"""Local test section: data utils."""

from datasets import Dataset, DatasetDict

class DataUtilsTests:
    """Test data utility functions from _utils/_data_utils.py."""
    
    def __init__(self, temp_dir, test_data):
        self.temp_dir = temp_dir
        self.test_data = test_data
    
    def test_filter_long_cifs(self):
        """Test filtering CIFs that exceed context length."""
        from _utils._data_utils import filter_long_CIFs
        from _tokenizer import CustomCIFTokenizer
        
        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        
        # Create mock dataset with varying lengths
        short_ids = list(range(50))
        long_ids = list(range(200))
        
        mock_data = {
            "train": Dataset.from_dict({
                "input_ids": [short_ids, long_ids, short_ids]
            })
        }
        dataset = DatasetDict(mock_data)
        
        # Filter with context_length=100
        filtered = filter_long_CIFs(dataset, context_length=100)
        
        # Should keep only entries with length <= 100
        assert len(filtered["train"]) == 2, "Should filter out long entries"
    
    def test_filter_cifs_with_unk(self):
        """Test filtering CIFs with unknown tokens."""
        from _utils._data_utils import filter_CIFs_with_unk
        from _tokenizer import CustomCIFTokenizer
        
        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        unk_id = tokenizer.unk_token_id
        
        # Create mock dataset: one with unk, one without
        mock_data = {
            "train": Dataset.from_dict({
                "input_ids": [
                    [1, 2, 3, 4, 5],  # No unk
                    [1, unk_id, 3, 4, 5],  # Has unk
                    [1, 2, 3, unk_id, 5],  # Has unk
                ]
            })
        }
        dataset = DatasetDict(mock_data)
        
        filtered = filter_CIFs_with_unk(dataset, tokenizer)
        
        assert len(filtered["train"]) == 1, "Should filter entries with unk tokens"
    
    def test_tokenize_function_unconditional(self):
        """Test tokenize_function in unconditional mode."""
        from _utils._data_utils import tokenize_function
        from _tokenizer import CustomCIFTokenizer
        
        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        
        examples = {
            "CIF": [self.test_data['augmented_cif'], self.test_data['test_cif']]
        }
        
        result = tokenize_function(examples, tokenizer, mode="unconditional")
        
        assert "input_ids" in result, "Should have input_ids"
        assert "fixed_mask" in result, "Should have fixed_mask"
        assert len(result["input_ids"]) == 2, "Should have 2 tokenized samples"
    
    def test_tokenize_function_conditional(self):
        """Test tokenize_function in conditional mode."""
        from _utils._data_utils import tokenize_function
        from _tokenizer import CustomCIFTokenizer
        
        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        
        examples = {
            "CIF": [self.test_data['augmented_cif'], self.test_data['test_cif']],
            "bandgap": [0.5, 0.7],
            "density": [0.3, 0.4]
        }
        
        result = tokenize_function(
            examples, 
            tokenizer, 
            condition_columns="['bandgap', 'density']",
            mode="conditional"
        )
        
        assert "input_ids" in result, "Should have input_ids"
        assert "condition_values" in result, "Should have condition_values"
        assert len(result["condition_values"]) == 2, "Should have 2 condition value sets"
        assert len(result["condition_values"][0]) == 2, "Each should have 2 conditions"
    
    def test_tokenize_function_raw(self):
        """Test tokenize_function in raw mode (text conditioning)."""
        from _utils._data_utils import tokenize_function
        from _tokenizer import CustomCIFTokenizer
        
        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        
        examples = {
            "CIF": [self.test_data['augmented_cif']],
            "bandgap": [0.5]
        }
        
        result = tokenize_function(
            examples, 
            tokenizer, 
            condition_columns="['bandgap']",
            mode="raw"
        )
        
        assert "input_ids" in result, "Should have input_ids"
        # Raw mode embeds conditions as text, so no separate condition_values
        assert "condition_values" not in result, "Raw mode should not have condition_values"
    
    def test_create_fixed_format_mask(self):
        """Test fixed format mask creation for variable tokens."""
        from _utils._data_utils import create_fixed_format_mask
        from _tokenizer import CustomCIFTokenizer
        
        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        
        # Text with variable regions in brackets
        text = "data_test\n_cell_length_a [5.0]\n_cell_length_b [4.0]"
        
        mask = create_fixed_format_mask(text, tokenizer, full_length=100)
        
        assert len(mask) > 0, "Mask should not be empty"
        assert 0 in mask, "Should have variable tokens (0s)"
        assert 1 in mask, "Should have fixed tokens (1s)"
    
    def test_parse_condition_value(self):
        """Test condition value parsing helper."""
        from _utils._data_utils import _parse_condition_value
        
        # Test float
        assert _parse_condition_value(0.5) == [0.5]
        
        # Test int
        assert _parse_condition_value(1) == [1.0]
        
        # Test string number
        assert _parse_condition_value("0.7") == [0.7]
        
        # Test string list
        result = _parse_condition_value("[0.5, 0.3]")
        assert len(result) == 2
        assert abs(result[0] - 0.5) < 1e-6
        
        # Test list
        assert _parse_condition_value([0.1, 0.2]) == [0.1, 0.2]

    def test_get_cif_candidate_columns(self):
        """Train CIF column discovery should include every CIF-like column."""
        from _utils._data_utils import get_cif_candidate_columns

        columns = [
            "Material ID",
            "CIF",
            "CIF_SUPERCELL_1",
            "CIF_SUPERCELL_2",
            "token_count_by_cif_variant",
        ]

        result = get_cif_candidate_columns(columns)
        assert result == ["CIF", "CIF_SUPERCELL_1", "CIF_SUPERCELL_2"]

    def test_build_train_cif_variant_texts_with_fallback(self):
        """Train CIF variants should preserve empty augmented columns for downstream base fallback."""
        from _utils._data_utils import build_train_cif_variant_texts

        examples = {
            "CIF": ["base-0", "base-1", "base-2"],
            "CIF_SUPERCELL_1": ["sc1-0", "", "sc1-2"],
            "CIF_SUPERCELL_2": ["sc2-0", "sc2-1", ""],
        }
        cif_columns = ["CIF", "CIF_SUPERCELL_1", "CIF_SUPERCELL_2"]

        variants = build_train_cif_variant_texts(examples=examples, cif_columns=cif_columns)

        assert len(variants) == 3
        assert variants[0] == ["base-0", "sc1-0", "sc2-0"]
        assert variants[1] == ["base-1", "", "sc2-1"]
        assert variants[2] == ["base-2", "sc1-2", ""]

    def test_tokenize_function_train_variants(self):
        """Tokenization should support pre-tokenizing all CIF variants per train row."""
        from _utils._data_utils import tokenize_function
        from _tokenizer import CustomCIFTokenizer

        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        examples = {
            "CIF": [self.test_data['augmented_cif']],
            "bandgap": [0.5],
        }
        cif_variant_texts = [[
            self.test_data['augmented_cif'],
            self.test_data['augmented_cif'],
            self.test_data['augmented_cif'],
            self.test_data['augmented_cif'],
        ]]

        result = tokenize_function(
            examples=examples,
            tokenizer=tokenizer,
            condition_columns="['bandgap']",
            mode="conditional",
            cif_variant_texts=cif_variant_texts,
        )

        assert "input_ids_variants" in result
        assert "fixed_mask_variants" in result
        assert len(result["input_ids_variants"]) == 1
        assert len(result["input_ids_variants"][0]) == 4
        assert len(result["condition_values"]) == 1

    def test_tokenize_function_train_variants_blank_fallback(self):
        """Blank augmented variants should reuse base tokens instead of tokenizing empty text."""
        from _utils._data_utils import tokenize_function
        from _tokenizer import CustomCIFTokenizer

        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        base_cif = self.test_data['augmented_cif']

        result = tokenize_function(
            examples={"CIF": [base_cif]},
            tokenizer=tokenizer,
            mode="unconditional",
            cif_variant_texts=[[base_cif, "", "", ""]],
        )

        variants = result["input_ids_variants"][0]
        assert len(variants) == 4
        assert variants[1] == variants[0]
        assert variants[2] == variants[0]
        assert variants[3] == variants[0]

    def test_tokenize_function_train_variants_avoids_mask_retokenization(self):
        """Variant tokenization should not invoke tokenizer again while building fixed masks."""
        from _utils._data_utils import tokenize_function

        class CountingTokenizer:
            def __init__(self):
                self.bos_token = "<bos>"
                self.eos_token = "<eos>"
                self.call_count = 0
                self._next_id = 10
                self.token_to_id = {"<bos>": 1, "<eos>": 2, "[": 3, "]": 4}
                self.id_to_token = {v: k for k, v in self.token_to_id.items()}

            def _encode_text(self, text):
                ids = []
                for tok in str(text).replace("\n", " ").split():
                    if tok not in self.token_to_id:
                        self.token_to_id[tok] = self._next_id
                        self.id_to_token[self._next_id] = tok
                        self._next_id += 1
                    ids.append(self.token_to_id[tok])
                return ids

            def __call__(
                self,
                texts,
                truncation=False,
                return_special_tokens_mask=False,
                return_attention_mask=False,
            ):
                self.call_count += 1
                if isinstance(texts, str):
                    texts = [texts]

                all_ids = [self._encode_text(text) for text in texts]
                output = {"input_ids": all_ids}
                if return_attention_mask:
                    output["attention_mask"] = [[1] * len(ids) for ids in all_ids]
                if return_special_tokens_mask:
                    output["special_tokens_mask"] = [
                        [1 if tok_id in (1, 2) else 0 for tok_id in ids]
                        for ids in all_ids
                    ]
                return output

            def convert_ids_to_tokens(self, token_ids):
                return [self.id_to_token[tok_id] for tok_id in token_ids]

        tokenizer = CountingTokenizer()
        base_cif = "data_test _cell_length_a [ 5.0 ]"

        result = tokenize_function(
            examples={"CIF": [base_cif]},
            tokenizer=tokenizer,
            mode="unconditional",
            cif_variant_texts=[[base_cif, "", "", ""]],
        )

        assert "fixed_mask_variants" in result
        assert tokenizer.call_count == 1
