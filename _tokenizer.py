"""
Hugging Face compatible custom tokenizer for CIF data.
"""

import os
import re
import json
from transformers import PreTrainedTokenizer

class CustomCIFTokenizer(PreTrainedTokenizer):
    """
    Hugging Face-compatible custom tokenizer for CIF data.
    """
    def __init__(
        self,
        vocab_file,
        spacegroups_file,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        var_open_token="[",
        var_close_token="]",
        prop_token="<prop>",
        **kwargs
    ):
        with open(vocab_file, "r") as f:
            self.token_to_id = json.load(f)

        # Invert to get ID -> token
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        # Keep track of all known tokens
        self._tokens = list(self.token_to_id.keys())

        # For convenience, load the raw spacegroups
        with open(spacegroups_file, "r") as f:
            self.space_groups = [sg.strip() for sg in f.readlines()]

        # Sort tokens by length for regex matching
        self._escaped_tokens = sorted(
            [re.escape(t) for t in self._tokens],
            key=len,
            reverse=True
        )
        
        # Escaped tokens refer to tokens that have been processed 
        # to ensure that any special characters they contain are treated as literal characters
        # escaping means prefixing the special character with a backslash (like \\n)
        # you sort the tokens by length so that the longest token is matched first
        # this is important because if you have a token "a" and "ab", you want to match "ab" first

        # Define main special tokens
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.var_open_token = var_open_token
        self.var_close_token = var_close_token
        self.prop_token = prop_token

        # Make sure they're all in the vocabulary
        for special_tk in [self.unk_token, 
                           self.pad_token, 
                           self.bos_token, 
                           self.eos_token, 
                           self.var_open_token,
                           self.var_close_token,
                           self.prop_token]:
            if special_tk not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[special_tk] = new_id
                self.id_to_token[new_id] = special_tk
                self._tokens.append(special_tk)

        # Validate id_to_token mapping
        self.validate_id_to_token()

        super().__init__(
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            var_open_token=self.var_open_token,
            var_close_token=self.var_close_token,
            prop_token=self.prop_token,
            **kwargs
        )

    @property
    def vocab_size(self):
        """
        Return the size of the base vocabulary (without added special tokens).
        """
        return len(self.token_to_id)

    def _tokenize(self, text):
        """
        Custom tokenization logic for CIF data.
        """
        # Disambiguate space groups in the text:
        spacegroups_pattern = "|".join(self.space_groups)
        
        # Disambiguate bracketed space groups:
        text = re.sub(
            fr'(_symmetry_space_group_name_H-M\s*\[)({spacegroups_pattern})(\])',
            r'\1\2_sg\3',
            text
        )


        # here just adding "_sg" to the space group name inside the sample

        # Build the tokenization pattern:
        token_pattern = "|".join(self._escaped_tokens)
        # all tokens from sample are escaped and joined by "|"
        # allows to match any of predefined tokens

        # Define the full pattern to match tokens, words, and punctuation
        full_pattern = f"({token_pattern}|\\w+|[\\.,;!?\\[\\]])"
        # can match any of the predefined tokens, words, or punctuation

        # Clean up spaces
        text = re.sub(r"[ \t]+", " ", text)
        # replace multiple spaces or tabs with a single space, gives uniform

        # Extract tokens
        tokens = re.findall(full_pattern, text)
        # find all matches of the pattern in the text

        # Replace unknown tokens
        output_tokens = [
            t if t in self._tokens else self.unk_token
            for t in tokens
        ]
        # if the token is in the predefined tokens, keep it, otherwise replace with unk_token
        return output_tokens

    def _convert_token_to_id(self, token):
        """
        Convert a token to its corresponding ID.
        """
        return self.token_to_id.get(token, self.token_to_id.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """
        Convert an ID to its corresponding token.
        """
        #print the id_to_token dictionary
        if hasattr(index, "item"):
            index = int(index.item())  # Convert tensor to int
        if index not in self.id_to_token:
            print(f"Warning: Token ID {index} not found in id_to_token. Returning <unk>.")
        return self.id_to_token.get(index, self.unk_token)
    
    def validate_id_to_token(self):
        """
        Validate that id_to_token covers all token IDs up to vocab_size.
        """
        missing_ids = [i for i in range(len(self.token_to_id)) if i not in self.id_to_token]
        if missing_ids:
            print(f"Warning: Missing IDs in id_to_token: {missing_ids}")
        else:
            print("Tokenizer validation passed: token vocabulary is consistent.")

    def convert_tokens_to_string(self, tokens):
        """
        Convert a list of tokens to a string.
        """
        return "".join(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Add special tokens to a sequence or a pair of sequences.
        For GPT-2-style models, no additional special tokens are used.
        either returns token_ids_0 or token_ids_0 + token_ids_1
        """
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def get_vocab(self):
        """
        Return the tokenizer's vocabulary (including base vocabulary).
        """
        return dict(self.token_to_id)

    def get_added_vocab(self):
        """
        Return the additional vocabulary tokens added after the initial vocab.
        """
        return {
            tok: idx
            for tok, idx in self._added_tokens_encoder.items()
        }

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        """
        Decode a sequence of token IDs into a string.
        """
        tokens = [self._convert_id_to_token(idx) for idx in token_ids]
        if skip_special_tokens:
            tokens = [tok for tok in tokens if tok not in [self.bos_token, 
                                                           self.eos_token, 
                                                           self.pad_token, 
                                                           self.var_open_token, 
                                                           self.var_close_token,
                                                           self.prop_token]]
        # Remove '_sg' suffix from space group tokens
        tokens = [re.sub(r'_sg$', '', tok) for tok in tokens]
        
        result = self.convert_tokens_to_string(tokens)
        return result

    @classmethod
    def from_pretrained(cls, pretrained_dir, **kwargs):
        """
        Load a tokenizer from a pretrained directory.
        """
        vocab_file = os.path.join(pretrained_dir, "vocabulary.json")
        spacegroups_file = os.path.join(pretrained_dir, "spacegroups.txt")
        tokenizer_config_file = os.path.join(pretrained_dir, "tokenizer_config.json")

        with open(tokenizer_config_file, "r") as f:
            config = json.load(f)

        return cls(
            vocab_file=vocab_file,
            spacegroups_file=spacegroups_file,
            unk_token=config.get("unk_token", "<unk>"),
            **kwargs
        )

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the base vocabulary (token->ID) to a file.
        Hugging Face's `save_pretrained` will call this.
        Returns the path(s) of the saved vocab file(s).
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        vocab_file_name = (filename_prefix + "-" if filename_prefix else "") + "vocabulary.json"
        vocab_file = os.path.join(save_directory, vocab_file_name)
        with open(vocab_file, "w") as f:
            json.dump(self.token_to_id, f)

        return (vocab_file,)

    def save_pretrained(self, save_directory, **kwargs):
        """
        Save the tokenizer configuration, vocabulary, and other data (at every save).
        """
        super().save_pretrained(save_directory, **kwargs)

        # Also save spacegroups
        spacegroups_file = os.path.join(save_directory, "spacegroups.txt")
        with open(spacegroups_file, "w") as f:
            f.write("\n".join(self.space_groups))

        # Save tokenizer configuration
        tokenizer_config = {
            "unk_token": self.unk_token,
            "vocab_size": self.vocab_size,
            "tokenizer_class": self.__class__.__name__,
        }
        tokenizer_config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(tokenizer_config_file, "w") as f:
            json.dump(tokenizer_config, f)

    def add_custom_tokens(self, tokens):
        """
        Add a list of tokens to the tokenizer vocabulary.
        Updates internal mappings and escaped tokens for regex.
        """
        for token in tokens:
            if token not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[token] = new_id
                self.id_to_token[new_id] = token
                self._tokens.append(token)  # Add to the list of tokens
                print(f"Added token '{token}' with ID {new_id}")

        # Update escaped tokens for regex matching
        self._escaped_tokens = sorted(
            [re.escape(t) for t in self._tokens],
            key=len,
            reverse=True
        )

    def remove_custom_tokens(self, tokens):
        """
        Remove a list of tokens from the tokenizer vocabulary (if present).
        Updates internal mappings and escaped tokens for regex.
        Note: Removing tokens changes ID assignments for the rest of the vocab.
        Use with caution for a trained model.
        """
        for token in tokens:
            if token in self.token_to_id:
                old_id = self.token_to_id.pop(token)
                if old_id in self.id_to_token:
                    del self.id_to_token[old_id]
        # Rebuild tokens and IDs from scratch to maintain consistency
        # (IDs shift, so any model trained with old IDs is no longer compatible)
        sorted_pairs = sorted(self.token_to_id.items(), key=lambda x: x[1])
        self.token_to_id = {}
        self.id_to_token = {}
        for i, (token, _) in enumerate(sorted_pairs):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        self._tokens = list(self.token_to_id.keys())
        self._escaped_tokens = sorted(
            [re.escape(t) for t in self._tokens],
            key=len,
            reverse=True
        )

    def validate_tokenizer_state(self):
        """
        Validate that the tokenizer's internal state is consistent.
        """
        # Check token_to_id and id_to_token mappings
        for token, idx in self.token_to_id.items():
            if self.id_to_token.get(idx) != token:
                print(f"Warning: Inconsistent mapping for token '{token}' (ID {idx})")

        # Check _tokens list
        missing_tokens = [token for token in self.token_to_id.keys() if token not in self._tokens]
        if missing_tokens:
            print(f"Warning: Missing tokens in _tokens: {missing_tokens}")

        # Check _escaped_tokens list
        missing_escaped = [re.escape(token) for token in self.token_to_id.keys() if re.escape(token) not in self._escaped_tokens]
        if missing_escaped:
            print(f"Warning: Missing escaped tokens in _escaped_tokens: {missing_escaped}")

        print("Tokenizer state validation complete.") 