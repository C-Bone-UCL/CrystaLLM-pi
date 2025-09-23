import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch import nn

class PrependGPT2Config(GPT2Config):
    """
    Configuration class for PrependGPT. Inherits from GPT2Config.
    """
    # model_type = "prepend_gpt2"

    def __init__(self,
                 n_input_vector: int = 2,       # Size of the conditioning vector
                 n_prefix_tokens: int = 2,      # Number of prefix tokens to prepend
                 n_hidden_cond: int = 128,      # Hidden size for the conditioning encoder MLP
                 dropout: float = 0.1,          # Dropout for the conditioning encoder
                 **kwargs):
        # Handle add_cross_attention explicitly to ensure it's False
        kwargs.pop('add_cross_attention', None)
        super().__init__(**kwargs)
        self.add_cross_attention = False

        self.n_input_vector = n_input_vector
        self.n_prefix_tokens = n_prefix_tokens
        self.n_hidden_cond = n_hidden_cond
        self.dropout = dropout

class PrependEncoder(nn.Module):
    """
    Encodes the conditioning vector into prefix embeddings.
    """
    def __init__(self, config: PrependGPT2Config):
        super().__init__()
        self.config = config
        self.processor = nn.Sequential(
            nn.Linear(config.n_input_vector, config.n_hidden_cond),
            nn.LayerNorm(config.n_hidden_cond),
            nn.ReLU(),
            nn.Linear(config.n_hidden_cond, config.hidden_size) # Project to GPT's hidden size
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.processor(x)
        x = self.dropout(x)
        x = x.unsqueeze(1) # Add sequence dimension: [B, 1, H]
        x = x.repeat(1, self.config.n_prefix_tokens, 1) # Repeat for n_prefix_tokens: [B, P, H]
        return x
    

class PrependGPT(GPT2LMHeadModel):
    """
    Inherits from GPT2LMHeadModel and adds prefix tokens to the input_ids.
    """
    config_class = PrependGPT2Config
    def __init__(self, config: PrependGPT2Config):
        super().__init__(config)
        self.conditioning = PrependEncoder(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        condition_values=None,
        labels=None,
        past_key_values=None,
        **kwargs
    ):
        if past_key_values is not None:
            input_embds = self.transformer.wte(input_ids)
            all_embds = input_embds

            if 'inputs_embeds' in kwargs:
                _ = kwargs.pop("inputs_embeds", None)
            outputs = super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                labels=labels,
                inputs_embeds=all_embds,
                past_key_values=past_key_values,
                **kwargs
            )
            return outputs
        else:
            prefix_embeddings = self.conditioning.forward(condition_values)
            batch_size = prefix_embeddings.shape[0]
            prefix_length = prefix_embeddings.shape[1]

            input_embds = self.transformer.wte(input_ids)
            all_embds = torch.cat([prefix_embeddings, input_embds], dim=1)

            if attention_mask is not None:
                prefix_attention = torch.ones(
                    batch_size,
                    prefix_length,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)

            if labels is not None:
                prefix_labels = torch.full(
                    (batch_size, prefix_length),
                    fill_value=-100,
                    dtype=labels.dtype,
                    device=labels.device
                )
                labels = torch.cat([prefix_labels, labels], dim=1)

            if 'inputs_embeds' in kwargs:
                _ = kwargs.pop("inputs_embeds", None)

            outputs = super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                labels=labels,
                inputs_embeds=all_embds,
                past_key_values=past_key_values,
                **kwargs
            )
            logits = outputs.logits
            logits_slice_prefix = logits[:, prefix_length:, :]

            new_outputs = CausalLMOutputWithPast(
                loss=outputs.loss,
                logits=logits_slice_prefix,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions
            )

            return new_outputs