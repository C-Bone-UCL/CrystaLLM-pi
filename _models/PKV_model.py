import torch
from transformers import GPT2Config, GPT2LMHeadModel
from torch import nn

class PKVGPT2Config(GPT2Config):
    """
    Configuration class for PKVGPT (PKV method). Inherits from GPT2Config.
    """
    # model_type = "pkv_gpt2"
    
    def __init__(self,
                 n_input_vector: int = 2,
                 n_prefix_tokens: int = 2,
                 n_hidden_cond: int = 128,
                 dropout: float = 0.1,
                 share_layers: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.add_cross_attention = False
        kwargs.pop('add_cross_attention', None)

        self.n_input_vector = n_input_vector
        self.n_prefix_tokens = n_prefix_tokens
        self.n_hidden_cond = n_hidden_cond
        self.dropout = dropout
        self.share_layers = share_layers


class PKVEncoder(nn.Module):
    def __init__(self, config: PKVGPT2Config
):
        super().__init__()
        self.config = config

        # Calculate KV size using unified config
        self.kv_size = (
            config.n_prefix_tokens # so we have n_prefix_tokens kv pairs
            * config.n_head # number of heads (as there are n_head kv pairs per layer)
            * (config.hidden_size // config.n_head) # as each kv pair has hidden_size // n_head dimensions)
            * 2
        )

        if not config.share_layers:
            self.kv_size *= config.n_layer

        self.processor = nn.Sequential(
            nn.Linear(config.n_input_vector, config.n_hidden_cond),
            nn.LayerNorm(config.n_hidden_cond),
            nn.ReLU(),
            nn.Linear(config.n_hidden_cond, config.n_hidden_cond * 2)
        )

        self.to_kv = nn.Linear(config.n_hidden_cond * 2, self.kv_size)
        self.dropout = nn.Dropout(config.dropout) # Use conditioning dropout

    # forward method needs updates to use self.config instead of self.cond_config
    def forward(self, x):
        batch_size = x.shape[0]

        x = self.processor(x)
        x = self.to_kv(x)
        x = self.dropout(x)

        if self.config.share_layers:
            x = x.view(batch_size, self.config.n_prefix_tokens, -1)
            k, v = x.chunk(2, dim=-1)
            k = k.view(
                batch_size,
                self.config.n_prefix_tokens,
                self.config.n_head,
                -1
            ).permute(0, 2, 1, 3)
            v = v.view(
                batch_size,
                self.config.n_prefix_tokens,
                self.config.n_head,
                -1
            ).permute(0, 2, 1, 3)
            pkv = tuple([(k.clone(), v.clone()) for _ in range(self.config.n_layer)])
            return pkv
        else:
            x = x.view(
                batch_size,
                self.config.n_layer,
                self.config.n_prefix_tokens,
                -1
            )
            k, v = x.chunk(2, dim=-1)
            k = k.view(
                batch_size,
                self.config.n_layer,
                self.config.n_prefix_tokens,
                self.config.n_head,
                -1
            ).permute(0, 1, 3, 2, 4)
            v = v.view(
                batch_size,
                self.config.n_layer,
                self.config.n_prefix_tokens,
                self.config.n_head,
                -1
            ).permute(0, 1, 3, 2, 4)
            pkv = tuple((k[:, i], v[:, i]) for i in range(self.config.n_layer))
            return pkv
        
class PKVGPT(GPT2LMHeadModel):
    """
    Inherits from GPT2LMHeadModel and adds a PKVEncoder to the forward pass.
    """
    config_class = PKVGPT2Config

    def __init__(self, config: PKVGPT2Config
):
        super().__init__(config)
        self.conditioning = PKVEncoder(config)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        condition_values=None,
        labels=None,
        **kwargs
    ):
        # Check if cached past_key_values exist in kwargs
        if "past_key_values" in kwargs:
            cached_pkv = kwargs.pop("past_key_values")
            past_key_values = cached_pkv
        else:
            if condition_values is not None:
                past_key_values = self.conditioning.forward(condition_values)
                # Update attention mask to account for prefix
                if attention_mask is not None:
                    batch_size = attention_mask.shape[0]
                    prefix_length = self.config.n_prefix_tokens
                    prefix_attention = torch.ones(
                        batch_size,
                        prefix_length,
                        device=attention_mask.device,
                        dtype=attention_mask.dtype
                    )
                    attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)
            else:
                print("WARNING: Condition values activated but not passed correctly.")
                past_key_values = None

        if past_key_values is None:
            raise ValueError(
                "WARNING: PKV Condition values activated but not passed correctly."
            )

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            labels=labels,
            **kwargs
        )

        return output
        