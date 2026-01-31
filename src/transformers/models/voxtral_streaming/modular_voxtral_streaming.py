# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from ...models.voxtral.modeling_voxtral import VoxtralEncoder, VoxtralForConditionalGeneration, VoxtralEncoderLayer
from ...models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding, LlamaAttention
from ...models.mistral.modeling_mistral import MistralRMSNorm, MistralMLP
from ...models.mistral.modeling_mistral import MistralForCausalLM
from ...models.mimi.modeling_mimi import MimiConv1d
from ...models.llama.modeling_llama import LlamaMLP


class VoxtralStreamingRotaryEmbedding(LlamaRotaryEmbedding): ...


class VoxtralStreamingCausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        self.left_pad = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = nn.functional.pad(x, (self.left_pad, 0))
        x = super().forward(x)

        if mask is not None:
            mask = nn.functional.pad(mask, (self.left_pad, 0))[:, None, :]
            weight = torch.ones(1, 1, self.kernel_size[0], device=mask.device)
            mask = nn.functional.conv1d(mask.float(), weight, stride=self.stride)
            mask = mask > 0
            x *= mask

        if mask is not None:
            mask = mask.squeeze(1)
        return x, mask


class VoxtralStreamingRMSNorm(MistralRMSNorm): ...


class VoxtralStreamingAttention(LlamaAttention):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=True
        )


class VoxtralStreamingMLP(MistralMLP):
    def __init__(self, config):
        super().__init__(config)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)


class VoxtralStreamingEncoderLayer(VoxtralEncoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config)
        self.self_attn_layer_norm = VoxtralStreamingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = VoxtralStreamingAttention(config, layer_idx)
        self.mlp = VoxtralStreamingMLP(config)
        self.final_layer_norm = VoxtralStreamingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        del self.fc1
        del self.fc2


class VoxtralStreamingEncoder(VoxtralEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.rotary_emb = VoxtralStreamingRotaryEmbedding(config)
        self.conv1 = VoxtralStreamingCausalConv1d(config.num_mel_bins, config.d_model, kernel_size=3, padding=1)
        self.conv2 = VoxtralStreamingCausalConv1d(config.d_model, config.d_model, kernel_size=3, stride=2, padding=1)
        self.layer_norm = VoxtralStreamingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList([VoxtralStreamingEncoderLayer(config, layer_idx) for layer_idx in range(config.encoder_layers)])

        del self.embed_positions


# class MistralStreamingAdaRmsNorm(nn.Module):
#     def __init__(self, config: MistralConfig):
#         super().__init__()
#         # TODO: how to add the intermediate size to the config? since it already the mistral one? new model? new config only?
#         self.linear1 = nn.Linear(config.hidden_size, 32, bias=False)
#         self.linear2 = nn.Linear(32, config.hidden_size, bias=False)

#     def forward(self, hidden_states):
#         hidden_states = self.linear1(hidden_states)
#         hidden_states = nn.functional.gelu(hidden_states)
#         hidden_states = self.linear2(hidden_states)
#         return hidden_states


# class MistralStreamingDecoderLayer(MistralDecoderLayer):
#     def __init__(self, config: MistralConfig, layer_idx: int):
#         super().__init__(config, layer_idx)
#         self.ada_rms_norm = MistralStreamingAdaRmsNorm(config)


# class MistralStreamingForCausalLM(MistralForCausalLM): ...


class VoxtralStreamingForConditionalGeneration(VoxtralForConditionalGeneration): ...
    # def __init__(self, config):
    #     super().__init__(config)
    #     # TODO: what name to use here?
    #     self.language_model = MistralStreamingForCausalLM(config.text_config)


__all__ = [
    "VoxtralStreamingForConditionalGeneration",
    "VoxtralStreamingEncoder",
]