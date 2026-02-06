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
"""Testing suite for the PyTorch VoxtralRealtime model."""

import tempfile
import unittest

from transformers import (
    VoxtralRealtimeConfig,
    VoxtralRealtimeForConditionalGeneration,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch


class VoxtralRealtimeModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        audio_token_id=0,
        seq_length=35,
        feat_seq_length=60,
        text_config={
            "model_type": "voxtral_realtime_text",
            "intermediate_size": 36,
            "initializer_range": 0.02,
            "hidden_size": 32,
            "max_position_embeddings": 52,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "use_labels": True,
            "vocab_size": 99,
            "head_dim": 8,
            "pad_token_id": 1,  # can't be the same as the audio token id
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "attention_dropout": 0.0,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
        },
        is_training=True,
        audio_config={
            "model_type": "voxtral_realtime_encoder",
            "hidden_size": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 32,
            "encoder_layers": 2,
            "num_mel_bins": 80,
            "d_model": 16,
            "max_position_embeddings": 100,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "activation_function": "silu",
            "activation_dropout": 0.0,
            "attention_dropout": 0.0,
            "head_dim": 4,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.audio_token_id = audio_token_id
        self.text_config = text_config
        self.audio_config = audio_config
        self.seq_length = seq_length
        self.feat_seq_length = feat_seq_length

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.encoder_seq_length = seq_length

    def get_config(self):
        return VoxtralRealtimeConfig(
            text_config=self.text_config,
            audio_config=self.audio_config,
            ignore_index=self.ignore_index,
            audio_token_id=self.audio_token_id,
        )

    def prepare_config_and_inputs(self):
        input_features_values = floats_tensor(
            [
                self.batch_size,
                self.audio_config["num_mel_bins"],
                self.feat_seq_length,
            ]
        )
        config = self.get_config()
        return config, input_features_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_features_values = config_and_inputs
        num_audio_tokens_per_batch_idx = 30

        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(torch_device)
        attention_mask[:, :1] = 0

        input_ids[:, 1 : 1 + num_audio_tokens_per_batch_idx] = config.audio_token_id
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features_values,
        }
        return config, inputs_dict


@require_torch
class VoxtralRealtimeForConditionalGenerationModelTest(
    ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    """
    Model tester for `VoxtralRealtimeForConditionalGeneration`.
    """

    all_model_classes = (VoxtralRealtimeForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"any-to-any": VoxtralRealtimeForConditionalGeneration}
        if is_torch_available()
        else {}
    )

    _is_composite = True

    def setUp(self):
        self.model_tester = VoxtralRealtimeModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VoxtralRealtimeConfig, has_text_modality=False)

    @unittest.skip(
        reason="This test does not apply to VoxtralRealtime since inputs_embeds corresponding to audio tokens are replaced when input features are provided."
    )
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(
        reason="VoxtralRealtime need lots of steps to prepare audio/mask correctly to get pad-free inputs. Cf llava (reference multimodal model)"
    )
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(
        reason="VoxtralRealtime need lots of steps to prepare audio/mask correctly to get pad-free inputs. Cf llava (reference multimodal model)"
    )
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(
        reason="VoxtralRealtime need lots of steps to prepare audio/mask correctly to get pad-free inputs. Cf llava (reference multimodal model)"
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(
        reason="VoxtralRealtime need lots of steps to prepare audio/mask correctly to get pad-free inputs. Cf llava (reference multimodal model)"
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids_and_fa_kwargs(self):
        pass

    @unittest.skip(
        reason="VoxtralRealtime need lots of steps to prepare audio/mask correctly to get pad-free inputs. Cf llava (reference multimodal model)"
    )
    def test_flash_attention_3_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(
        reason="VoxtralRealtime need lots of steps to prepare audio/mask correctly to get pad-free inputs. Cf llava (reference multimodal model)"
    )
    def test_flash_attention_3_padding_matches_padding_free_with_position_ids_and_fa_kwargs(self):
        pass

    @unittest.skip(reason="VoxtralRealtime has no separate base model without a head.")
    def test_model_base_model_prefix(self):
        pass

    def test_sdpa_can_dispatch_composite_models(self):
        # overwrite because VoxtralRealtime is audio+text model (not vision+text)
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                text_attn = "sdpa" if model.language_model._supports_sdpa else "eager"
                vision_attn = "sdpa" if model.audio_tower._supports_sdpa else "eager"

                # `None` as it is the requested one which will be assigned to each sub-config
                # Sub-model will dispatch to SDPA if it can (checked below that `SDPA` layers are present)
                self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")
                self.assertTrue(model.language_model.config._attn_implementation == text_attn)
                self.assertTrue(model.audio_tower.config._attn_implementation == vision_attn)

                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                self.assertTrue(model_eager.config._attn_implementation == "eager")
                self.assertTrue(model_eager.language_model.config._attn_implementation == "eager")
                self.assertTrue(model_eager.audio_tower.config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                        raise ValueError("The eager model should not have SDPA attention layers")


# TODO: Add integration tests once checkpoint is available
# @require_torch
# class VoxtralRealtimeForConditionalGenerationIntegrationTest(unittest.TestCase):
#     def setUp(self):
#         self.checkpoint_name = "mistralai/VoxtralRealtime-Mini-3B-2507"
#         self.dtype = torch.bfloat16
#         self.processor = AutoProcessor.from_pretrained(self.checkpoint_name)
#
#     def tearDown(self):
#         cleanup(torch_device, gc_collect=True)
#
#     @slow
#     def test_realtime_streaming_inference(self):
#         """Test streaming inference with the realtime model."""
#         pass
