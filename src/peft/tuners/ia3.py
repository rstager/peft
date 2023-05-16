# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..import_utils import is_bnb_available
from ..utils import (
    TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftConfig,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)


if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class IA3Config(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`IA3Model`].

    Args:
        init_scale (`float`): The scaling rank for IA3.
        target_modules (`Union[List[str],str]`): The names of the modules to apply IA3 to.
         (`float`): The scaling rank for ia3.
        modules_to_save (`List[str]`):List of modules apart from IA3 layers to be set as trainable
            and saved in the final checkpoint.
    """

    init_scale: float = field(default=0.0, metadata={"help": "IA3 init scaling"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with IA3."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )

    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from IA3 layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )


    def __post_init__(self):
        self.peft_type = PeftType.IA3


class IA3Model(torch.nn.Module):
    """
    Creates IA3 model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`IA3Config`]): The configuration of the IA3 model.

    Returns:
        `torch.nn.Module`: The IA3 model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM, IA3Config
        >>> from peft import IA3Model, IA3Config

        >>> config = IA3Config(
        ...     peft_type="IA3",
        ...     task_type="SEQ_2_SEQ_LM",
        ...     init_scale=0.0,
        ...     target_modules=["q", "v"],
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> ia3_model = IA3Model(config, model)
        ```

        ```py
        >>> import transformers
        >>> from peft import IA3Config, PeftModel, get_peft_model, prepare_model_for_int8_training

        >>> target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        >>> config = IA3Config(
        ...     init_scale=0.0, target_modules=target_modules, task_type="CAUSAL_LM"
        ... )

        >>> model = transformers.GPTJForCausalLM.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     pad_token_id=tokenizer.eos_token_id,
        ...     use_cache=False,
        ...     device_map={"": rank},
        ...     torch_dtype=torch.float16,
        ...     load_in_8bit=True,
        ... )
        >>> model = prepare_model_for_int8_training(model)
        >>> IA3_model = get_peft_model(model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`IA3Config`]): The configuration of the IA3 model.
    """

    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_ia3_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        mark_only_ia3_as_trainable(self.model)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name):
        ia3_config = self.peft_config[adapter_name]
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use ia3 with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        kwargs = {
            "init_scale": ia3_config.init_scale,
            "fan_in_fan_out": ia3_config.fan_in_fan_out,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(ia3_config.target_modules, str):
                target_module_found = re.fullmatch(ia3_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in ia3_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                if hasattr(target, "bias"):
                    bias = target.bias is not None


                if isinstance(target, IA3Layer):
                    target.update_layer(
                        adapter_name,
                        ia3_config.init_scale,
                    )
                else:
                    if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                        eightbit_kwargs = kwargs.copy()
                        eightbit_kwargs.update(
                            {
                                "has_fp16_weights": target.state.has_fp16_weights,
                                "memory_efficient_backward": target.state.memory_efficient_backward,
                                "threshold": target.state.threshold,
                                "index": target.index,
                            }
                        )
                        new_module = Linear8bitLt(
                            adapter_name, target.in_features, target.out_features, **eightbit_kwargs
                        )
                    else:
                        if isinstance(target, torch.nn.Linear):
                            in_features, out_features = target.in_features, target.out_features
                        elif isinstance(target, Conv1D):
                            in_features, out_features = (
                                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                            )
                            if not kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                                    "Setting fan_in_fan_out to True."
                                )
                                kwargs["fan_in_fan_out"] = ia3_config.fan_in_fan_out = True
                        else:
                            raise ValueError(
                                f"Target module {target} is not supported. "
                                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                            )

                        new_module = Linear(adapter_name, in_features, out_features, bias=bias, **kwargs)

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {ia3_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "ia3_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, IA3Layer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, IA3Layer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, IA3Layer):
                module.merge()

    def unmerge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, IA3Layer):
                module.unmerge()

    @staticmethod
    def _prepare_ia3_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING[model_config["model_type"]]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def merge_and_unload(self):
        r"""
        This method merges the ia3 layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.
        """
        if getattr(self.config, "model_type", None) == "gpt2":
            raise ValueError("GPT2 models are not supported for merging IA3layers")

        if getattr(self.model, "is_loaded_in_8bit", False):
            raise ValueError("Cannot merge IA3 layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "ia3" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, IA3Layer):
                bias = target.bias is not None
                new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model





# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `ia3_only` to work
def mark_only_ia3_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if "ia3_" not in n:
            p.requires_grad = False


class IA3Layer:
    def __init__(
        self,
        out_features: int,
    ):
        self.ia3_init_scale = {}
        self.ia3_scale = nn.ParameterDict({})

        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.out_features = out_features

    def update_layer(self, adapter_name, init_scale, init_weights):
        self.ia3_init_scale[adapter_name] = init_scale
        self.ia3_scale.update(nn.ParameterDict({adapter_name: nn.Parameter(torch.ones(1, self.out_features))}))
        if init_weights:
            self.reset_ia3_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_ia3_parameters(self, adapter_name):
        if adapter_name in self.ia3_scale.keys():
            self.ia3_scale[adapter_name] = nn.Parameter(
                torch.ones(1, self.out_features)
                + torch.randn(1, self.out_features) * self.ia3_init_scale[adapter_name]
            )


class Linear(nn.Linear, IA3Layer):
    # IA3 implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        init_scale: float = 0.0,
        init_ia3_weights: bool = True,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        init_ia3_weights = kwargs.pop("init_ia3_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        IA3Layer.__init__(self, out_features=out_features)

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, init_scale, init_ia3_weights)
        self.active_adapter = adapter_name

    def merge(self):
        if self.active_adapter not in self.ia3_scale.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        assert True, "merge not implemented"


    def unmerge(self):
        if self.active_adapter not in self.ia3_scale.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        assert True, "unmerge not implemented"


    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        if self.active_adapter not in self.ia3_scale.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif not self.merged:
            result =F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            
            result = result.to(self.ia3_scale[self.active_adapter].dtype)
            
            if self.ia3_scale[self.active_adapter].requires_grad:
                result = result * self.ia3_scale[self.active_adapter].flatten()
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result


if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, IA3Layer):
        # IA3 implemented in a dense layer
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            IA3Layer.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.update_layer(adapter_name, ia3_init_scale, init_ia3_weights)
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters or self.active_adapter not in self.ia3_scale.keys():
                return result
            elif self.r[self.active_adapter] > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = (
                        self.lora_B[self.active_adapter](
                            self.ia3_scale[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                        ).to(expected_dtype)
                        * self.scaling[self.active_adapter]
                    )
                else:
                    output = (
                        self.lora_B[self.active_adapter](
                            self.ia3_scale[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                        )
                        * self.scaling[self.active_adapter]
                    )
                result += output
            return result
