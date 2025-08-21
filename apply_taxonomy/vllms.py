import os
from contextlib import contextmanager
from dataclasses import asdict
from PIL import Image
from argparse import Namespace
from typing import NamedTuple, Optional, List

from glob import glob

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoProcessor

from vllm import LLM, EngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.multimodal.image import convert_image_mode
from vllm.sampling_params import GuidedDecodingParams
import json
import pandas as pd
from input_output_utils import InputData
from qwen_vl_utils import smart_resize


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str
    image_data: list[Image.Image] = None
    stop_token_ids: Optional[list[int]] = None
    chat_template: Optional[str] = None
    lora_requests: Optional[list[LoRARequest]] = None


@contextmanager
def time_counter(enable: bool):
    if enable:
        import time

        start_time = time.time()
        yield
        elapsed_time = time.time() - start_time
        print("-" * 50)
        print("-- generate time = {}".format(elapsed_time))
        print("-" * 50)
    else:
        yield


class VLM(object):
    models = {
        "gemma3": "google/gemma-3-12b-it",
        "qwen25": "Qwen/Qwen2.5-VL-7B-Instruct",
        "phi4mm": "microsoft/Phi-4-multimodal-instruct",
    }
    image_placeholder_map = {
        "gemma3": "<start_of_image>",
        "qwen25": "<|vision_start|><|image_pad|><|vision_end|>",
        "phi4mm": "<|image_1|>",
    }

    def __init__(self, args: Namespace):
        self.args = args
        self.model_name = args.model_name
        self.seed = args.seed
        assert (
            self.model_name in self.models
        ), f"Invalid model name: {self.model_name}. Available models: {self.models.keys()}"
        self.model = self.models[self.model_name]
        self.image_substring_marker = self.image_placeholder_map[self.model_name]

        self.processor = AutoProcessor.from_pretrained(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        if args.debug_mode:
            self.vlm = None
        else:
            self.vlm = self.load_vlm()

    def load_vlm(self):
        if self.model_name == "gemma3":
            self.engine_args = EngineArgs(
                model=self.model,
                max_model_len=8192,
                max_num_seqs=2,
                limit_mm_per_prompt={"image": 1},
                seed=self.seed,
            )
            self.lora_requests = None

        elif self.model_name == "qwen25":
            self.engine_args = EngineArgs(
                model=self.model,
                max_model_len=32768 if smart_resize is None else 14000,
                max_num_seqs=5,
                limit_mm_per_prompt={"image": 1},
                seed=self.seed,
            )
            self.lora_requests = None

        elif self.model_name == "phi4mm":
            model_path = snapshot_download(self.model)
            vision_lora_path = os.path.join(model_path, "vision-lora")
            self.engine_args = EngineArgs(
                model=model_path,
                trust_remote_code=True,
                max_model_len=20000,
                max_num_seqs=2,
                limit_mm_per_prompt={"image": 1},
                enable_lora=True,
                max_lora_rank=320,
                seed=self.seed,
            )
            self.lora_requests = [LoRARequest("vision", 1, vision_lora_path)]
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return LLM(**asdict(self.engine_args))

    def apply_chat_template(self, system_prompt: str, user_prompt: str) -> str:
        """
        Applies a chat template to the given system and user prompts.
        This is a placeholder method and should be implemented in subclasses.
        """
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    def batch_generate(
        self,
        input_data: List[InputData],
        images: List[Image.Image],
        sampling_params: SamplingParams,
    ):
        batch_size = self.args.batch_size
        assert len(input_data) == batch_size, "Number of images must match batch size."

        inputs = []

        for input, image in zip(input_data, images):
            inputs.append(
                {
                    "prompt": input.chat_template,
                    "multi_modal_data": {"image": image},
                }
            )
        lora_request = (
            [
                LoRARequest("vision", i + 1, self.lora_requests[0].path)
                for i in range(batch_size)
            ]
            if self.lora_requests
            else None
        )
        with time_counter(True):
            outputs = self.vlm.generate(
                inputs, sampling_params=sampling_params, lora_request=lora_request
            )
        return outputs

