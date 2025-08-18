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

from qwen_vl_utils import smart_resize


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str
    image_data: list[Image.Image] = None
    stop_token_ids: Optional[list[int]] = None
    chat_template: Optional[str] = None
    lora_requests: Optional[list[LoRARequest]] = None


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

    def __init__(
        self, args: Namespace
    ): 
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
        if args.load_model:
            self.vlm = self.load_vlm()
        else:
            self.vlm = None

    @contextmanager
    def time_counter(self, enable: bool):
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

        self.vlm = LLM(**asdict(self.engine_args))

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
    
    def batch_generate(self, prompt: str, images: List[Image.Image], sampling_params: SamplingParams):
        batch_size = self.args.batch_size
        assert len(images) == batch_size, "Number of images must match batch size."
        
        inputs = []

        for i, image in enumerate(images):
            inputs.append(
                {
                    "prompt": prompt,
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
        with self.time_counter(True):
            outputs = self.vlm.generate(
                inputs, sampling_params=sampling_params, lora_request=lora_request
            )
        return outputs


def run_generate(
    args,
    prompt: str,
    examples: List[str],
    image_paths: List[str],
    system_prompt: Optional[str] = None,
):

    req_data = model_example_map[args.model_type](
        prompt, num_images=len(examples) + 1, system_prompt=system_prompt
    )
    schema = ImageDescriptorBinary if "binary" in args.task else ImageDescriptor
    guided_decoding_params = GuidedDecodingParams(json=schema.model_json_schema())
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop_token_ids=req_data.stop_token_ids,
        guided_decoding=guided_decoding_params,
    )

    engine_args = asdict(req_data.engine_args) | {"seed": args.seed}
    llm = LLM(**engine_args)
    examples = load_images(examples, enable_smart_resize=args.model_type == "qwen25")
    batch_generate(
        args,
        llm,
        req_data,
        sampling_params,
        examples=examples,
        image_paths=image_paths,
    )


def run_chat(
    model: str,
    question: str,
    image_paths: list[str],
    seed: Optional[int],
    temperature: float = 0.0,
    max_tokens: int = 256,
):
    req_data = model_example_map[model](question, image_paths)

    # Disable other modalities to save memory
    default_limits = {"image": 0, "video": 0, "audio": 0}
    req_data.engine_args.limit_mm_per_prompt = default_limits | dict(
        req_data.engine_args.limit_mm_per_prompt or {}
    )

    engine_args = asdict(req_data.engine_args) | {"seed": seed}
    llm = LLM(**engine_args)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop_token_ids=req_data.stop_token_ids,
    )
    outputs = llm.chat(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question,
                    },
                    *(
                        {
                            "type": "image_url",
                            "image_url": {"url": image_path},
                        }
                        for image_path in image_paths
                    ),
                ],
            }
        ],
        sampling_params=sampling_params,
        chat_template=req_data.chat_template,
        lora_request=req_data.lora_requests,
    )

    print("-" * 50)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        print("-" * 50)


def main(args: Namespace):
    model = args.model_type
    method = args.method
    seed = args.seed

    suffix = "chat" if args.chat else ""
    suffix = "zero_shot" if args.zero_shot and not suffix else suffix

    prompt = load_prompt(args.task, suffix=suffix)
    system_prompt = (
        load_prompt(args.task, suffix="system_prompt") if args.system_prompt else None
    )
    examples = load_examples(args.task) if not args.zero_shot else []
    image_paths = glob(args.image_path + "/*.png")
    print(f"There are a total of {len(image_paths)} images")

    done_path = os.path.join(args.save_path, "image_type_annotation.csv")
    if os.path.exists(done_path):
        print()
        done = pd.read_csv(done_path)
        done_images = set(done["image_path"].values)
        print(f"There are a total of {len(done_images)} images already processed")
        image_paths = [path for path in image_paths if path not in done_images]
        print(f"Processing {len(image_paths)} images.")

    if method == "generate":
        run_generate(
            args,
            prompt,
            examples,
            image_paths,
            system_prompt=system_prompt,
        )
    elif method == "chat":
        run_chat(model, prompt, image_paths, seed)
    else:
        raise ValueError(f"Invalid method: {method}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
