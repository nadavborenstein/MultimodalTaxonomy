import os


from apply_taxonomy.multimodal_interactor import (
    MultimodalQAInteractor,
    MultimodalNote,
    MultimodalAnnotationInteractor,
)
from apply_taxonomy.demonstrations import Demonstrations
from apply_taxonomy.input_output_utils import load_taxonomy
from vllm.utils import FlexibleArgumentParser
from apply_taxonomy.vllms import VLM
import yaml

# include /Users/knf792/gits/MultimodalTaxonomy im the namepsace

from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
from vllm import LLM, EngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.multimodal.image import convert_image_mode
from vllm.sampling_params import GuidedDecodingParams
from dataclasses import asdict


def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "vision language models that support multi-image input for text "
        "generation"
    )
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="gemma3",
        # choices=VLM.models.keys(),
        help='Huggingface "model_type".',
    )
    parser.add_argument(
        "--notes-path",
        type=str,
        help="path to the directory containing the images",
        default="data/tweets_with_images.csv",
    )
    parser.add_argument(
        "--prompt-path",
        type=str,
        help="path to the directory containing the images",
        default="prompts/flowchart_prompt_annotations.txt",
    )
    parser.add_argument(
        "--debug-mode", action="store_true", help="Whether to load the model or not. "
    )
    parser.add_argument(
        "--taxonomy-path",
        type=str,
        help="path to the taxonomy file",
        default="prompts/full_taxonomy.json",
    )
    parser.add_argument(
        "--taxonomy-level",
        nargs="+",
        type=str,
        default=["multimodal_taxonomy"],
        help="The taxonomy level to apply. Currently only 'type' and 'subtype' are supported.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        help="path to the directory containing the images",
        default="/home/knf792/gits/MMFC-cnotes/data/tweet_images/",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="generate",
        choices=["generate", "chat"],
        help="The method to run in `vllm.LLM`.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set the seed when initializing `vllm.LLM`.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="baseline",
        choices=["baseline", "flowchart"],
        help="The task to run. Currently only 'type_analysis' is supported.",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.0,
        help="The temperature to use for sampling. 0.0 means greedy decoding.",
    )
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        help="Whether to run the model in zero-shot mode. ",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Whether to include a system prompt in the chat template.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        help="path to where to save the results",
        default="results/",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing images. This is useful for large datasets.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum samples to load.",
    )
    parser.add_argument(
        "--num-demos",
        type=int,
        default=4,
        help="Number of demonstrations to include in the prompt.",
    )
    parser.add_argument(
        "--demonstration-type",
        type=str,
        default="same_single",
        choices=[
            "same_single",
            "same_flow",
            "all",
            "random",
            "random_2",
            "random_3",
            "random_4",
            "random_5",
        ],
        help="Type of demonstrations to include in the prompt.",
    )
    parser.add_argument(
        "--demo-data-path",
        type=str,
        help="Path to the demonstration data CSV file.",
        default="data/qualification_dataset_en.csv",
    )
    parser.add_argument(
        "--demo-images-path",
        type=str,
        help="Path to the demonstration images directory.",
        default="data/qualification_images/",
    )
    return parser.parse_args(
        args=[
            "--taxonomy_path",
            "prompts/taxonomy_annoation_experiment.yaml",
            "--demo-images-path",
            "data/qualification_images/",
            "--demo-data-path",
            "data/qualification_dataset_en.csv",
            "--prompt-path",
            "prompts/flowchart_prompt_annotations.txt",
            "--demonstration-type",
            "same_flow",
            "--model-name",
            "smolvlm",
        ]
    )


def main():
    print("Hello from testbed!")
    args = parse_args()
    taxonomy = load_taxonomy(args.taxonomy_path)
    note = MultimodalNote(
        user="Ordnance Arbiter.",
        note="The Americans joined the war right at the end of 1941",
        post="@mikenelson586 Wondering how many Brits there were",
        image_path="'/Users/knf792/Documents/danish-vocab-extention copy/icon.png'",
    )

    interactor = MultimodalAnnotationInteractor(note=note, args=args)

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    messages = interactor.messages
    tokenized = tokenizer.apply_chat_template(messages, tokenize=False)

    llm = LLM(
        **asdict(
            EngineArgs(
                model="HuggingFaceTB/SmolVLM-256M-Instruct",
                seed=args.seed,
                max_model_len=1024,
            )
        )
    )
    print('Done')


if __name__ == "__main__":
    main()
