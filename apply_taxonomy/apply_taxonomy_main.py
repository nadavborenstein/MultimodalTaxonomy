from argparse import Namespace
from vllm.utils import FlexibleArgumentParser
from input_output_utils import (
    InputData,
    construct_raw_prompt,
    load_images,
    process_outputs,
)
from structured_outputs import Labels
from vllms import VLM
from glob import glob
import pandas as pd
import os
import logging
from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from typing import List

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
        choices=VLM.models.keys(),
        help='Huggingface "model_type".',
    )
    parser.add_argument(
        "--notes-path",
        type=str,
        help="path to the directory containing the images",
        default="/home/knf792/gits/MultimodalTaxonomy/data/tweets_with_images.csv",
    )
    parser.add_argument(
        "--prompt-path",
        type=str,
        help="path to the directory containing the images",
        default="/home/knf792/gits/MultimodalTaxonomy/prompts/",
    )
    parser.add_argument(
        "--debug-mode", action="store_true", help="Whether to load the model or not. "
    )
    parser.add_argument(
        "--taxonomy-level",
        type=str,
        default="multimodal_taxonomy",
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
        default="type_analysis",
        choices=["type_analysis", "type_analysis_binary"],
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
        default="/home/knf792/gits/MultimodalTaxonomy/results/",
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
        default=2048,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum samples to load.",
    )
    return parser.parse_args()


def load_inputs(args, vlm):
    notes = pd.read_csv(args.notes_path)
    if args.max_samples > 0:
        notes = notes.head(args.max_samples)

    inputs = []
    for image_name, note, post in zip(
        notes["image_name"].values, notes["summary"].values, notes["full_text"].values
    ):

        input_data = InputData(
            image_path=args.image_path + image_name,
            note=note,
            post=post,
            image_placeholder=vlm.image_substring_marker,
            taxonomy_level=args.taxonomy_level,
        )
        prompt = construct_raw_prompt(args.prompt_path, input_data)
        input_data.system_prompt = prompt["system_prompt"]
        input_data.user_prompt = prompt["user_prompt"]
        inputs.append(input_data)
    return inputs


def one_chat_turn(vlm: VLM, inputs_data: List[InputData], images: List[Image.Image]):
    """
    Generates outputs for a single chat turn using the VLM.
    """

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        stop_token_ids=None,
        guided_decoding=GuidedDecodingParams(json=Labels.model_json_schema()),
    )


def main(args: Namespace):

    vlm = VLM(args)
    logging.info(f"Using model: {args.model_name}")

    inputs = load_inputs(args, vlm)
    logging.info(f"Loaded {len(inputs)} inputs for processing.")

    schema = Labels
    guided_decoding_params = GuidedDecodingParams(json=schema.model_json_schema())

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop_token_ids=None,
        guided_decoding=guided_decoding_params,
    )
    all_outputs = []
    for batch_data in range(0, len(inputs), args.batch_size):
        batch_id = batch_data // args.batch_size
        logging.info(f"Processing batch {batch_id + 1} with {args.batch_size} inputs.")

        batch_inputs = inputs[batch_data : batch_data + args.batch_size]
        image_paths = [input_data.image_path for input_data in batch_inputs]
        images = load_images(image_paths, enable_smart_resize=True)

        for input in batch_inputs:
            input.chat_template = vlm.apply_chat_template(
                input.user_prompt, input.system_prompt
            )

        logging.info(f"Loaded {len(images)} images for batch {batch_id + 1}.")

        # Prepare the prompt and system prompt
        logging.info(f"Generating outputs for batch {batch_id + 1}.")
        # Generate outputs using the VLM
        outputs = vlm.batch_generate(batch_inputs, images, sampling_params)
        logging.info(f"Done generating outputs for batch {batch_id + 1}.")
        # Process outputs as needed, e.g., save to file or print
        processed_outputs = process_outputs(outputs, batch_inputs)
        all_outputs.extend(processed_outputs)

    # Save all outputs to a CSV file
    logging.info("Saving outputs to CSV file.")
    output_df = pd.DataFrame(all_outputs)
    output_df.to_csv(
        os.path.join(args.save_path, f"vlm_outputs_{args.model_name}.csv"),
        index=False,
    )
    logging.info(f"Outputs saved to {args.save_path}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
