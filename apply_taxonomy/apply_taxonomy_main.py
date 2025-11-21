from argparse import Namespace
from vllm.utils import FlexibleArgumentParser
from structured_outputs import Labels
from vllms import VLM
from glob import glob
import pandas as pd
import os
import logging
from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from typing import List
from PIL import Image
from multimodal_interactor import (
    MultimodalNote,
    MultimodalQAInteractor,
    MultimodalTaxonomyInteractor,
)

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
        default="/home/knf792/gits/MultimodalTaxonomy/prompts/main_flat_prompt.txt",
    )
    parser.add_argument(
        "--debug-mode", action="store_true", help="Whether to load the model or not. "
    )
    parser.add_argument(
        "--taxonomy-path",
        type=str,
        help="path to the taxonomy file",
        default="/home/knf792/gits/MultimodalTaxonomy/prompts/full_taxonomy.json",
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
        default=1024,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum samples to load.",
    )
    return parser.parse_args()


def log_args(args):
    logging.info("Arguments:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")


def load_inputs(args, vlm):
    notes = pd.read_csv(args.notes_path)
    interactor_type = (
        MultimodalTaxonomyInteractor
        if "baseline" in args.task
        else MultimodalQAInteractor
    )
    if args.max_samples > 0:
        notes = notes.head(args.max_samples)

    interactors = []
    for image_name, image_url, note, post in zip(
        notes["image_name"].values,
        notes["image_urls"].values,
        notes["summary"].values,
        notes["full_text"].values,
    ):

        input_data = MultimodalNote(
            image_path=args.image_path + image_name,
            image_url=image_url,
            note=note,
            post=post,
        )
        interactor = interactor_type(
            note=input_data,
            args=args,
            image_substring_marker=vlm.image_substring_marker,
        )
        interactors.append(interactor)
    return interactors


def update_taxonomy_level(
    interactors: List[MultimodalTaxonomyInteractor], taxonomy_level
) -> None:
    for interactor in interactors:
        interactor.update_taxonomy(level=taxonomy_level)


def get_all_outputs(interactors: List[MultimodalTaxonomyInteractor]) -> List[dict]:
    all_outputs = []
    for interactor in interactors:
        all_outputs.append(interactor.get_output_dict())
    return all_outputs


def baseline_main(args):
    logging.info(f"starting baseline run...")
    logging.info(f"Using model: {args.model_name}")
    vlm = VLM(args)
    logging.info(f"Model loaded.")

    logging.info(f"loading inputs.")
    interactors = load_inputs(args, vlm)

    logging.info(f"Loaded {len(interactors)} inputs for processing.")

    for batch_data in range(0, len(interactors), args.batch_size):
        batch_id = batch_data // args.batch_size
        logging.info(f"Processing batch {batch_id + 1} with {args.batch_size} inputs.")

        batch_interactors = interactors[batch_data : batch_data + args.batch_size]
        logging.info(f"loading images")
        for interactor in batch_interactors:
            interactor.load_image(enable_smart_resize=True)

        for taxonomy_level in args.taxonomy_level:
            update_taxonomy_level(batch_interactors, taxonomy_level)
            logging.info(f"Set taxonomy level to {taxonomy_level}.")
            guided_decoding_params = interactors[0].get_guided_decoding_params()
            sampling_params = SamplingParams(
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stop_token_ids=None,
                guided_decoding=guided_decoding_params,
                repetition_penalty=1.1,
                stop=["\n    \n    \n", "\n  \n  \n", "\n \n \n", "\n\n\n\n"],
            )
            logging.info(f"Generating outputs for batch {batch_id + 1}.")
            outputs = vlm.batch_generate(batch_interactors, sampling_params)
            logging.info(f"Done generating outputs for batch {batch_id + 1}.")
            logging.info(f"updating labels with llm answers")
            for interactor, output in zip(batch_interactors, outputs):
                interactor.process_output(output)

        logging.info(f"Done processing batch {batch_id + 1}.")
        logging.info(f"Deleting images")
        for interactor in batch_interactors:
            interactor.remove_image()
    all_outputs = get_all_outputs(interactors)
    logging.info("Saving outputs to CSV file.")
    output_df = pd.DataFrame(all_outputs)
    output_df.to_csv(
        os.path.join(args.save_path, f"vlm_outputs_{args.model_name}.csv"),
        index=False,
    )
    logging.info(f"Outputs saved to {args.save_path}.")


def flowchart_main(args):
    logging.info(f"starting baseline run...")
    logging.info(f"Using model: {args.model_name}")
    vlm = VLM(args)
    logging.info(f"Model loaded.")

    logging.info(f"loading inputs.")
    interactors = load_inputs(args, vlm)

    logging.info(f"Loaded {len(interactors)} inputs for processing.")


if __name__ == "__main__":
    args = parse_args()
    log_args(args)
    baseline_main(args)
