from argparse import Namespace
from vllm.utils import FlexibleArgumentParser
from apply_taxonomy.input_output_utils import InputData, construct_raw_prompt, load_images
from apply_taxonomy.vllms import VLM, model_example_map
from glob import glob
import pandas as pd
import os
import logging

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
        "--model-type",
        "-m",
        type=str,
        default="gemma3",
        choices=model_example_map.keys(),
        help='Huggingface "model_type".',
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
        "--question",
        type=str,
        default="What is the content of these images?",
        help="The question to ask the model.",
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
        "--system-prompt",
        action="store_true",
        help="Whether to include a system prompt in the chat template.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Whether to include a system prompt in the chat template.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        help="path to the directory containing the images",
        default="/home/knf792/gits/MMFC-cnotes/docs/test_images/",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        help="path to where to save the results",
        default="/home/knf792/gits/MMFC-cnotes/results/vlm_annotation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing images. This is useful for large datasets.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate.",
    )
    return parser.parse_args()


def load_inputs(args, vlm):
    image_paths = glob(args.image_path + "/*.png")
    notes = pd.read_csv(args.notes_path)
    
    inputs = []
    for image_path, note, post in zip(
        image_paths, notes["note"].value, notes["post"].value
    ):  
        
        input_data = InputData(
            image_path=image_path,
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


def main(args: Namespace):

    vlm = VLM(args)
    logging.info(f"Using model: {args.model_type}")
    
    inputs = load_inputs(image_paths, vlm)
    logging.info(f"Loaded {len(inputs)} inputs for processing.")
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop_token_ids=req_data.stop_token_ids,
        guided_decoding=guided_decoding_params,
    )
        
    for batch_data in range(0, len(inputs), args.batch_size):
        batch_id = batch_data // args.batch_size
        logging.info(f"Processing batch {batch_id + 1} with {args.batch_size} inputs.")
        
        batch_inputs = inputs[batch_data:batch_data + args.batch_size]
        image_paths = [input_data.image_path for input_data in batch_inputs]
        images = load_images(image_paths, enable_smart_resize=True)
        
        for input, image in zip(batch_inputs, images):
            input.image = image
            input.chat_template = vlm.apply_chat_template(input.user_prompt, input.system_prompt)
            
        logging.info(f"Loaded {len(images)} images for batch {batch_id + 1}.")
        
        # Prepare the prompt and system prompt
               
        outputs = vlm.batch_generate(
            batch_inputs,
            sampling_params
        )
        
        # Process outputs as needed, e.g., save to file or print
        for output in outputs:
            print(output)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
