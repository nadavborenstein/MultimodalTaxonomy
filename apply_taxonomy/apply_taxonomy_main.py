from argparse import Namespace
from vllm.utils import FlexibleArgumentParser
from vllms import model_example_map


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