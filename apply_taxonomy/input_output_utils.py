import json
from typing import List, Dict, Any
from PIL import Image
from qwen_vl_utils import smart_resize
import pandas as pd
from dataclasses import dataclass


@dataclass
class InputData:
    image_path: str
    image_placeholder: str = "<|image_1|>"  # Placeholder for the image in the prompt
    image: Image.Image = None  # Placeholder for the image object
    post: str = ""
    note: str = ""
    taxonomy_level: str = "head"
    system_prompt: str = ""
    user_prompt: str = ""
    chat_template: str = ""

    def __str__(self):
        return f"InputData(image_path={self.image_path}, post={self.post}, note={self.note}, taxonomy_level={self.taxonomy_level})"


def get_possible_taxonomy_levels(taxonomy: Dict[str, Any]) -> List[str]:
    """Get the possible taxonomy levels from the taxonomy JSON file.

    Args:
        path (str): path to the folder containing the taxonomy JSON file.

    Returns:
        List[str]: list of possible taxonomy levels.
    """

    def recursive_key_generator(d: Dict):
        """Recursively generate keys from a dictionary."""
        for key, value in d.items():
            yield key
            if isinstance(value, dict):
                yield from recursive_key_generator(value)

    keys = [
        key
        for key in recursive_key_generator(taxonomy)
        if not key.startswith("label") and not key.startswith("description")
    ]
    return keys


def find_taxonomy_level(
    taxonomy: Dict[str, Any], taxonomy_level: str
) -> Dict[str, Any]:
    """
    Find the specified taxonomy level in the taxonomy dictionary.
    """

    def recursive_find_key(d: Dict, level: str) -> Dict[str, Any]:
        """Recursively find the taxonomy level in the dictionary."""
        if level in d:
            return {level: d[level]}
        for key, value in d.items():
            if isinstance(value, dict):
                result = recursive_find_key(value, level)
                if result:
                    return result
        return {}

    possible_taxonomy_levels = get_possible_taxonomy_levels(taxonomy)
    assert (
        taxonomy_level in possible_taxonomy_levels
    ), f"Invalid taxonomy level: {taxonomy_level}. Possible levels are: {['head'] + possible_taxonomy_levels}"

    if taxonomy_level == "head":
        return taxonomy
    else:
        return recursive_find_key(taxonomy, taxonomy_level)


def multiple_replace(s, replacements: Dict[str, str]) -> str:
    """Replace multiple substrings in a string with corresponding values from a dictionary."""
    for key, replacement in replacements.items():
        s = s.replace(r"{{" + key + r"}}", replacement)
    return s


def construct_raw_prompt(path: str, input_data: InputData) -> Dict[str, str]:
    """_summary_

    Args:
        path (str): path to the folder containing the taxonomy JSON file and the prompt template.
        inputs (Dict, optional): additional inputs to include in the prompt.
        taxonomy_level (str, optional): the level of the taxonomy to use. Defaults to "head" (entire taxonomy).

    Returns:
        str: _description_
    """
    taxonomy_level = input_data.taxonomy_level
    taxonomy = json.load(open(f"{path}/full_taxonomy.json", "r"))
    taxonomy = find_taxonomy_level(taxonomy, taxonomy_level)

    prompt = open(f"{path}/main_flat_prompt.txt", "r").read()
    prompt = multiple_replace(
        prompt,
        {
            "image": input_data.image_placeholder,
            "post": input_data.post,
            "note": input_data.note,
            "taxonomy": json.dumps(taxonomy, indent=2),
        },
    )
    system_prompt = prompt[
        prompt.find("<SYSTEM_PROMPT>")
        + len("<SYSTEM_PROMPT>") : prompt.find("</SYSTEM_PROMPT>")
    ]
    user_prompt = prompt[
        prompt.find("<USER_PROMPT>")
        + len("<USER_PROMPT>") : prompt.find("</USER_PROMPT>")
    ]
    return {
        "system_prompt": system_prompt.strip(),
        "user_prompt": user_prompt.strip(),
        "taxonomy": taxonomy,
    }


def load_images(image_paths: list[str], enable_smart_resize=False) -> list[Image.Image]:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if enable_smart_resize:

        def post_process_image(image: Image) -> Image:
            width, height = image.size
            resized_height, resized_width = smart_resize(
                height, width, max_pixels=2048 * 28 * 28
            )
            return image.resize((resized_width, resized_height))

        images = [post_process_image(image) for image in images]
    return images


def parse_generations(inputs, outputs):
    """Parse the model outputs into a DataFrame."""
    json_outputs = []
    for input, output in zip(inputs, outputs):
        try:
            generated_text = output.outputs[0].text
            json_output = json.loads(generated_text)
            json_output["image_path"] = input.image_path
            json_outputs.append(json_output)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for image {input.image_path}: {e}")
    df = pd.DataFrame(json_outputs)
    return df


def strip_text(text: str) -> str:
    """Strip leading and trailing whitespace from the text."""
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    return text


def process_outputs(outputs: List, inputs: List[InputData]) -> List[Dict[str, Any]]:
    """Process the outputs from the model into a list of dictionaries."""
    processed_outputs = []
    for output, input_data in zip(outputs, inputs):
        try:
            generated_text = output.outputs[0].text
            generated_text = strip_text(generated_text)
            json_output = json.loads(generated_text)
            output_data = {
                "image_path": input_data.image_path,
                "taxonomy_level": input_data.taxonomy_level,
                "labels": json_output.get("labels", []),
                "note": input_data.note,
                "post": input_data.post,
            }
            processed_outputs.append(output_data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for image {input_data.image_path}: {e}")
            processed_outputs.append(
                {
                    "image_path": input_data.image_path,
                    "taxonomy_level": input_data.taxonomy_level,
                    "json_output": None,
                    "note": input_data.note,
                    "post": input_data.post,
                }
            )
    return processed_outputs
