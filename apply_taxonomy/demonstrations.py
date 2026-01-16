import yaml
import pandas as pd
from typing import List, Dict, Union
from PIL import Image
from .input_output_utils import load_images, multiple_replace, single_turn_message
from glob import glob
import random


PROMPT = """
User: {{user}}
Image: {{image}}
Post: {{post}}
Note: {{note}}
"""


class Demonstrations(object):

    def __init__(self, args, taxonomy: Dict, image_placeholder: str = "<|image_1|>"):
        self.args = args
        self.taxonomy = taxonomy
        self.num_demos: int = args.num_demos
        self.only_with_explanations: bool = True
        self.demonstrations: pd.DataFrame = self._load_demonstrations(
            args.demo_data_path
        )
        self.images: Dict[str, Image.Image] = self._load_images(args.demo_images_path)
        self.image_placeholder: str = image_placeholder
        self.seed = args.seed
        random.seed(self.seed)
        self.question_to_explanation = self._load_question_explanations()
        random.shuffle(self.personas)

        assert (
            self.num_demos <= self.demonstrations.shape[0]
        ), "Number of demonstrations exceeds available demonstrations."

    def _load_question_explanations(self) -> Dict[str, str]:
        """
        Construct a dictionary mapping questions to their explanations from the taxonomy.
        """
        question_to_explanation = {}

        def dfs(node: Dict):
            if "question" in node:
                question_to_explanation[node["question"]] = node.get("explanation", "")
            if "questions" in node:
                for child in node["questions"]:
                    dfs(child)
            if "answers" in node:
                for answer in node["answers"]:
                    dfs(node["answers"][answer])

        dfs(self.taxonomy)
        return question_to_explanation

    def _load_demonstrations(self, demo_data_path: str) -> pd.DataFrame:
        """
        Load demonstration data from a CSV file.
        """
        demos = pd.read_csv(demo_data_path)
        self.personas = [c for c in demos.columns.tolist() if "annotator" in c]
        for persona in self.personas:
            demos[persona] = demos[persona].apply(self._decode_annotations)
        return demos

    def _load_images(self, images_path: str) -> Dict[str, Image.Image]:
        """
        Load images from the specified directory.
        """
        image_paths = (
            glob(f"{images_path}/*.jpg")
            + glob(f"{images_path}/*.png")
            + glob(f"{images_path}/*.jpeg")
        )
        images = load_images(image_paths)
        images = {path.split("/")[-1]: img for path, img in zip(image_paths, images)}
        return images

    def _decode_annotations(self, annotations: str) -> Dict[str, Dict[str, str]]:
        """
        Decode the annotations from a string representation to a dictionary.
        """
        decoded_annotations = {}
        annotations = eval(annotations)
        for labels in annotations:
            decoded_annotations[labels["Question"]] = labels

        return decoded_annotations

    def _select_question_annotations(
        self,
        dem_type: str,
        decoded_annotations: Dict[str, Dict[str, str]],
        question: str = None,
        question_flow: List[str] = None,
    ) -> Dict[str, str]:
        """
        Select question annotations based on the demonstration type.
        """

        def get_question_pool():
            pool = []
            for q in decoded_annotations.keys():
                if decoded_annotations[q]["Answer"] and (
                    decoded_annotations[q]["Answer"] != "I don't know"
                ):
                    if (
                        "Explanation" in decoded_annotations[q]
                        and decoded_annotations[q]["Explanation"].strip() != ""
                    ) or not self.only_with_explanations:
                        pool.append(q)
            return pool

        def eval_same_question(q, decoded_annotations):
            if q not in decoded_annotations:
                raise KeyError(
                    "the provided question does not appear in the taxonomy. Please check for typos."
                )
            if not decoded_annotations[q]["Answer"]:
                return False
            if decoded_annotations[q]["Answer"] == "I don't know":
                return False
            if self.only_with_explanations:
                if "Explanation" not in decoded_annotations[q]:
                    return False
                if decoded_annotations[q]["Explanation"].strip() == "":
                    return False
            return True

        if dem_type.startswith("random"):
            n = int(dem_type.split("_")[-1]) if "_" in dem_type else 1
            pool = get_question_pool()
            selected_questions = random.sample(pool, k=min(n, len(pool)))
            selected_annotations = {
                q: decoded_annotations[q] for q in selected_questions
            }
        elif dem_type == "same_single":
            if question is None:
                raise ValueError(
                    "Question must be provided for 'same_single' demonstration type."
                )
            if not eval_same_question(question, decoded_annotations):
                return None
            selected_annotations = {question: decoded_annotations[question]}
        elif dem_type == "same_flow":
            if question_flow is None:
                raise ValueError(
                    "Question flow must be provided for 'same_flow' demonstration type."
                )
            if not eval_same_question(question_flow[-1], decoded_annotations):
                return None
            selected_annotations = {
                q: decoded_annotations[q]
                for q in question_flow
                if q in decoded_annotations
            }
        elif dem_type == "all":
            pool = get_question_pool()
            selected_annotations = {q: decoded_annotations[q] for q in pool}
        else:
            raise ValueError(
                f"Unknown demonstration type: {dem_type}. Choose from 'random_X', 'same_single', 'same_flow', or 'all'."
            )
        return selected_annotations

    def get_demonstrations(
        self,
        dem_type: str = "random",
        question: str = None,
        question_flow: List[str] = None,
    ) -> List[Dict]:
        """
        Gets demonstrations based on the selected demonstration type. If semonstratioin type is "same_single",
        "question" should be provided. If it is "same_flow", "question_flow" must be provided.
        """
        instances = self.demonstrations.sample(frac=1.0, random_state=self.seed)

        if dem_type != "random":
            backup_demonstrations = self.get_demonstrations(dem_type="random")
        demonstrations = []

        for _, instance in instances.iterrows():
            for persona in self.personas:
                annotations = self._select_question_annotations(
                    dem_type, instance[persona], question, question_flow
                )
                if annotations:
                    demo = {
                        "post": instance["tweet"],
                        "note": instance["note"],
                        "user_name": instance["user_name"],
                        "image": self.images.get(instance["image_name"], None),
                        "annotations": annotations,
                    }
                    demonstrations.append(demo)
                    break
            if len(demonstrations) >= self.num_demos:
                break
        if len(demonstrations) < self.num_demos and dem_type != "random":
            posts_in_demos = {d["post"] for d in demonstrations}
            backup_demonstrations = [
                demo
                for demo in backup_demonstrations
                if demo["post"] not in posts_in_demos
            ]
            demonstrations.extend(
                backup_demonstrations[: self.num_demos - len(demonstrations)]
            )
            random.shuffle(demonstrations)
        return demonstrations

    def embed_demonstrations_in_prompt(
        self, demonstrations, prompt: str = None
    ) -> Union[List[Dict], List[Image.Image]]:
        """
        Embed the demonstrations into the prompt or messages, which can be used for LLM input.
        """
        prompt = prompt if prompt else PROMPT
        messages = []
        images = []
        for demo in demonstrations:
            content = multiple_replace(
                prompt,
                {
                    "image": self.image_placeholder,
                    "post": demo["post"],
                    "note": demo["note"],
                    "user": demo["user_name"],
                },
            )
            demo_messages = single_turn_message(
                user_text=content,
                image=demo["image"],
                assistant_answer="What would you like to know?",
            )
            for question, annotation in demo["annotations"].items():
                reasoning = (
                    annotation["Explanation"] if "Explanation" in annotation else ""
                )
                question_explanation = self.question_to_explanation.get(question, "")

                if reasoning is not "":
                    json_llm_reply = (
                        '{"Anwser": "{{answer}}",\n "Reasoning": "{{explanation}}"}'
                    )
                else:
                    json_llm_reply = '{"Anwser": "<ANSWER>"}'
                json_llm_reply = multiple_replace(
                    json_llm_reply,
                    {"answer": annotation["Answer"], "explanation": reasoning},
                )
                demo_messages.extend(
                    single_turn_message(
                        user_text=f"Question: {question} {question_explanation}",
                        assistant_answer=json_llm_reply,
                        image=None,
                    )
                )
            messages.extend(demo_messages)
            images.append(demo["image"])
        return messages, images
