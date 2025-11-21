from dataclasses import dataclass
from typing import List, Dict, Any
from PIL import Image
from vllm.sampling_params import GuidedDecodingParams
from argparse import Namespace
from input_output_utils import multiple_replace, strip_output_text, post_process_image
import json
from structured_outputs import Labels, make_label_model


@dataclass
class MultimodalNote:
    image_path: str
    note: str
    image: Image.Image = None  # Placeholder for the image object
    post: str = ""
    image_url: str = ""

    def __str__(self):
        return f"InputData(image_path={self.image_path}, post={self.post}, note={self.note}, taxonomy_level={self.taxonomy_level})"


class MultimodalInteractorBase(object):

    def initialize(self) -> None:
        raise NotImplementedError

    def select_schema(self) -> GuidedDecodingParams:
        raise NotImplementedError

    def process_output(self, output: str) -> List[str]:
        raise NotImplementedError

    def _load_image(self, image_path: str, enable_smart_resize) -> list[Image.Image]:
        image = Image.open(image_path).convert("RGB")
        if enable_smart_resize:
            image = post_process_image(image)
        return image

    def update_image(
        self,
        image: Image.Image = None,
        image_path: str = None,
        enable_smart_resize: bool = True,
    ) -> None:
        assert (
            image is not None or image_path is not None
        ), "Either image or image_path must be provided."
        if image is None:
            image = self._load_image(image_path, enable_smart_resize)
            self.note.image_path = image_path
        self.note.image = image

    def load_image(self, enable_smart_resize: bool = True) -> None:
        if self.note.image is None and self.note.image_path:
            self.note.image = self._load_image(
                self.note.image_path, enable_smart_resize
            )
        return self.note.image

    def remove_image(self) -> None:
        self.note.image.close()
        self.note.image = None

    def get_output_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.note.image_path,
            "image_url": self.note.image_url,
            "labels": self.labels,
            "note": self.note.note,
            "post": self.note.post,
        }


class MultimodalQAInteractor(MultimodalInteractorBase):

    def __init__(
        self,
        note: MultimodalNote,
        image_substring_marker: str = "<|image_1|>",
        args: Namespace = None,
    ):
        self.note = note
        self.args = args
        self.image_placeholder = image_substring_marker

        self.user_prompt = None
        self.system_prompt = None
        self.messages = []
        self.current_question = None
        self.question_stack = []
        self.labels = []  # List to store selected labels
        self._binarize_question_flag = True  # Flag to control binarization of questions

        self.initialize()

    def initialize(self) -> None:
        prompt_path = self.args.prompt_path
        taxonomy_path = self.args.taxonomy_path

        taxonomy = json.load(open(taxonomy_path, "r"))
        prompt = open(prompt_path, "r").read()
        system_prompt, user_prompt = self._parse_prompt(prompt)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            },
        ]

        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.messages = messages

        self.question_stack = [taxonomy["questions"]]
        self.current_question = self.question_stack[-1][-1]

        self._update_messages("I am ready to assist you. What would you like to know?")

    def _get_taxonomy_root_key(self) -> str:
        """Get the root key of the taxonomy."""
        if self.taxonomy:
            return next(iter(self.taxonomy.keys()))
        return None

    def _roll_out_stack(self) -> None:
        while self.question_stack and len(self.question_stack[-1]) == 1:
            self.question_stack.pop()
            self.messages.pop()
            self.messages.pop()

        if self.question_stack:
            # removing the right-most branch from the question stack and messages
            self.question_stack[-1].pop()
            self.messages.pop()

    def _is_label_node(self, answer_root: Dict) -> bool:
        """Check if the output is a label node."""
        return "label" in answer_root.keys()

    def _get_answers_roots(self, output: List[str]) -> List[Dict]:
        answers = []
        answers_key = (
            "multi_answers"
            if "multi_answers" in self.current_question.keys()
            else "answers"
        )
        for ans in self.current_question[answers_key]:
            if ans["text"].strip() in output:
                answers.append(ans)

        if not answers:
            raise ValueError(
                f"Model output '{output}' does not match any answers in the current question: {self.current_question}"
            )

        return answers

    def _get_question_roots(self, answers: List[Dict]) -> List[Dict]:
        questions = []
        for ans in answers:
            if "questions" in ans.keys():
                questions.extend(ans["questions"])
            else:
                raise ValueError(f"Answer does not contain 'questions': {ans}")
        return questions

    def _validate_answers(self, output: List[str]) -> bool:
        """Validate the model's output against the current question's answers."""
        if "multi_answers" in self.current_question.keys():
            answers = self.current_question["multi_answers"]
            text_answers = {ans["text"].strip() for ans in answers}
            assert text_answers.intersection(
                output
            ), f"Model output '{output}' does not match any answers in the current question: {self.current_question}"
        elif "answers" in self.current_question.keys():
            assert len(output) == 1, "Model output should contain exactly one answer."
            answers = self.current_question["answers"]
            text_answers = {ans["text"].strip() for ans in answers}
            assert (
                output[0] in text_answers
            ), f"Model output '{output}' does not match any answers in the current question: {self.current_question}"
        else:
            raise ValueError(
                f"Current question does not contain 'answers' or 'multi_answers': {self.current_question}"
            )
        return True

    def _update_messages(self, answer: str = None) -> None:
        if answer is None:
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._construct_question_promot()}
                ],
            }
            self.messages.append(message)
        else:
            message_1 = {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            }
            message_2 = {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._construct_question_promot()}
                ],
            }
            self.messages.append(message_1)
            self.messages.append(message_2)

    def _add_labels(self, labels: List[str]) -> None:
        for label in labels:
            if label not in self.labels:
                self.labels.append(label)

    def _binarize_question(self, question: Dict) -> List[Dict]:
        possible_answers = question["multi_answers"]
        binarised_questions = []
        for answer in possible_answers:
            binary_answer = (
                answer["description"] if "description" in answer else answer["text"]
            )
            binary_question = (
                f"{question['text']} Is the answer '{binary_answer}' correct?"
            )
            negative_answer = {"text": "no", "label": "None"}
            if "label" in answer.keys():
                positive_answer = {"text": "yes", "label": answer["label"]}
            else:
                positive_answer = {"text": "yes", "questions": answer["questions"]}
            binarised_questions.append(
                {"text": binary_question, "answers": [positive_answer, negative_answer]}
            )
        return binarised_questions

    def _binarize_questions(self, questions: List[Dict]) -> List[Dict]:
        """
        Converts a multi-option question into a set of binary questions.
        """
        binarized_questions = []
        for question in questions:
            if "multi_answers" in question.keys():
                # If the question has multiple answers, create binary questions for each answer
                binarized_questions.extend(self._binarize_question(question))
            else:
                # If the question has a single answer, keep it as is
                binarized_questions.append(question)
        return binarized_questions

    def _get_answer_options(self, question) -> List[str]:
        key = "multi_answers" if "multi_answers" in question.keys() else "answers"
        options = [ans["text"].strip() for ans in question[key]]
        options = sorted(options, key=lambda x: x.lower())
        return options

    def select_schema(self) -> GuidedDecodingParams:
        """
        Selects a schema for guided decoding based on the current question.
        Returns a GuidedDecodingParams object with the options for the next question.
        """
        # multioption = "multi_answers" in self.current_question.keys()
        options = self._get_answer_options(self.current_question)
        guided_decoding_params = GuidedDecodingParams(choice=options)
        return guided_decoding_params

    def _construct_question_promot(self) -> str:
        question = self.current_question["text"]
        possible_answers = self._get_answer_options(self.current_question)
        prompt = f"{question}\nPossible answers: {', '.join(possible_answers)}\nPlease select the correct answer."
        return prompt

    def _parse_prompt(self, prompt: str) -> None:
        prompt = multiple_replace(
            prompt,
            {
                "image": self.image_placeholder,
                "post": self.note.post,
                "note": self.note.note,
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
        system_prompt = system_prompt.strip()
        user_prompt = user_prompt.strip()

        return system_prompt, user_prompt
    
    def update_next_turn(self, model_answer: List[str]) -> Dict:
        """Processes the model's answer and updates the question stack and messages.
        If the answer is a label node, it adds the label to the labels list.
        If the answer is a question node, it updates the question stack with the new questions.
        The method supports multi_answers nodes, but will binarize such questions as a default behavior.
        """
        self._validate_answers(model_answer)

        answer_roots: List[Dict] = self._get_answers_roots(model_answer)
        not_label_nodes = []
        for answer_root in answer_roots:
            if self._is_label_node(answer_root):
                self._add_labels([answer_root["label"]])
            else:
                not_label_nodes.append(answer_root)

        if not_label_nodes:
            question_roots = self._get_question_roots(not_label_nodes)
            if self._binarize_question_flag:
                question_roots = self._binarize_questions(question_roots)
            self.question_stack.append(question_roots)
            self.current_question = self.question_stack[-1][-1]
            self._update_messages(", ".join(model_answer))
        else:
            self._roll_out_stack()
            if not self.question_stack:
                raise StopIteration(
                    "No more questions in the stack. The interaction is complete."
                )
            self.current_question = self.question_stack[-1][-1]
            self._update_messages()

    def process_output(self, output: str) -> List[str]:
        try:
            generated_text = output.outputs[0].text
            generated_text = strip_output_text(generated_text)
            json_output = json.loads(generated_text)
            answers = json_output.get("answers")
            if type(answers) is not list:
                answers = [answers]
            return answers
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for image {self.note.image_path}: {e}")
            return []

    def process_outputs_deprecated(self, output: List) -> List[Dict[str, Any]]:
        """Process the outputs from the model into a list of dictionaries."""
        try:
            generated_text = output.outputs[0].text
            generated_text = strip_output_text(generated_text)
            json_output = json.loads(generated_text)
            output_data = {
                "image_path": self.note.image_path,
                "taxonomy_level": self._get_taxonomy_root_key(),
                "output": json_output,
                "note": self.note.note,
                "post": self.note.post,
            }

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for image {self.note.image_path}: {e}")
            output_data.append(
                {
                    "image_path": self.note.image_path,
                    "taxonomy_level": self._get_taxonomy_root_key(),
                    "json_output": None,
                    "note": self.note.note,
                    "post": self.note.post,
                }
            )
        return output_data


class MultimodalTaxonomyInteractor(MultimodalInteractorBase):
    def __init__(
        self,
        note: MultimodalNote,
        image_substring_marker: str = "<|image_1|>",
        args: Namespace = None,
    ):
        self.note = note
        self.args = args
        self.image_placeholder = image_substring_marker

        self.user_prompt = None
        self.system_prompt = None
        self.messages = []
        self.labels = []  # List to store selected labels
        self.prompt = None
        self.full_taxonomy = None
        self.taxonomy = None

        self.initialize()

    def _get_possible_taxonomy_levels(self, taxonomy: Dict[str, Any]) -> List[str]:
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

    def _get_taxonomy_labels(self, taxonomy: Dict[str, Any]) -> List[str]:
        """Get the labels from the taxonomy dictionary."""
        labels = []

        def recursive_label_extractor(d: Dict):
            for key, value in d.items():
                if key == "labels":
                    for value in d["labels"]:
                        labels.append(value)
                elif isinstance(value, dict):
                    recursive_label_extractor(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            recursive_label_extractor(item)

        recursive_label_extractor(taxonomy)
        return list(set(labels))

    def _find_taxonomy_level(
        self, taxonomy: Dict[str, Any], taxonomy_level: str
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

        possible_taxonomy_levels = self._get_possible_taxonomy_levels(taxonomy)
        if type(taxonomy_level) is list:
            taxonomy_level = taxonomy_level[0]
        assert (
            taxonomy_level in possible_taxonomy_levels
        ), f"Invalid taxonomy level: {taxonomy_level}. Possible levels are: {['head'] + possible_taxonomy_levels}"

        if taxonomy_level == "head":
            return taxonomy
        else:
            return recursive_find_key(taxonomy, taxonomy_level)

    def _parse_prompt(self, prompt: str, taxonomy: Dict) -> None:
        prompt = multiple_replace(
            prompt,
            {
                "image": self.image_placeholder,
                "post": self.note.post,
                "note": self.note.note,
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
        system_prompt = system_prompt.strip()
        user_prompt = user_prompt.strip()

        return system_prompt, user_prompt

    def _update_messages(self, taxonomy: Dict) -> None:
        system_prompt, user_prompt = self._parse_prompt(self.prompt, taxonomy)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            },
        ]

        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.messages = messages

    def initialize(self) -> None:
        prompt_path = self.args.prompt_path
        taxonomy_path = self.args.taxonomy_path

        self.prompt = open(prompt_path, "r").read()
        self.full_taxonomy = json.load(open(taxonomy_path, "r"))
        self.taxonomy = self._find_taxonomy_level(
            self.full_taxonomy, self.args.taxonomy_level
        )

        self._update_messages(self.taxonomy)

    def update_taxonomy(
        self, taxonomy: Dict[str, Any] = None, level: str = None
    ) -> None:
        """
        Update the taxonomy used in the interactor.
        This method can be called to change the taxonomy dynamically.
        """
        assert (
            taxonomy is not None or level is not None
        ), "Either taxonomy or level must be provided."

        if taxonomy:
            self.full_taxonomy = taxonomy

        if level:
            self.taxonomy = self._find_taxonomy_level(self.full_taxonomy, level)
        else:
            self.taxonomy = self.full_taxonomy

        self._update_messages(self.taxonomy)

    def process_output(self, output: str) -> List[str]:
        try:
            generated_text = output.outputs[0].text
            generated_text = strip_output_text(generated_text)
            json_output = json.loads(generated_text)
            answers = json_output.get("labels")
            self.labels.extend(answers)
            return answers
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for image {self.note.image_path}: {e}")
            return []

    def get_guided_decoding_params(self) -> GuidedDecodingParams:
        """
        Selects a schema for guided decoding based on the current question.
        Returns a GuidedDecodingParams object with the options for the next question.
        """
        options = self._get_taxonomy_labels(self.taxonomy)
        schema = make_label_model(options=options + ["None"])
        guided_decoding_params = GuidedDecodingParams(json=schema.model_json_schema())
        return guided_decoding_params
