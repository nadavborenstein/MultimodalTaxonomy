import yaml
import pandas as pd
from typing import List, Dict
from PIL import Image
from input_output_utils import load_images
from glob import glob
import random


class Demonstrations(object):
    
    def __init__(self, args):
        self.args = args
        self.num_demos: int = args.num_demos
        self.demonstration_type: str = args.demonstration_type
        self.demonstrations: pd.DataFrame = self._load_demonstrations(args.demo_data_path)
        self.images: Dict[str, Image.Image] = self._load_images(args.demo_images_path)
        self.image_placeholder: str = args.image_placeholder
        self.seed = args.seed
        
        assert self.num_demos <= self.demonstrations.shape[0], "Number of demonstrations exceeds available demonstrations."
        
    def _load_demonstrations(self, demo_data_path: str) -> pd.DataFrame:
        """
        Load demonstration data from a CSV file.
        """
        demos = pd.read_csv(demo_data_path)
        return demos
        
    def _load_images(self, images_path: str) -> Dict[str, Image.Image]:
        """
        Load images from the specified directory.
        """
        image_paths = glob(f"{images_path}/*.jpg") + glob(f"{images_path}/*.png") + glob(f"{images_path}/*.jpeg")
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
            
    def _select_question_annotations(self, decoded_annotations: Dict[str, Dict[str, str]],
                                     question: str = None,
                                     question_flow: List[str] = None) -> Dict[str, str]:
        """
        Select question annotations based on the demonstration type.
        """
        if self.demonstration_type.startswith("random"):
            if "_" in self.demonstration_type:
                n = int(self.demonstration_type.split("_")[-1])
            else:
                n = 1
            selected_questions = random.sample(list(decoded_annotations.keys()), k=n)
            selected_annotations = {q: decoded_annotations[q] for q in selected_questions}
        elif self.demonstration_type == "same_single":
            assert question is not None, "Question must be provided for same_single demonstration type."
            try:
                selected_annotations = {question: decoded_annotations[question]}
            except KeyError:
                raise KeyError("the provided question does not appear in the taxonomy. Please check for typos.")
        elif self.demonstration_type == "same_flow":
            assert question_flow is not None, "question_flow must be provided for same_flow demonstration type."
            selected_annotations = {q: decoded_annotations[q] for q in question_flow if q in decoded_annotations}
        elif self.demonstration_type == "all":
            selected_annotations = decoded_annotations
        else:
            raise ValueError(f"Unknown demonstration type: {self.demonstration_type}. Choose from 'random', 'same_single', 'same_flow', or 'all'.")
        return selected_annotations
        
    def get_demonstrations(self, question: str = None, question_flow: List[str] = None) -> List[Dict]:
        """
        Gets demonstrations based on the selected demonstration type. If semonstratioin type is "same_single",
        "question" should be provided. If it is "same_flow", "question_flow" must be provided.
        """
        instances = self.demonstrations.sample(n=self.num_demos, random_state=self.seed)
        annoators = [c for c in instances.columns.tolist() if "annotator" in c]
        persona = random.choices(annoators, k=1)[0]
        instances = instances[["tweet_id", "note", "tweet", "user_name", "image_name"] + [annoators[persona]]]
        instances["decoded_annotations"] = instances[persona].apply(self._decode_annotations)
        
        demonstrations = []
        for _, instance in instances.iterrows():
            demo = {
                "post": instance["tweet"],
                "note": instance["note"],
                "user_name": instance["user_name"],
                "image": self.images.get(instance["image_name"], None),
                "annotations": self._select_question_annotations(instance["decoded_annotations"], question, question_flow)
            }
            demonstrations.append(demo)
        return demonstrations
    
    def embed_demonstrations_in_prompt(self, demonstrations) -> str:
        demo_texts = []
        for demo in demonstrations:
            demo_text = f"Image: {self.image_placeholder}\nPost: {demo['post']}\nNote: {demo['note']}\nUser: {demo['user_name']}\n"
            for question, annotation in demo["annotations"].items():
                demo_text += f"Question: {question}\nAnswer: {annotation['Answer']}\nReasoning: {annotation['Reasoning']}\n"
            demo_texts.append(demo_text)
        demos_combined = "\n---\n".join(demo_texts)
        return demos_combined
            
    