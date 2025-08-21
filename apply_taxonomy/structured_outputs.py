from pydantic import BaseModel, Field
from typing import NamedTuple, Optional, Literal, List


class Label(BaseModel):
    """
    Represents an a selected label from a taxonomy.
    """

    reasoning: str = Field(
        ...,
        description="The reason why the label was selected.",
    )
    label: str = Field(..., description="The name of the label.")


class Labels(BaseModel):
    """
    Represents a binary label for an image.
    """

    labels: List[Label] = Field(
        ...,
        description="A list of labels that were selected for the <post, image, fact-check verdict> triplet.",
    )
