from pydantic import BaseModel, Field, create_model
from typing import NamedTuple, Optional, Literal, List, Set


class Label(BaseModel):
    """
    Represents an a selected label from a taxonomy.
    """

    reasoning: str = Field(
        ...,
        description="The reason why the label was selected.",
    )
    label: Literal["Option_1", "Option_2"] = Field(
        ..., description="The name of the label."
    )


class Labels(BaseModel):
    """
    Represents a binary label for an image.
    """

    labels: List[Label] = Field(
        ...,
        description="A list of labels that were selected for the <post, image, fact-check verdict> triplet.",
    )


# def make_label_model(options: List[str]):
#     # Label = create_model(
#     #     "Label",
#     #     label=(
#     #         Literal[tuple(options)],
#     #         Field(..., description="The name of the label."),
#     #     ),
#     # )
#     Labels = create_model(
#         "Labels",
#         labels=(
#             Set[Literal[tuple(options)]],
#             Field(
#                 ...,
#                 description="A list of labels that were selected for the <post, image, fact-check verdict> triplet.",
#             ),
#         ),
#     )
#     return Labels


def make_label_model(options: List[str]):
    Label = create_model(
        "Label",
        label=(
            Literal[tuple(options)],
            Field(..., description="The name of the label."),
        ),
        reasoning=(
            str,
            Field(..., description="The reason why the label was selected in a single sentence."),
        ),
    )
    Labels = create_model(
        "Labels",
        labels=(
            List[Label],
            Field(
                ...,
                description="A list of labels that were selected for the <post, image, fact-check verdict> triplet.",
            ),
        ),
    )
    return Labels
