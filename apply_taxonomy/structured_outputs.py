from pydantic import BaseModel, Field
from typing import NamedTuple, Optional, Literal, List


class ImageDescriptor(BaseModel):
    """
    Represents an image that was shared on Twitter.
    """

    description: str = Field(
        ...,
        description="A short description of the image. This should be a string that describes the image in a few words. For example, if the image is a photo of a dog, this could be 'A photo of a dog'.",
    )
    type_explanation: str = Field(
        ..., description="A short explanation of why 'image_type' was selected."
    )
    image_type: List[str] = Field(
        ...,
        description="The type of the image. Possible values are 'simple photo', 'complex image', 'screenshot', 'document', 'computer graphic', 'just text', 'infographics', 'other'. More than one value can be selected for each image.",
    )


class BinaryLabel(BaseModel):
    """
    Represents a binary label for an image.
    """
    resoning: str = Field(..., description="A short explanation of why the label was selected.",)
    label: bool = Field(..., description="A binary label for the image. Possible values are True or False.",)


class ImageDescriptorBinary(BaseModel):
    """
    Represents an image that was shared on Twitter.
    """

    description: str = Field(..., description="A short description of the image. This should be a string that describes the image in a few words. For example, if the image is a photo of a dog, this could be 'A photo of a dog'.",)
    simple_photo: BinaryLabel = Field(..., description="A binary label for the image indicating if it is a simple photo.")
    complex_image: BinaryLabel = Field(..., description="A binary label for the image indicating if it is a complex image.")
    screenshot: BinaryLabel = Field(..., description="A binary label for the image indicating if it is a screenshot.")
    document: BinaryLabel = Field(..., description="A binary label for the image indicating if it is a document.")
    computer_graphic: BinaryLabel = Field(..., description="A binary label for the image indicating if it is a computer graphic.")
    just_text: BinaryLabel = Field(..., description="A binary label for the image indicating if it is just text.")
    infographics: BinaryLabel = Field(..., description="A binary label for the image indicating if it is infographics.")
    other: BinaryLabel = Field(..., description="A binary label for the image indicating if it is other.")


class Images(BaseModel):
    """
    Represents a list of images that were shared on Twitter.
    """

    images: list[ImageDescriptor] = Field(
        ..., description="A list of images that were shared on Twitter."
    )