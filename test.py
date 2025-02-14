import base64
import io
import os

from huggingface_hub import HfApi
from byaldi import RAGMultiModalModel
from pdf2image import convert_from_path
from rich import print
from openai import OpenAI

import logging

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

ENDPOINT_NAME = "llama-3-2-11b-vision-instruc-egg"


def encode_image_to_base64(image) -> str:
    """
    Encode a PIL Image to base64.

    Args:
        image: PIL Image object

    Returns:
        str: Base64 encoded string of the image
    """
    # Create a buffer to store the image
    buffer = io.BytesIO()
    # Save the image as JPEG to the buffer with reduced quality
    image.save(buffer, format="JPEG", quality=10)
    # Get the bytes from the buffer
    image_bytes = buffer.getvalue()
    # Encode to base64
    return base64.b64encode(image_bytes).decode("utf-8")


def main() -> None:
    """
    Main function to perform multi-modal inference via a Hugging Face Inference Endpoint using OpenAI API format.

    Steps:
      - Convert a PDF into images
      - Index the PDF using a RAG multi-modal model and search by a text query
      - Convert the selected image to base64
      - Use OpenAI client to interact with the Hugging Face endpoint
      - Stream and print the response
    """
    # Convert PDF pages to images
    images = convert_from_path("data/microsoft-q2-2025-small.pdf")

    # Initialize and index the PDF with the RAG multi-modal model
    RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")
    RAG.index(
        input_path="data/microsoft-q2-2025-small.pdf",
        index_name="image_index_small",  # index will be saved at index_root/index_name/
        store_collection_with_index=False,
        overwrite=True,
    )

    # Define the text query and search the PDF index
    text_query = "How is Microsoft doing, financially?"
    results = RAG.search(text_query, k=3)

    # Convert the selected image to JPEG bytes after further resizing and compression
    selected_image = images[results[0]["page_num"] - 1]

    # Create a resized copy to reduce file size
    resized_image = selected_image.copy()
    resized_image.thumbnail((64, 64))  # Reduce resolution

    # Convert image to base64
    base64_image = encode_image_to_base64(resized_image)

    hf_api = HfApi()
    endpoint = hf_api.get_inference_endpoint(ENDPOINT_NAME, namespace="zenml")
    base_url = endpoint.url
    # Initialize OpenAI client with Hugging Face endpoint
    client = OpenAI(
        base_url=base_url + "/v1",
        api_key=os.environ.get("HF_TOKEN"),
    )

    # Prepare the chat completion request
    chat_completion = client.chat.completions.create(
        model="tgi",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": text_query},
                ],
            }
        ],
        max_tokens=200,
    )
    analysis_response = chat_completion.choices[0].message.content
    print(analysis_response)


if __name__ == "__main__":
    main()
