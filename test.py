from byaldi import RAGMultiModalModel
from rich import print
from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch
from pdf2image import convert_from_path


def main():
    images = convert_from_path("data/microsoft-q2-2025-full.pdf")

    RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")
    RAG.index(
        input_path="data/microsoft-q2-2025-full.pdf",
        index_name="image_index_full",  # index will be saved at index_root/index_name/
        store_collection_with_index=False,
        overwrite=True,
    )
    text_query = "How is Microsoft doing, financially?"
    results = RAG.search(text_query, k=3)

    model = MllamaForConditionalGeneration.from_pretrained(
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(
        "meta-llama/Llama-3.2-11B-Vision-Instruct"
    )

    image_index = results[0]["page_num"] - 1
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {"type": "text", "text": text_query},
            ],
        }
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        images=images[image_index],
        text=input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=500)
    output_text = processor.decode(output[0])

    output_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)
    ]

    output_text = processor.batch_decode(
        output_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(output_text[0])


if __name__ == "__main__":
    main()
