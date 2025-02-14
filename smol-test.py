from byaldi import RAGMultiModalModel
from rich import print
import torch
from pdf2image import convert_from_path

from transformers import Idefics3ForConditionalGeneration, AutoProcessor


def main():
    all_images = convert_from_path("data/microsoft-q2-2025-full.pdf")

    docs_retrieval_model = RAGMultiModalModel.from_pretrained("vidore/colsmolvlm-alpha")
    docs_retrieval_model.index(
        input_path="data/microsoft-q2-2025-full.pdf",
        index_name="image_index_full",
        store_collection_with_index=False,
        overwrite=True,
    )
    text_query = "How is Microsoft doing, financially?"
    results = docs_retrieval_model.search(text_query, k=3)
    result_images = [all_images[result["doc_id"]] for result in results]

    model_id = "HuggingFaceTB/SmolVLM-Instruct"
    vl_model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",
    )
    vl_model.eval()

    vl_model_processor = AutoProcessor.from_pretrained(model_id)

    chat_template = [
        {
            "role": "user",
            "content": [
                *[{"type": "image"} for _ in range(len(result_images))],
                {"type": "text", "text": text_query},
            ],
        }
    ]

    text = vl_model_processor.apply_chat_template(
        chat_template, add_generation_prompt=True
    )

    inputs = vl_model_processor(
        text=text,
        images=result_images,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = vl_model.generate(**inputs, max_new_tokens=2500)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = vl_model_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print(output_text[0])

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
