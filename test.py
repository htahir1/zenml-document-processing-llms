from byaldi import RAGMultiModalModel
from rich import print
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from pdf2image import convert_from_path


def main():
    images = convert_from_path("data/microsoft-q2-2025-small.pdf")

    RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")
    RAG.index(
        input_path="data/microsoft-q2-2025-small.pdf",
        index_name="image_index",  # index will be saved at index_root/index_name/
        store_collection_with_index=False,
        overwrite=True,
    )
    text_query = "What was Microsoft's net income in Q2 2025?"
    results = RAG.search(text_query, k=1)

    # reset the CUDA cache
    torch.cuda.empty_cache()

    model = (
        Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        .cuda()
        .eval()
    )

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True
    )

    image_index = results[0]["page_num"] - 1
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": images[image_index],
                },
                {"type": "text", "text": text_query},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_text)


if __name__ == "__main__":
    main()
