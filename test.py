from pdf2image import convert_from_path
from byaldi import RAGMultiModalModel


def main():
    images = convert_from_path("data/microsoft-q2-2025-small.pdf")
    images[2].save("microsoft-q2-2025-small_page_2.png")

    RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")
    RAG.index(
        input_path="data/microsoft-q2-2025-small_page_2.png",
        index_name="image_index",  # index will be saved at index_root/index_name/
        store_collection_with_index=False,
        overwrite=True,
    )


if __name__ == "__main__":
    main()
