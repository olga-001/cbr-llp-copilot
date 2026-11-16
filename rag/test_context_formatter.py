from pprint import pprint

from rag.context_formatter import prepare_context_package
from rag.retrieve_all import retrieve_all
from rag.selection_utils import select_context_items


def main():
    question = "В случае если у заемщика несколько ссуд как определяется величина резерва по вновь выданной ссуде?"

    retrieval_result = retrieve_all(question)

    selection = select_context_items(
        retrieval_result,
        primary_k=3,
        secondary_k=2,
        tertiary_k=0,
    )

    context_package = prepare_context_package(selection["selected_items"])

    print("=" * 100)
    print("CONTEXT BLOCKS:")
    print(context_package["context_blocks"][:4000])

    print("\n" + "=" * 100)
    print("CITATION CANDIDATES:")
    pprint(context_package["citation_candidates"][:3])

    print("\n" + "=" * 100)
    print("DOCUMENT COUNT:")
    print(len(context_package["documents"]))


if __name__ == "__main__":
    main()