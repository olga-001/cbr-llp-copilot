from __future__ import annotations

from typing import List, Dict


def get_item_identity(item: Dict) -> tuple:
    doc = item["document"]
    md = doc.metadata or {}

    return (
        item.get("source_doc"),
        md.get("page_num"),
        md.get("chunk_num"),
        md.get("row_id"),
        md.get("block_id"),
        (doc.page_content or "")[:200],
    )


def deduplicate_items(items: List[Dict]) -> List[Dict]:
    seen = set()
    result = []

    for item in items:
        key = get_item_identity(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)

    return result


def sort_items(items: List[Dict]) -> List[Dict]:
    return sorted(
        items,
        key=lambda x: (
            -x.get("rank_score", 0),
            0 if x.get("source_type") == "primary" else 1,
            (x["document"].metadata or {}).get("page_num", 9999),
            (x["document"].metadata or {}).get("chunk_num", 9999),
            (x["document"].metadata or {}).get("row_id", 999999),
            (x["document"].metadata or {}).get("block_id", 999999),
        )
    )


def merge_primary_secondary(
    primary_results: List[Dict],
    secondary_results: List[Dict],
    top_k_primary: int = 5,
    top_k_secondary: int = 5,
) -> Dict:
    primary = []
    for item in primary_results[:top_k_primary]:
        new_item = item.copy()
        new_item["source_type"] = "primary"
        primary.append(new_item)

    secondary = []
    for item in secondary_results[:top_k_secondary]:
        new_item = item.copy()
        new_item["source_type"] = "secondary"
        secondary.append(new_item)

    primary = deduplicate_items(primary)
    secondary = deduplicate_items(secondary)

    combined = deduplicate_items(primary + secondary)
    combined = sort_items(combined)

    return {
        "primary_results": primary,
        "secondary_results": secondary,
        "combined_results": combined,
    }


def merge_with_fallback(
    primary_results: List[Dict],
    secondary_results: List[Dict],
    top_k_primary: int = 5,
    top_k_secondary: int = 5,
) -> Dict:
    if primary_results:
        mode = "primary+secondary"
        merged = merge_primary_secondary(
            primary_results=primary_results,
            secondary_results=secondary_results,
            top_k_primary=top_k_primary,
            top_k_secondary=top_k_secondary,
        )
    else:
        mode = "secondary_only"
        secondary = []
        for item in secondary_results[:top_k_secondary]:
            new_item = item.copy()
            new_item["source_type"] = "secondary"
            secondary.append(new_item)

        secondary = deduplicate_items(secondary)
        secondary = sort_items(secondary)

        merged = {
            "primary_results": [],
            "secondary_results": secondary,
            "combined_results": secondary,
        }

    merged["mode"] = mode
    return merged