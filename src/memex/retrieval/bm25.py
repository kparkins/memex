"""BM25 fulltext search over the Neo4j revision index.

Provides query sanitization, fuzzy matching, and deprecated-item
filtering for the ``revision_search_text`` fulltext index created
by :func:`memex.stores.neo4j_schema.ensure_schema`.
"""

from __future__ import annotations

from neo4j import AsyncDriver
from pydantic import BaseModel

from memex.domain.models import ItemKind, Revision
from memex.stores.neo4j_schema import NodeLabel, RelType

_FULLTEXT_INDEX_NAME = "revision_search_text"

_LUCENE_ESCAPE_TABLE = str.maketrans({ch: f"\\{ch}" for ch in '+-&|!(){}[]^"~*?:\\/'})
_LUCENE_KEYWORDS: frozenset[str] = frozenset({"AND", "OR", "NOT"})


def sanitize_query(raw: str) -> str:
    """Escape Lucene special characters in a user-provided query.

    Prevents index injection by escaping all characters that carry
    special meaning in the Lucene classic query parser used by
    Neo4j fulltext indexes.

    Args:
        raw: Raw user-provided search string.

    Returns:
        Escaped query string safe for fulltext index execution.
    """
    return raw.translate(_LUCENE_ESCAPE_TABLE).strip()


def build_search_query(raw: str) -> str:
    """Build a sanitized fulltext query with fuzzy matching.

    Sanitizes the raw input, lowercases Lucene reserved keywords
    (``AND``, ``OR``, ``NOT``) to prevent operator interpretation,
    then appends ``~1`` (edit distance 1) to each term longer than
    2 characters per FR-7.

    Args:
        raw: Raw user-provided search string.

    Returns:
        Fulltext query string, or empty string if no valid terms
        remain after sanitization.
    """
    terms = raw.split()
    parts: list[str] = []
    for term in terms:
        escaped = sanitize_query(term)
        if not escaped:
            continue
        if escaped in _LUCENE_KEYWORDS:
            escaped = escaped.lower()
        if len(term) > 2:
            parts.append(f"{escaped}~1")
        else:
            parts.append(escaped)
    return " ".join(parts)


class BM25Result(BaseModel, frozen=True):
    """A single BM25 fulltext search result.

    Args:
        revision: The matched Revision domain model.
        score: BM25 relevance score from the fulltext index.
        item_id: ID of the owning Item.
        item_kind: Kind of the owning Item.
    """

    revision: Revision
    score: float
    item_id: str
    item_kind: ItemKind


async def bm25_search(
    driver: AsyncDriver,
    query: str,
    *,
    limit: int = 10,
    include_deprecated: bool = False,
    database: str = "neo4j",
) -> list[BM25Result]:
    """Execute a BM25 fulltext search over revision search_text.

    Sanitizes the query, applies fuzzy matching for terms longer than
    2 characters, queries the ``revision_search_text`` fulltext index,
    and filters deprecated items in Cypher before returning results.

    Args:
        driver: Async Neo4j driver instance.
        query: Raw user-provided search string.
        limit: Maximum number of results to return.
        include_deprecated: If True, include results from deprecated items.
        database: Neo4j database name.

    Returns:
        BM25Results ordered by descending score.
    """
    search_query = build_search_query(query)
    if not search_query:
        return []

    deprecation_filter = "" if include_deprecated else "WHERE i.deprecated = false "

    cypher = (
        f"CALL db.index.fulltext.queryNodes("
        f"'{_FULLTEXT_INDEX_NAME}', $q) "
        f"YIELD node AS r, score "
        f"MATCH (r)-[:{RelType.REVISION_OF}]->(i:{NodeLabel.ITEM}) "
        f"{deprecation_filter}"
        f"RETURN r, score, i.id AS item_id, i.kind AS item_kind "
        f"ORDER BY score DESC LIMIT $lim"
    )

    async with driver.session(database=database) as session:
        result = await session.run(cypher, q=search_query, lim=limit)
        return [
            BM25Result(
                revision=Revision.model_validate(dict(rec["r"])),
                score=float(rec["score"]),
                item_id=str(rec["item_id"]),
                item_kind=ItemKind(str(rec["item_kind"])),
            )
            async for rec in result
        ]
