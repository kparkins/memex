"""BM25 fulltext search over the Neo4j revision index.

Provides query sanitization, fuzzy matching, and deprecated-item
filtering for the ``revision_search_text`` fulltext index created
by :func:`memex.stores.neo4j_schema.ensure_schema`.
"""

from __future__ import annotations

from neo4j import AsyncDriver

from memex.domain.models import ItemKind, Revision
from memex.retrieval.models import BM25Result, SearchRequest
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


class BM25Search:
    """BM25 fulltext search strategy over Neo4j revision index.

    Satisfies :class:`~memex.retrieval.strategy.SearchStrategy`.

    Args:
        driver: Async Neo4j driver instance.
        database: Neo4j database name.
    """

    def __init__(
        self,
        driver: AsyncDriver,
        *,
        database: str = "neo4j",
    ) -> None:
        self._driver = driver
        self._database = database

    async def search(self, request: SearchRequest) -> list[BM25Result]:
        """Execute a BM25 fulltext search over revision search_text.

        Sanitizes the query, applies fuzzy matching for terms longer than
        2 characters, queries the ``revision_search_text`` fulltext index,
        and filters deprecated items in Cypher before returning results.

        Args:
            request: Search parameters (uses ``query``, ``limit``,
                ``include_deprecated``).

        Returns:
            BM25Results ordered by descending score.
        """
        search_query = build_search_query(request.query)
        if not search_query:
            return []

        dep_filter = (
            "" if request.include_deprecated else "WHERE i.deprecated = false "
        )

        cypher = (
            f"CALL db.index.fulltext.queryNodes("
            f"'{_FULLTEXT_INDEX_NAME}', $q) "
            f"YIELD node AS r, score "
            f"MATCH (r)-[:{RelType.REVISION_OF}]->(i:{NodeLabel.ITEM}) "
            f"{dep_filter}"
            f"RETURN r, score, i.id AS item_id, i.kind AS item_kind "
            f"ORDER BY score DESC LIMIT $lim"
        )

        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, q=search_query, lim=request.limit)
            return [
                BM25Result(
                    revision=Revision.model_validate(dict(rec["r"])),
                    score=float(rec["score"]),
                    item_id=str(rec["item_id"]),
                    item_kind=ItemKind(str(rec["item_kind"])),
                )
                async for rec in result
            ]
