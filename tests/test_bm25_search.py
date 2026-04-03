"""Tests for BM25 fulltext search over revisions."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from neo4j import AsyncDriver

from memex.domain.models import Item, ItemKind, Project, Revision, Space, Tag
from memex.retrieval.bm25 import (
    BM25Result,
    bm25_search,
    build_search_query,
    sanitize_query,
)
from memex.stores.neo4j_schema import ensure_schema
from memex.stores.neo4j_store import Neo4jStore

# -- Unit tests: sanitize_query ------------------------------------------


class TestSanitizeQuery:
    """Tests for Lucene special character escaping."""

    def test_plain_text_unchanged(self) -> None:
        """Plain alphanumeric text passes through unmodified."""
        assert sanitize_query("hello world") == "hello world"

    def test_escapes_plus(self) -> None:
        """Plus operator is escaped."""
        assert sanitize_query("hello+world") == "hello\\+world"

    def test_escapes_minus(self) -> None:
        """Minus/exclusion operator is escaped."""
        assert sanitize_query("hello-world") == "hello\\-world"

    def test_escapes_double_ampersand(self) -> None:
        """Individual ampersands are escaped, preventing AND operator."""
        assert sanitize_query("a&&b") == "a\\&\\&b"

    def test_escapes_double_pipe(self) -> None:
        """Individual pipes are escaped, preventing OR operator."""
        assert sanitize_query("a||b") == "a\\|\\|b"

    def test_escapes_exclamation(self) -> None:
        """NOT operator is escaped."""
        assert sanitize_query("!important") == "\\!important"

    def test_escapes_parentheses(self) -> None:
        """Grouping parentheses are escaped."""
        assert sanitize_query("(group)") == "\\(group\\)"

    def test_escapes_brackets(self) -> None:
        """Range brackets are escaped."""
        assert sanitize_query("[range]") == "\\[range\\]"

    def test_escapes_braces(self) -> None:
        """Range braces are escaped."""
        assert sanitize_query("{curly}") == "\\{curly\\}"

    def test_escapes_caret(self) -> None:
        """Boost caret is escaped."""
        assert sanitize_query("boost^2") == "boost\\^2"

    def test_escapes_tilde(self) -> None:
        """Fuzzy/proximity tilde is escaped."""
        assert sanitize_query("fuzzy~1") == "fuzzy\\~1"

    def test_escapes_asterisk(self) -> None:
        """Wildcard asterisk is escaped."""
        assert sanitize_query("wild*") == "wild\\*"

    def test_escapes_question_mark(self) -> None:
        """Wildcard question mark is escaped."""
        assert sanitize_query("wild?") == "wild\\?"

    def test_escapes_colon(self) -> None:
        """Field separator colon is escaped."""
        assert sanitize_query("field:value") == "field\\:value"

    def test_escapes_quote(self) -> None:
        """Phrase quotes are escaped."""
        assert sanitize_query('"phrase"') == '\\"phrase\\"'

    def test_escapes_backslash(self) -> None:
        """Escape character itself is escaped."""
        assert sanitize_query("back\\slash") == "back\\\\slash"

    def test_escapes_slash(self) -> None:
        """Regex delimiter slash is escaped."""
        assert sanitize_query("path/to") == "path\\/to"

    def test_multiple_specials_combined(self) -> None:
        """Multiple special characters in one query are all escaped."""
        result = sanitize_query('hello + "world"')
        assert result == 'hello \\+ \\"world\\"'

    def test_strips_leading_trailing_whitespace(self) -> None:
        """Leading and trailing whitespace is stripped."""
        assert sanitize_query("  hello  ") == "hello"

    def test_empty_string(self) -> None:
        """Empty input produces empty output."""
        assert sanitize_query("") == ""


# -- Unit tests: build_search_query --------------------------------------


class TestBuildSearchQuery:
    """Tests for fuzzy query construction."""

    def test_long_terms_get_fuzzy(self) -> None:
        """Terms > 2 characters receive ~1 fuzzy suffix."""
        result = build_search_query("hello world")
        assert result == "hello~1 world~1"

    def test_short_terms_no_fuzzy(self) -> None:
        """2-character terms do not receive fuzzy suffix."""
        result = build_search_query("hi is")
        assert result == "hi is"

    def test_mixed_lengths(self) -> None:
        """Only terms > 2 chars get fuzzy; shorter terms are plain."""
        result = build_search_query("go fast")
        assert result == "go fast~1"

    def test_boundary_three_chars_gets_fuzzy(self) -> None:
        """3-character term exceeds threshold and gets fuzzy."""
        result = build_search_query("the")
        assert result == "the~1"

    def test_special_chars_escaped_then_fuzzy(self) -> None:
        """Special chars are escaped before fuzzy suffix is added."""
        result = build_search_query("hello+world")
        assert result == "hello\\+world~1"

    def test_empty_query_returns_empty(self) -> None:
        """Empty input produces empty query."""
        assert build_search_query("") == ""

    def test_whitespace_only_returns_empty(self) -> None:
        """Whitespace-only input produces empty query."""
        assert build_search_query("   ") == ""


# -- Integration test fixtures -------------------------------------------


@pytest.fixture
async def search_env(
    neo4j_driver: AsyncDriver,
) -> AsyncIterator[dict[str, Any]]:
    """Seed test data for BM25 search tests.

    Creates a project, space, and three items:
    - item1 (active): machine learning content
    - item2 (active): graph database content
    - item3 (deprecated): overlapping machine learning content

    Yields:
        Dict with driver, store, and all created domain objects.
    """
    await ensure_schema(neo4j_driver)
    store = Neo4jStore(neo4j_driver)

    # Clean slate
    async with neo4j_driver.session() as session:
        await (await session.run("MATCH (n) DETACH DELETE n")).consume()

    project = await store.create_project(Project(name="search-test"))
    space = await store.create_space(
        Space(project_id=project.id, name="research"),
    )

    # Active item about machine learning
    item1 = Item(space_id=space.id, name="ml-basics", kind=ItemKind.FACT)
    rev1 = Revision(
        item_id=item1.id,
        revision_number=1,
        content="Machine learning fundamentals",
        search_text=(
            "Machine learning is a subset of artificial intelligence"
            " that enables systems to learn from data"
        ),
    )
    tag1 = Tag(item_id=item1.id, name="active", revision_id=rev1.id)
    await store.create_item_with_revision(item1, rev1, [tag1])

    # Active item about graph databases
    item2 = Item(space_id=space.id, name="graph-db", kind=ItemKind.DECISION)
    rev2 = Revision(
        item_id=item2.id,
        revision_number=1,
        content="Graph database overview",
        search_text=(
            "Graph databases store data as nodes and relationships"
            " enabling efficient traversal queries"
        ),
    )
    tag2 = Tag(item_id=item2.id, name="active", revision_id=rev2.id)
    await store.create_item_with_revision(item2, rev2, [tag2])

    # Deprecated item with overlapping ML content
    item3 = Item(space_id=space.id, name="old-ml", kind=ItemKind.FACT)
    rev3 = Revision(
        item_id=item3.id,
        revision_number=1,
        content="Deprecated ML information",
        search_text=(
            "Machine learning deprecated old information"
            " that should be hidden from default retrieval"
        ),
    )
    tag3 = Tag(item_id=item3.id, name="active", revision_id=rev3.id)
    await store.create_item_with_revision(item3, rev3, [tag3])
    await store.deprecate_item(item3.id)

    # Allow fulltext index to update
    await asyncio.sleep(1)

    yield {
        "driver": neo4j_driver,
        "store": store,
        "item1": item1,
        "rev1": rev1,
        "item2": item2,
        "rev2": rev2,
        "item3": item3,
        "rev3": rev3,
    }

    # Cleanup
    async with neo4j_driver.session() as session:
        await (await session.run("MATCH (n) DETACH DELETE n")).consume()


# -- Integration tests: basic search -------------------------------------


class TestBM25Search:
    """Tests for BM25 fulltext search functionality."""

    async def test_finds_matching_revision(self, search_env: dict[str, Any]) -> None:
        """Search for known terms returns the matching revision."""
        results = await bm25_search(search_env["driver"], "machine learning")
        rev_ids = {r.revision.id for r in results}
        assert search_env["rev1"].id in rev_ids

    async def test_results_ordered_by_score_descending(
        self, search_env: dict[str, Any]
    ) -> None:
        """Results are returned in descending score order."""
        results = await bm25_search(search_env["driver"], "data")
        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    async def test_limit_caps_results(self, search_env: dict[str, Any]) -> None:
        """Limit parameter restricts result count."""
        results = await bm25_search(search_env["driver"], "data", limit=1)
        assert len(results) <= 1

    async def test_result_model_structure(self, search_env: dict[str, Any]) -> None:
        """Results contain valid BM25Result with correct types."""
        results = await bm25_search(search_env["driver"], "graph databases")
        assert len(results) >= 1
        r = results[0]
        assert isinstance(r, BM25Result)
        assert isinstance(r.revision, Revision)
        assert r.score > 0
        assert isinstance(r.item_kind, ItemKind)
        assert r.item_id == search_env["item2"].id

    async def test_no_results_for_unmatched_query(
        self, search_env: dict[str, Any]
    ) -> None:
        """Query with no matching terms returns empty list."""
        results = await bm25_search(search_env["driver"], "quantum superconductor")
        assert results == []

    async def test_empty_query_returns_empty(self, search_env: dict[str, Any]) -> None:
        """Empty query string returns empty list without hitting index."""
        results = await bm25_search(search_env["driver"], "")
        assert results == []


# -- Integration tests: deprecated-item exclusion -------------------------


class TestDeprecatedExclusion:
    """Tests for deprecated item filtering in Cypher."""

    async def test_deprecated_excluded_by_default(
        self, search_env: dict[str, Any]
    ) -> None:
        """Default search excludes revisions from deprecated items."""
        results = await bm25_search(search_env["driver"], "machine learning")
        rev_ids = {r.revision.id for r in results}
        assert search_env["rev1"].id in rev_ids
        assert search_env["rev3"].id not in rev_ids

    async def test_deprecated_included_when_flag_set(
        self, search_env: dict[str, Any]
    ) -> None:
        """include_deprecated=True returns deprecated item revisions."""
        results = await bm25_search(
            search_env["driver"],
            "machine learning",
            include_deprecated=True,
        )
        rev_ids = {r.revision.id for r in results}
        assert search_env["rev1"].id in rev_ids
        assert search_env["rev3"].id in rev_ids


# -- Integration tests: query sanitization safety -------------------------


class TestQuerySanitizationIntegration:
    """Tests that special characters don't cause index errors."""

    async def test_special_chars_safe(self, search_env: dict[str, Any]) -> None:
        """Query with various special characters executes without error."""
        results = await bm25_search(search_env["driver"], 'hello + (world) "test"')
        assert isinstance(results, list)

    async def test_lucene_injection_safe(self, search_env: dict[str, Any]) -> None:
        """Lucene operator injection attempt is neutralized."""
        results = await bm25_search(search_env["driver"], "field:value AND !excluded")
        assert isinstance(results, list)


# -- Integration tests: fuzzy matching ------------------------------------


class TestFuzzyMatching:
    """Tests for fuzzy lexical matching with edit distance 1."""

    async def test_fuzzy_finds_close_match(self, search_env: dict[str, Any]) -> None:
        """Misspelled term within edit distance 1 still matches."""
        # "machne" is 1 edit from "machine" (missing 'i')
        results = await bm25_search(search_env["driver"], "machne learning")
        rev_ids = {r.revision.id for r in results}
        assert search_env["rev1"].id in rev_ids

    async def test_fuzzy_applied_to_long_terms_only(self) -> None:
        """Verify build_search_query adds ~1 only to terms > 2 chars."""
        query = build_search_query("an machne")
        assert "an " in query
        assert "machne~1" in query
