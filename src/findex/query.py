from __future__ import annotations

import re
from dataclasses import dataclass

from findex.index import Index
from findex.tokenize import tokenize

_TOKEN_RE = re.compile(r'\(|\)|"[^"]*"|[^\s()"]+')


class QueryNode:
    """Base AST node supporting &, |, ~ operators and evaluation against an Index."""

    def __and__(self, other: QueryNode) -> And:
        return And(self, other)

    def __or__(self, other: QueryNode) -> Or:
        return Or(self, other)

    def __invert__(self) -> Not:
        return Not(self)

    def evaluate(self, index: Index) -> set[int]:
        raise NotImplementedError

    def positive_terms(self) -> list[str]:
        """Return non-negated terms for scoring."""
        return []


@dataclass(frozen=True)
class Term(QueryNode):
    value: str

    def evaluate(self, index: Index) -> set[int]:
        if self.value not in index:
            return set()
        return {p.doc_id for p in index[self.value]}

    def positive_terms(self) -> list[str]:
        return [self.value]


@dataclass(frozen=True)
class Phrase(QueryNode):
    terms: tuple[str, ...]

    def evaluate(self, index: Index) -> set[int]:
        if not self.terms:
            return set()
        if len(self.terms) == 1:
            return Term(self.terms[0]).evaluate(index)

        for t in self.terms:
            if t not in index:
                return set()

        first_postings = {p.doc_id: p.positions for p in index[self.terms[0]]}
        if not first_postings:
            return set()

        candidate_docs = set(first_postings.keys())
        term_pos_maps: list[dict[int, tuple[int, ...]]] = [first_postings]

        for t in self.terms[1:]:
            pmap = {p.doc_id: p.positions for p in index[t] if p.doc_id in candidate_docs}
            candidate_docs &= pmap.keys()
            if not candidate_docs:
                return set()
            term_pos_maps.append(pmap)

        matched: set[int] = set()
        for doc_id in candidate_docs:
            positions_per_term = [pmap[doc_id] for pmap in term_pos_maps]
            if any(len(pos) == 0 for pos in positions_per_term):
                raise ValueError(
                    "Phrase query requires positional index (rebuild with --positions)."
                )
            if _match_consecutive_positions(positions_per_term):
                matched.add(doc_id)

        return matched

    def positive_terms(self) -> list[str]:
        return list(self.terms)


def _match_consecutive_positions(pos_lists: list[tuple[int, ...]]) -> bool:
    """Check via two-pointer scan if there exists p in pos_lists[0] such that p+i in pos_lists[i]."""
    valid_starts = set(pos_lists[0])
    for offset, positions in enumerate(pos_lists[1:], start=1):
        i = 0
        j = 0
        starts_sorted = sorted(valid_starts)
        next_valid: set[int] = set()
        while i < len(starts_sorted) and j < len(positions):
            target = starts_sorted[i] + offset
            cur = positions[j]
            if cur == target:
                next_valid.add(starts_sorted[i])
                i += 1
                j += 1
            elif cur < target:
                j += 1
            else:
                i += 1
        if not next_valid:
            return False
        valid_starts = next_valid
    return bool(valid_starts)


@dataclass(frozen=True)
class And(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)

    def positive_terms(self) -> list[str]:
        return self.left.positive_terms() + self.right.positive_terms()


@dataclass(frozen=True)
class Or(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def positive_terms(self) -> list[str]:
        return self.left.positive_terms() + self.right.positive_terms()


@dataclass(frozen=True)
class Not(QueryNode):
    operand: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        universe = set(index.doc_lengths.keys())
        return universe - self.operand.evaluate(index)

    def positive_terms(self) -> list[str]:
        return []


class QuerySyntaxError(ValueError):
    """Raised when a query string has invalid syntax."""


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self) -> str:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> QueryNode:
        if not self.tokens:
            raise QuerySyntaxError("Empty query")
        node = self._parse_or()
        if self.peek() is not None:
            raise QuerySyntaxError(f"Unexpected token: {self.peek()!r}")
        return node

    def _parse_or(self) -> QueryNode:
        node = self._parse_and()
        while self.peek() == "OR":
            self.consume()
            right = self._parse_and()
            node = Or(node, right)
        return node

    def _parse_and(self) -> QueryNode:
        node = self._parse_not()
        while (tok := self.peek()) is not None and tok not in (")", "OR"):
            if tok == "AND":
                self.consume()
            right = self._parse_not()
            node = And(node, right)
        return node

    def _parse_not(self) -> QueryNode:
        if self.peek() == "NOT":
            self.consume()
            return Not(self._parse_not())
        return self._parse_atom()

    def _parse_atom(self) -> QueryNode:
        tok = self.peek()
        if tok is None:
            raise QuerySyntaxError("Unexpected end of query")

        if tok == "(":
            self.consume()
            node = self._parse_or()
            if self.peek() != ")":
                raise QuerySyntaxError("Unclosed '(' in query")
            self.consume()
            return node

        if tok == ")":
            raise QuerySyntaxError("Unexpected ')' in query")

        if tok in ("AND", "OR"):
            raise QuerySyntaxError(f"Unexpected operator {tok!r}")

        self.consume()
        if tok.startswith('"') and tok.endswith('"') and len(tok) >= 2:
            phrase_tokens = tuple(tokenize(tok[1:-1]))
            if not phrase_tokens:
                raise QuerySyntaxError("Empty phrase query")
            return Phrase(phrase_tokens)

        norm_tokens = list(tokenize(tok))
        if not norm_tokens:
            raise QuerySyntaxError(f"Invalid term token: {tok!r}")
        if len(norm_tokens) == 1:
            return Term(norm_tokens[0])

        node: QueryNode = Term(norm_tokens[0])
        for extra in norm_tokens[1:]:
            node = And(node, Term(extra))
        return node


def parse_query(raw: str) -> QueryNode:
    """Parse a Boolean + phrase query string into a QueryNode AST."""
    if raw.count('"') % 2 != 0:
        raise QuerySyntaxError("Unclosed quote in query")
    raw_tokens = _TOKEN_RE.findall(raw)
    return _Parser(raw_tokens).parse()
