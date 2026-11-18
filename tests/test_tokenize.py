from __future__ import annotations

import types

import pytest
from hypothesis import given
from hypothesis import strategies as st

from findex.tokenize import tokenize


def test_is_generator() -> None:
    gen = tokenize("hello world")
    assert isinstance(gen, types.GeneratorType)
    assert list(gen) == ["hello", "world"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", []),
        ("   \t\n  ", []),
        ("Python PYTHON python", ["python", "python", "python"]),
        ("Cafe\u0301 vs Caf\u00e9", ["café", "vs", "café"]),
        ("П'ять м'ячів — це не п’ять яблук!", ["п'ять", "м'ячів", "це", "не", "п’ять", "яблук"]),
        (
            "state-of-the-art, well-known_test --foo-- _bar_",
            ["state-of-the-art", "well-known", "test", "foo", "bar"],
        ),
        ("don't can't it's", ["don't", "can't", "it's"]),
        ("123 45.67 v0.4.0", ["123", "45", "67", "v0", "4", "0"]),
    ],
)
def test_tokenize_cases(raw: str, expected: list[str]) -> None:
    assert list(tokenize(raw)) == expected


@given(st.text())
def test_prop_no_empty_tokens_and_casefolded(text: str) -> None:
    for tok in tokenize(text):
        assert tok != ""
        assert tok == tok.casefold()


@given(st.text())
def test_prop_tokenize_idempotent(text: str) -> None:
    first = list(tokenize(text))
    second = list(tokenize(" ".join(first)))
    assert second == first
