import inspect

from findex.tokenize import tokenize


def test_is_generator():
    assert inspect.isgeneratorfunction(tokenize)


def test_empty_and_whitespace():
    assert list(tokenize("")) == []
    assert list(tokenize("   \n\t  ")) == []


def test_mixed_case_and_casefold():
    assert list(tokenize("Python SEARCH Engine")) == ["python", "search", "engine"]
    assert list(tokenize("Straße")) == ["strasse"]


def test_unicode_nfc_normalization():
    composed = "café"
    decomposed = "cafe\u0301"
    assert list(tokenize(composed)) == list(tokenize(decomposed)) == ["café"]


def test_cyrillic_and_apostrophes():
    text = "П'ять великих м’ячів у Києві!"
    assert list(tokenize(text)) == ["п'ять", "великих", "м’ячів", "у", "києві"]


def test_hyphens_and_punctuation():
    text = "State-of-the-art — NLP, v2.0 (2026)!"
    assert list(tokenize(text)) == ["state-of-the-art", "nlp", "v2", "0", "2026"]


def test_english_contractions():
    assert list(tokenize("Don't stop believing.")) == ["don't", "stop", "believing"]
