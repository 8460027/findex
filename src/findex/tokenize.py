import re
import unicodedata
from collections.abc import Iterator

# Policy:
# - Matches Unicode word characters (\w excludes punctuation/whitespace, includes letters, digits, '_')
# - Keeps internal apostrophes (' or ’) for contractions/Ukrainian words (don't, п'ять)
# - Keeps internal hyphens (-) for compound words (state-of-the-art, південно-західний)
# - Strips standalone underscores
TOKEN_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


def tokenize(text: str) -> Iterator[str]:
    """Normalize text (NFC + casefold) and lazily yield tokens."""
    if not text:
        return
    normalized = unicodedata.normalize("NFC", text).casefold()
    for match in TOKEN_RE.finditer(normalized):
        yield match.group(0)
