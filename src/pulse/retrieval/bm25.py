"""Keyword search with BM25, the classic ranking formula behind most search engines.

The idea in one sentence: a chunk scores higher when it contains the query's words,
especially rare words, and it doesn't get unlimited credit for repeating a word or
for simply being long.

For each query word w and chunk c:

    idf(w)   = log(1 + (N - n_w + 0.5) / (n_w + 0.5))      rare words count more
    tf part  = f * (k1 + 1) / (f + k1 * (1 - b + b * len(c) / avg_len))
    score(c) = sum over query words of idf(w) * tf part

N = number of chunks, n_w = chunks containing w, f = times w appears in c.
k1 limits the reward for repeats; b corrects for chunk length.
"""

from __future__ import annotations

import math
import re
from collections import Counter

K1 = 1.5
B = 0.75

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "the", "this", "to", "what", "when", "where",
    "which", "with", "self", "def", "return", "import", "none", "true", "false",
}


def tokenize(text: str) -> list[str]:
    """Split code and prose into lowercase words.

    Identifiers are split too, so `read_file`, `readFile` and "read file" all
    produce the words "read" and "file".
    """
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  # camelCase -> camel Case
    words = re.split(r"[^A-Za-z0-9]+", text.lower())
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


class BM25:
    def __init__(self, documents: list[str]):
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.docs else 0.0
        doc_freq: Counter[str] = Counter()
        for counts in self.docs:
            doc_freq.update(counts.keys())
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - df + 0.5) / (df + 0.5)) for w, df in doc_freq.items()}

    def scores(self, query: str) -> list[float]:
        words = tokenize(query)
        out = []
        for counts, length in zip(self.docs, self.lengths, strict=True):
            score = 0.0
            for w in words:
                f = counts.get(w, 0)
                if f:
                    norm = K1 * (1 - B + B * length / (self.avg_len or 1))
                    score += self.idf[w] * f * (K1 + 1) / (f + norm)
            out.append(score)
        return out
