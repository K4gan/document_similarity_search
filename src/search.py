from __future__ import annotations

import argparse
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DOCUMENTS = {
    "runbook-cache": "Cache invalidation runbook for API latency and hot key incidents.",
    "billing-faq": "Customer billing questions around refunds, invoices and taxes.",
    "search-notes": "Search relevance notes covering tf idf, tokenization and ranking.",
    "deploy-guide": "Deployment checklist for services, migrations and rollback windows.",
}


@dataclass(frozen=True)
class SearchHit:
    document_id: str
    score: float
    text: str


def search(query: str, limit: int = 3) -> list[SearchHit]:
    keys = list(DOCUMENTS)
    corpus = [DOCUMENTS[key] for key in keys]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    order = scores.argsort()[::-1][:limit]
    return [SearchHit(keys[i], float(scores[i]), DOCUMENTS[keys[i]]) for i in order]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search local documents with TF-IDF.")
    parser.add_argument("query")
    args = parser.parse_args()
    for hit in search(args.query):
        print(f"{hit.document_id} score={hit.score:.3f} :: {hit.text}")


if __name__ == "__main__":
    main()
