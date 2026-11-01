"""Exercise multilingual retrieval with the actual local vector-store boundary."""

import math
import re
import unicodedata
import unittest

from langchain_core.vectorstores import InMemoryVectorStore

from maya.framework.memory import DeterministicMemoryEmbeddings
from maya.framework.rag.vectorstore import DeterministicEmbeddings


class MultilingualEmbeddingTests(unittest.TestCase):
    def test_greek_and_german_queries_retrieve_their_own_document(self):
        for backend in (DeterministicEmbeddings, DeterministicMemoryEmbeddings):
            for text in ("Τι είναι η καρδιοτοξικότητα;", "Wie wirkt sich die Behandlung auf das Herz aus?"):
                with self.subTest(backend=backend.__name__, text=text):
                    embeddings = backend(size=64)
                    store = InMemoryVectorStore.from_texts([text], embeddings)
                    [(document, score)] = store.similarity_search_with_score(text, k=1)
                    self.assertEqual(document.page_content, text)
                    self.assertTrue(math.isfinite(score))
                    self.assertAlmostEqual(score, 1.0)

    def test_composed_and_decomposed_accents_have_identical_embeddings(self):
        for backend in (DeterministicEmbeddings, DeterministicMemoryEmbeddings):
            for text in ("καρκίνος", "Körpergewicht"):
                with self.subTest(backend=backend.__name__, text=text):
                    embeddings = backend(size=64)
                    original = embeddings.embed_query(text)
                    self.assertTrue(any(original))
                    self.assertEqual(original, embeddings.embed_query(unicodedata.normalize("NFD", text)))
                    self.assertEqual(original, embeddings.embed_documents([text])[0])

    def test_existing_ascii_token_vectors_are_preserved(self):
        text = "Blood pressure 134/82 mmHg; dose_2.5 mg."
        values = [0.0] * 64
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            values[hash(token) % 64] += 1.0
        norm = sum(value * value for value in values) ** 0.5
        legacy = [value / norm for value in values]
        for backend in (DeterministicEmbeddings, DeterministicMemoryEmbeddings):
            with self.subTest(backend=backend.__name__):
                self.assertEqual(backend(size=64).embed_query(text), legacy)


if __name__ == "__main__":
    unittest.main()
