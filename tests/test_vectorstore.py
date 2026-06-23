import os
import shutil
import pytest
from typing import List

pytest.importorskip("langchain_core", reason="langchain-core not installed")

from epochdb import EpochDB, AsyncEpochDB
from epochdb.vectorstore import EpochDBVectorStore, EpochDBMultiHopRetriever
from langchain_core.documents import Document


@pytest.fixture
def sync_db():
    storage_dir = "./.test_epochdb_vectorstore_sync"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir, ignore_errors=True)
    db = EpochDB(storage_dir=storage_dir, dim=4)
    yield db
    db.close()
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir, ignore_errors=True)


@pytest.fixture
async def async_db():
    storage_dir = "./.test_epochdb_vectorstore_async"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir, ignore_errors=True)
    db = AsyncEpochDB(storage_dir=storage_dir, dim=4)
    yield db
    await db.close()
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir, ignore_errors=True)


def test_sync_vectorstore_and_retriever(sync_db):
    store = EpochDBVectorStore(sync_db)
    
    # 1. Test add_texts
    ids = store.add_texts(
        texts=["Apple is a fruit.", "Banana is yellow."],
        metadatas=[{"category": "fruit"}, {"color": "yellow"}]
    )
    assert len(ids) == 2
    assert all(isinstance(i, str) for i in ids)

    # 2. Test similarity_search (use k=2 since dummy zero embeddings are used in test db)
    docs = store.similarity_search("fruit", k=2)
    assert len(docs) == 2
    assert isinstance(docs[0], Document)
    
    page_contents = [d.page_content for d in docs]
    assert "Apple is a fruit." in page_contents
    assert "Banana is yellow." in page_contents

    # 3. Test similarity_search_with_score
    docs_and_scores = store.similarity_search_with_score("Banana", k=2)
    assert len(docs_and_scores) == 2
    
    doc, score = docs_and_scores[0]
    assert isinstance(doc, Document)
    assert isinstance(score, float)

    # 4. Test Multi-Hop Retriever
    retriever = EpochDBMultiHopRetriever(db=sync_db, k=2, hops=1)
    retrieved_docs = retriever.invoke("Apple fruit")
    assert len(retrieved_docs) >= 1
    assert isinstance(retrieved_docs[0], Document)


@pytest.mark.anyio
async def test_async_vectorstore_and_retriever(async_db):
    async with async_db as db:
        store = EpochDBVectorStore(db)

        # 1. Test aadd_texts
        ids = await store.aadd_texts(
            texts=["Dog barked loudly.", "Cat meowed softly."],
            metadatas=[{"animal": "dog"}, {"animal": "cat"}]
        )
        assert len(ids) == 2

        # 2. Test asimilarity_search (use k=2 since dummy zero embeddings are used in test db)
        docs = await store.asimilarity_search("bark", k=2)
        assert len(docs) == 2
        
        page_contents = [d.page_content for d in docs]
        assert "Dog barked loudly." in page_contents
        assert "Cat meowed softly." in page_contents

        # 3. Test asimilarity_search_with_score
        docs_and_scores = await store.asimilarity_search_with_score("Cat", k=2)
        assert len(docs_and_scores) == 2
        doc, score = docs_and_scores[0]
        assert isinstance(doc, Document)
        assert isinstance(score, float)

        # 4. Test Async Multi-Hop Retriever
        retriever = EpochDBMultiHopRetriever(db=db, k=2, hops=1)
        retrieved_docs = await retriever.ainvoke("Dog meow")
        assert len(retrieved_docs) >= 1
