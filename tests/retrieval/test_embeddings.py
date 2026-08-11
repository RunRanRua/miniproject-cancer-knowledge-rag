from backend.retrieval.embeddings import get_embeddings


def test_embedding_model():
    embeddings = get_embeddings()

    result = embeddings.embed_query(
        "What are the symptoms of bladder cancer?"
    )

    assert isinstance(result, list)
    assert len(result) > 0