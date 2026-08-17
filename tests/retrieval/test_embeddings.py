from unittest.mock import patch, MagicMock
from backend.retrieval.embeddings import get_embeddings


@patch("backend.retrieval.embeddings.OllamaEmbeddings")
def test_embedding_model(mock_ollama_embeddings):
    # Mock the embedding function to return a vector
    mock_embeddings_instance = MagicMock()
    mock_embeddings_instance.embed_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_ollama_embeddings.return_value = mock_embeddings_instance
    
    embeddings = get_embeddings()

    result = embeddings.embed_query(
        "What are the symptoms of bladder cancer?"
    )

    assert isinstance(result, list)
    assert len(result) > 0