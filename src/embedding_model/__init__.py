from .NVEmbedV2 import NVEmbedV2EmbeddingModel

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    if "NV-Embed-v2" in embedding_model_name:
        return NVEmbedV2EmbeddingModel
    assert False, f"Unknown embedding model name: {embedding_model_name}"
