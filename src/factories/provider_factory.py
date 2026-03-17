"""
Factory for creating LLM and embedding providers from settings.

Single entry point that reads the provider config and returns the right
adapter. Adding a new provider means adding one elif block here — nothing
else in the core needs to change.
"""

from config.settings import Settings
from src.ports.embedding_provider import BaseEmbeddingProvider
from src.ports.llm_provider import BaseLLMProvider


def create_llm_provider(settings: Settings, api_key: str) -> BaseLLMProvider:
    """
    Instantiate the correct LLM adapter based on ``settings.llm_provider``.

    Args:
        settings: Application settings (drives which adapter is created).
        api_key:  Credential for the chosen provider. Passed explicitly so
                  the factory never reads from the environment directly.

    Returns:
        A concrete BaseLLMProvider ready to use.

    Raises:
        ValueError: If ``settings.llm_provider`` names an unsupported backend.

    Example::

        provider = create_llm_provider(settings, api_key=os.getenv("OPENAI_API_KEY"))
    """
    name = settings.llm_provider.lower()

    if name == "openai":
        from src.adapters.llm.openai_llm import OpenAILLMAdapter

        return OpenAILLMAdapter(
            model=settings.openai_chat_model,
            temperature=settings.openai_temperature,
            api_key=api_key,
        )

    # ── future providers ──────────────────────────────────────────────────
    # elif name == "ollama":
    #     from src.adapters.llm.ollama_llm import OllamaLLMAdapter
    #     return OllamaLLMAdapter(model=settings.ollama_chat_model)
    #
    # elif name == "azure":
    #     from src.adapters.llm.azure_llm import AzureLLMAdapter
    #     return AzureLLMAdapter(...)
    # ─────────────────────────────────────────────────────────────────────

    raise ValueError(
        f"Unsupported LLM provider: '{settings.llm_provider}'. "
        f"Supported values: 'openai'."
    )


def create_embedding_provider(settings: Settings, api_key: str) -> BaseEmbeddingProvider:
    """
    Instantiate the correct embedding adapter based on
    ``settings.embedding_provider``.

    Args:
        settings: Application settings (drives which adapter is created).
        api_key:  Credential for the chosen provider.

    Returns:
        A concrete BaseEmbeddingProvider ready to use.

    Raises:
        ValueError: If ``settings.embedding_provider`` names an unsupported backend.

    Example::

        provider = create_embedding_provider(settings, api_key=os.getenv("OPENAI_API_KEY"))
    """
    name = settings.embedding_provider.lower()

    if name == "openai":
        from src.adapters.embeddings.openai_embeddings import OpenAIEmbeddingAdapter

        return OpenAIEmbeddingAdapter(
            model=settings.openai_embedding_model,
            api_key=api_key,
        )

    # ── future providers ──────────────────────────────────────────────────
    # elif name == "ollama":
    #     from src.adapters.embeddings.ollama_embeddings import OllamaEmbeddingAdapter
    #     return OllamaEmbeddingAdapter(model=settings.ollama_embedding_model)
    #
    # elif name == "azure":
    #     from src.adapters.embeddings.azure_embeddings import AzureEmbeddingAdapter
    #     return AzureEmbeddingAdapter(...)
    # ─────────────────────────────────────────────────────────────────────

    raise ValueError(
        f"Unsupported embedding provider: '{settings.embedding_provider}'. "
        f"Supported values: 'openai'."
    )
