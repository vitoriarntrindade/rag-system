"""OpenAI adapter for the LLM provider port."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.ports.llm_provider import BaseLLMProvider
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class OpenAILLMAdapter(BaseLLMProvider):
    """
    Concrete implementation of BaseLLMProvider using OpenAI's chat models.

    This adapter is the *only* place in the codebase that imports
    langchain_openai for LLM purposes. The rest of the core never
    touches OpenAI directly.
    """

    def __init__(self, model: str, temperature: float, api_key: str) -> None:
        """
        Initialize the OpenAI LLM adapter.

        Args:
            model:       OpenAI chat model name (e.g. "gpt-3.5-turbo").
            temperature: Sampling temperature in [0.0, 2.0].
            api_key:     OpenAI API key. Passed explicitly to avoid relying
                         on global environment state.
        """
        self._llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
        )
        logger.info(f"OpenAILLMAdapter ready (model={model}, temperature={temperature})")

    def generate(self, system_prompt: str, user_message: str) -> str:
        """
        Send a chat request to OpenAI and return the response as a string.

        Args:
            system_prompt: Instructions / context built by the generator.
            user_message:  The user's question.

        Returns:
            The model's reply as a plain string.

        Raises:
            Exception: Re-raises any error from the OpenAI API so the caller
                       can handle or log it.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        logger.debug("Sending request to OpenAI chat API")
        response = self._llm.invoke(messages)
        return response.content

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"OpenAILLMAdapter("
            f"model={self._llm.model_name!r}, "
            f"temperature={self._llm.temperature!r})"
        )
