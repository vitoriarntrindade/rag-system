"""Port (interface) for LLM providers."""

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """
    Contract that any LLM provider must fulfill.

    The core (generator, pipeline) depends only on this interface,
    never on a concrete implementation such as OpenAI or Ollama.
    """

    @abstractmethod
    def generate(self, system_prompt: str, user_message: str) -> str:
        """
        Generate a text response from a prompt.

        Args:
            system_prompt: Instructions / context for the model.
            user_message:  The user's question or input.

        Returns:
            The model's response as a plain string.
        """
        ...
