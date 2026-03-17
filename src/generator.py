"""Response generator module for LLM-based answer generation."""

from typing import List, Tuple

from langchain_core.documents import Document

from src.ports.llm_provider import BaseLLMProvider
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ResponseGenerator:
    """
    Generates responses using retrieved context and LLM.
    
    This class handles creating prompts, calling the LLM, and formatting
    responses based on retrieved document context.
    """
    
    # Default system prompt template
    DEFAULT_SYSTEM_PROMPT = """You are a helpful expert assistant. Use the provided context to answer questions accurately and comprehensively.

Instructions:
- Base your answer strictly on the provided context
- If the answer isn't in the context, say "I don't have enough information to answer that question"
- Provide clear, well-structured explanations
- Include relevant details and examples when available
- Be concise but thorough

Context:
{context}
"""
    
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        system_prompt: str = None,
    ):
        """
        Initialize the response generator.

        Args:
            llm_provider:  Any object that implements BaseLLMProvider.
                           The generator does not know (or care) whether
                           it is OpenAI, Ollama, Azure, etc.
            system_prompt: Custom system prompt template with a {context}
                           placeholder. If None, uses DEFAULT_SYSTEM_PROMPT.
        """
        self.llm_provider = llm_provider
        self.system_prompt_template = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        logger.info("ResponseGenerator initialized")
    
    def _format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            documents: List of retrieved Document objects
        
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Source {i}:\n{doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    def generate(
        self,
        query: str,
        context_documents: List[Document]
    ) -> Tuple[str, List[Document]]:
        """
        Generate a response to the query using context documents.
        
        Args:
            query: User's question or query
            context_documents: Retrieved documents to use as context
        
        Returns:
            Tuple of (generated answer string, source documents used)
        """
        logger.info(f"Generating response for query")
        logger.debug(f"Query: {query}")
        logger.debug(f"Using {len(context_documents)} context documents")
        
        # Format context
        context = self._format_context(context_documents)

        # Build prompts
        system_prompt = self.system_prompt_template.format(context=context)
        user_message = f"Question: {query}"

        # Generate response — delegate to the injected provider
        try:
            answer = self.llm_provider.generate(system_prompt, user_message)

            logger.info("Response generated successfully")
            logger.debug(f"Answer length: {len(answer)} characters")

            return answer, context_documents

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def generate_streaming(
        self,
        query: str,
        context_documents: List[Document]
    ):
        """
        Generate a streaming response (future enhancement).
        
        Args:
            query: User's question or query
            context_documents: Retrieved documents to use as context
        
        Yields:
            Chunks of the generated response
        """
        # Future: implement streaming response
        raise NotImplementedError("Streaming generation not yet implemented")
