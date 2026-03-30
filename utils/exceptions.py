"""
utils/exceptions.py — Custom exception hierarchy for Bharat Law.
"""


class BharatLawError(Exception):
    """Base exception for all Bharat Law errors."""


class DomainNotFoundError(BharatLawError):
    """Raised when domain routing fails completely."""


class LowConfidenceError(BharatLawError):
    """Raised when retrieval confidence is below the LOW threshold."""


class CitationValidationError(BharatLawError):
    """Raised when a citation from the LLM cannot be validated."""


class IngestionError(BharatLawError):
    """Raised during ingestion pipeline failures."""


class EmbeddingError(BharatLawError):
    """Raised when embedding generation fails."""


class PineconeError(BharatLawError):
    """Raised on Pinecone connectivity or upsert errors."""


class GeminiError(BharatLawError):
    """Raised on Gemini API errors."""


class OutOfDomainError(BharatLawError):
    """Raised when a query is clearly outside Indian law scope."""
