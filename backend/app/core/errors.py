from fastapi import HTTPException, status


class ServiceError(Exception):
    """Base class for service errors"""

    def __init__(self, message: str = "An error occurred"):
        self.message = message
        super().__init__(self.message)


class UserNotFoundError(ServiceError):
    """Raised when a user is not found"""

    def __init__(self, message: str = "User not found"):
        super().__init__(message)


class TranscriptionNotFoundError(ServiceError):
    """Raised when a transcription is not found"""

    def __init__(self, message: str = "Transcription not found"):
        super().__init__(message)


class InvalidCredentialsError(ServiceError):
    """Raised when credentials are invalid"""

    def __init__(self, message: str = "Invalid credentials"):
        super().__init__(message)


class PermissionDeniedError(ServiceError):
    """Raised when a user doesn't have permission"""

    def __init__(self, message: str = "Permission denied"):
        super().__init__(message)


class FileUploadError(ServiceError):
    """Raised when there's an error uploading a file"""

    def __init__(self, message: str = "Error uploading file"):
        super().__init__(message)


class FileProcessingError(ServiceError):
    """Raised when there's an error processing a file"""

    def __init__(self, message: str = "Error processing file"):
        super().__init__(message)


class TranscriptionError(ServiceError):
    """Raised when there's an error during transcription"""

    def __init__(self, message: str = "Error during transcription"):
        super().__init__(message)


class LLMProcessingError(ServiceError):
    """Raised when there's an error processing with LLM"""

    def __init__(self, message: str = "Error during LLM processing"):
        super().__init__(message)


def service_error_handler(error: ServiceError):
    """Convert service errors to HTTP exceptions"""
    error_mapping = {
        UserNotFoundError: (status.HTTP_404_NOT_FOUND, "User not found"),
        TranscriptionNotFoundError: (status.HTTP_404_NOT_FOUND, "Transcription not found"),
        InvalidCredentialsError: (status.HTTP_401_UNAUTHORIZED, "Invalid credentials"),
        PermissionDeniedError: (status.HTTP_403_FORBIDDEN, "Permission denied"),
        FileUploadError: (status.HTTP_400_BAD_REQUEST, "Error uploading file"),
        FileProcessingError: (status.HTTP_500_INTERNAL_SERVER_ERROR, "Error processing file"),
        TranscriptionError: (status.HTTP_500_INTERNAL_SERVER_ERROR, "Error during transcription"),
        LLMProcessingError: (status.HTTP_500_INTERNAL_SERVER_ERROR, "Error during LLM processing"),
    }

    for error_type, (status_code, default_detail) in error_mapping.items():
        if isinstance(error, error_type):
            detail = error.message or default_detail
            raise HTTPException(status_code=status_code, detail=detail)

    # Default case for unhandled service errors
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=error.message or "An unexpected error occurred"
    )
