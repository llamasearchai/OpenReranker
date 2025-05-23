import re
from typing import Optional

try:
    from pygments import lexers
    from pygments.util import ClassNotFound

    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False


def detect_language(code: str) -> Optional[str]:
    """
    Detect the programming language of code.

    Args:
        code: The input code

    Returns:
        The detected language or None if detection fails
    """
    if PYGMENTS_AVAILABLE:
        try:
            lexer = lexers.guess_lexer(code)
            return lexer.name.lower()
        except ClassNotFound:
            pass

    # Try to guess based on common patterns
    if (
        re.search(r"function\s+\w+\s*\([^)]*\)\s*{", code)
        or re.search(r"var\s+\w+\s*=", code)
        or re.search(r"const\s+\w+\s*=", code)
    ):
        return "javascript"
    elif (
        re.search(r"def\s+\w+\s*\([^)]*\):", code)
        or re.search(r"import\s+\w+", code)
        or re.search(r"class\s+\w+:", code)
    ):
        return "python"
    elif re.search(r"public\s+class\s+\w+", code) or re.search(
        r"public\s+static\s+void\s+main", code
    ):
        return "java"
    elif re.search(r"#include\s*<\w+\.h>", code) or re.search(
        r"int\s+main\s*\(\s*\)", code
    ):
        return "c"
    elif re.search(r"package\s+\w+", code) or re.search(
        r"func\s+\w+\s*\([^)]*\)", code
    ):
        return "go"
    else:
        return None


def format_code_for_reranking(code: str, language: Optional[str] = None) -> str:
    """
    Format code for reranking.

    Args:
        code: The input code
        language: The programming language

    Returns:
        Formatted code for reranking
    """
    # Clean up the code
    code = code.strip()

    # Detect language if not provided
    if language is None:
        language = detect_language(code)

    # Create a formatted string with language info if available
    if language:
        formatted_code = f"Code in {language}:\n\n{code}"
    else:
        formatted_code = f"Code:\n\n{code}"

    return formatted_code
