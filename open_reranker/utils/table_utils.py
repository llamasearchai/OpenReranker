from typing import List, Optional

try:
    import pandas as pd
    import tabulate

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def format_table_for_reranking(
    headers: List[str], rows: List[List[str]], table_name: Optional[str] = None
) -> str:
    """
    Format a table for reranking.

    Args:
        headers: Table headers
        rows: Table rows as list of lists
        table_name: Optional table name

    Returns:
        Formatted table text
    """
    if PANDAS_AVAILABLE:
        # Create pandas DataFrame for easy formatting
        df = pd.DataFrame(rows, columns=headers)

        # Format as markdown table
        table_str = tabulate.tabulate(
            df, headers=headers, tablefmt="pipe", showindex=False
        )
    else:
        # Simple fallback if pandas is not available
        table_str = "| " + " | ".join(headers) + " |\n"
        table_str += "| " + " | ".join(["---"] * len(headers)) + " |\n"

        for row in rows:
            table_str += "| " + " | ".join(str(cell) for cell in row) + " |\n"

    # Add table name if provided
    if table_name:
        return f"Table: {table_name}\n\n{table_str}"
    else:
        return f"Table:\n\n{table_str}"


def table_to_text(
    headers: List[str], rows: List[List[str]], table_name: Optional[str] = None
) -> str:
    """
    Convert a table to plain text format.

    Args:
        headers: Table headers
        rows: Table rows as list of lists
        table_name: Optional table name

    Returns:
        Table as plain text
    """
    # Start with table name if provided
    result = f"Table: {table_name}\n\n" if table_name else "Table:\n\n"

    # Add headers
    result += "Headers: " + ", ".join(headers) + "\n"

    # Add rows
    result += "Rows:\n"
    for i, row in enumerate(rows):
        result += f"Row {i+1}: " + ", ".join(str(cell) for cell in row) + "\n"

    return result
