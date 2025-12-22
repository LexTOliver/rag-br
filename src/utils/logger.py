"""
Logger utility functions.
"""

import logging


def get_logger(
    file_name: str = "app.log", level: int = logging.DEBUG
) -> logging.Logger:
    """
    Get a logger with the specified name and level.

    Parameters:
        file_name (str): Name of the log file
        level (int): Logging level

    Returns:
        logging.Logger: Logger object
    """
    # Create a logger
    logger = logging.getLogger("rag-br")
    logger.setLevel(level)

    # Check if handlers are already added
    if not logger.handlers:
        # Create a formatter
        formatter = logging.Formatter(
            "{asctime} - {levelname}: {message}", style="{", datefmt="%d-%m-%Y %H:%M:%S"
        )

        # Create a console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)

        # Create a file handler
        fh = logging.FileHandler(file_name)
        fh.setLevel(level)
        fh.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger
