"""
Funções utilitárias para pré-processamento de texto.
"""

import re


def preprocess(text: str) -> str:
    """
    Converte o texto para lowercase, remove espaços extras e pontuação.
    
    Args:
        text (str): Texto de entrada.
    
    Returns:
        str: Texto pré-processado.
    """
    text = text.lower().strip()
    # Remove caracteres que não sejam letras, números ou espaços
    text = re.sub(r"[^\w\s]", "", text)
    return text
