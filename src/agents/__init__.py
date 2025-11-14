"""
Пакет агентов для системы сравнения и слияния текстов
"""
from .base_agent import BaseAgent
from .extractor import ExtractorAgent
from .aligner import AlignerAgent
from .judge import JudgeAgent
from .verifier import VerifierAgent

__all__ = [
    'BaseAgent',
    'ExtractorAgent',
    'AlignerAgent',
    'JudgeAgent',
    'VerifierAgent'
]
