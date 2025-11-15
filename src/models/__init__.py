"""
Models модуль для Semantic Contamination

Модели для улучшения качества агентов:
- PairRM: Pairwise ranking model для JudgeAgent
- Другие модели будут добавлены по мере необходимости

Пример использования:
    from models import PairRMRanker

    ranker = PairRMRanker()
    score = ranker.compare("Text A", "Text B", "Question")
"""

from .pairrm_ranker import PairRMRanker

__all__ = [
    "PairRMRanker",
]
