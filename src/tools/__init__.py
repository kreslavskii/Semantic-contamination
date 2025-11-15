"""
Tools модуль для Semantic Contamination

Инструменты для агентов:
- Веб-поиск (SerpAPI)
- Web scraping
- Другие внешние API

Пример использования:
    from tools import WebSearchTool

    search = WebSearchTool()
    results = search.search("What is the capital of France?")
"""

from .web_search import WebSearchTool, SearchResult

__all__ = [
    "WebSearchTool",
    "SearchResult",
]
