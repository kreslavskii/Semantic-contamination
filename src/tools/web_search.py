"""
Веб-поиск через SerpAPI для фактчекинга
"""
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

try:
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    Результат поиска

    Атрибуты:
        title: Заголовок результата
        url: URL страницы
        snippet: Короткий текст из страницы
        date: Дата публикации (если доступна)
        source_type: Тип источника (academic, news, other)
        position: Позиция в результатах поиска
    """
    title: str
    url: str
    snippet: str
    date: Optional[str] = None
    source_type: str = "other"
    position: int = 0

    def __str__(self) -> str:
        return f"{self.title}\n{self.url}\n{self.snippet}"


class WebSearchTool:
    """
    Инструмент для веб-поиска через SerpAPI

    Пример использования:
        search = WebSearchTool()
        results = search.search("climate change effects")

        for result in results:
            print(f"{result.title}: {result.snippet}")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        num_results: int = 10,
        safe_search: bool = True
    ):
        """
        Инициализация веб-поиска

        Args:
            api_key: SerpAPI ключ (если None, берётся из settings)
            num_results: Количество результатов для возврата
            safe_search: Включить безопасный поиск
        """
        self.api_key = api_key or settings.SERPAPI_API_KEY
        self.num_results = num_results
        self.safe_search = safe_search

        if not self.api_key:
            logger.warning(
                "SerpAPI key не найден. Веб-поиск будет недоступен. "
                "Установите SERPAPI_API_KEY в .env"
            )

        if GoogleSearch is None:
            logger.warning(
                "google-search-results не установлен. "
                "Установите: pip install google-search-results"
            )

    def search(
        self,
        query: str,
        num_results: Optional[int] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Выполнить поиск

        Args:
            query: Поисковый запрос
            num_results: Количество результатов (переопределяет default)
            **kwargs: Дополнительные параметры для SerpAPI

        Returns:
            Список SearchResult

        Raises:
            ValueError: Если API ключ не настроен
        """
        if settings.DRY_RUN:
            return self._mock_search(query)

        if not self.api_key:
            raise ValueError(
                "SerpAPI key не найден. Установите SERPAPI_API_KEY в .env"
            )

        if GoogleSearch is None:
            raise ImportError(
                "google-search-results не установлен. "
                "Установите: pip install google-search-results"
            )

        num = num_results or self.num_results

        try:
            # Параметры поиска
            params = {
                "q": query,
                "api_key": self.api_key,
                "num": num,
                "safe": "active" if self.safe_search else "off",
                **kwargs
            }

            logger.info(f"Searching for: {query}")

            # Выполнение поиска
            search = GoogleSearch(params)
            results_dict = search.get_dict()

            # Парсинг результатов
            results = self._parse_results(results_dict)

            logger.info(f"Found {len(results)} results for query: {query}")

            return results

        except Exception as e:
            logger.error(f"Search error: {e}")
            raise

    def _parse_results(self, results_dict: Dict) -> List[SearchResult]:
        """
        Парсинг результатов из SerpAPI

        Args:
            results_dict: Словарь ответа от SerpAPI

        Returns:
            Список SearchResult
        """
        results = []

        # Органические результаты
        organic_results = results_dict.get("organic_results", [])

        for i, item in enumerate(organic_results, 1):
            result = SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                date=item.get("date"),
                source_type=self._classify_source(item),
                position=i
            )
            results.append(result)

        # Knowledge graph (если есть)
        knowledge_graph = results_dict.get("knowledge_graph", {})
        if knowledge_graph:
            description = knowledge_graph.get("description", "")
            if description:
                result = SearchResult(
                    title=knowledge_graph.get("title", "Knowledge Graph"),
                    url=knowledge_graph.get("website", ""),
                    snippet=description,
                    source_type="knowledge_graph",
                    position=0  # Knowledge graph всегда первый
                )
                results.insert(0, result)

        return results

    def _classify_source(self, item: Dict) -> str:
        """
        Классификация типа источника

        Args:
            item: Элемент результата поиска

        Returns:
            Тип источника: academic, news, government, other
        """
        url = item.get("link", "").lower()
        title = item.get("title", "").lower()

        # Академические источники
        academic_domains = [
            ".edu", "scholar.google", "arxiv.org", "pubmed",
            "researchgate", "academia.edu", "doi.org"
        ]
        if any(domain in url for domain in academic_domains):
            return "academic"

        # Новостные источники
        news_indicators = [
            "news", "bbc", "cnn", "reuters", "nytimes",
            "guardian", "washingtonpost", "bloomberg"
        ]
        if any(indicator in url or indicator in title for indicator in news_indicators):
            return "news"

        # Правительственные источники
        if ".gov" in url or "government" in url:
            return "government"

        # Wikipedia
        if "wikipedia.org" in url:
            return "wikipedia"

        return "other"

    def _mock_search(self, query: str) -> List[SearchResult]:
        """
        Mock поиск для DRY_RUN режима

        Args:
            query: Поисковый запрос

        Returns:
            Список mock результатов
        """
        logger.info(f"[DRY RUN] Mock search for: {query}")

        return [
            SearchResult(
                title=f"Mock Result 1 for '{query}'",
                url="https://example.com/result1",
                snippet=f"This is a mock search result for the query: {query}. "
                       "In real mode, this would contain actual web search results.",
                date="2024-01-01",
                source_type="other",
                position=1
            ),
            SearchResult(
                title=f"Mock Academic Source for '{query}'",
                url="https://example.edu/paper",
                snippet=f"Academic research about {query}. Mock content.",
                date="2023-12-15",
                source_type="academic",
                position=2
            ),
        ]

    def search_and_format(
        self,
        query: str,
        max_results: int = 5
    ) -> str:
        """
        Поиск с форматированием результатов в текст

        Args:
            query: Поисковый запрос
            max_results: Максимальное количество результатов

        Returns:
            Отформатированная строка с результатами

        Пример:
            text = search.search_and_format("capital of France")
            # Returns:
            # 1. Paris - Wikipedia
            #    https://en.wikipedia.org/wiki/Paris
            #    Paris is the capital and most populous city of France...
            #
            # 2. ...
        """
        results = self.search(query, num_results=max_results)

        if not results:
            return f"No results found for: {query}"

        formatted = f"Search results for: {query}\n\n"

        for i, result in enumerate(results[:max_results], 1):
            formatted += f"{i}. {result.title}\n"
            formatted += f"   {result.url}\n"
            formatted += f"   {result.snippet}\n"
            if result.date:
                formatted += f"   Date: {result.date}\n"
            formatted += f"   Source type: {result.source_type}\n\n"

        return formatted

    def get_top_result(self, query: str) -> Optional[SearchResult]:
        """
        Получить только первый результат

        Args:
            query: Поисковый запрос

        Returns:
            Первый SearchResult или None
        """
        results = self.search(query, num_results=1)
        return results[0] if results else None

    def filter_by_source_type(
        self,
        results: List[SearchResult],
        source_type: str
    ) -> List[SearchResult]:
        """
        Фильтрация результатов по типу источника

        Args:
            results: Список результатов
            source_type: Тип источника для фильтрации

        Returns:
            Отфильтрованный список
        """
        return [r for r in results if r.source_type == source_type]

    def get_academic_sources(self, query: str) -> List[SearchResult]:
        """
        Получить только академические источники

        Args:
            query: Поисковый запрос

        Returns:
            Список академических результатов
        """
        results = self.search(query)
        return self.filter_by_source_type(results, "academic")


# Пример использования
if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)

    # Создание инструмента
    search = WebSearchTool()

    # Пример поиска
    query = "What is semantic contamination in NLP?"

    try:
        results = search.search(query, num_results=3)

        print(f"\nFound {len(results)} results:\n")
        for result in results:
            print(f"Title: {result.title}")
            print(f"URL: {result.url}")
            print(f"Snippet: {result.snippet[:100]}...")
            print(f"Source: {result.source_type}")
            print("-" * 50)

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Make sure SERPAPI_API_KEY is set in .env file")
