"""
Агент для фактчекинга через веб-поиск (Шаг 6) - Chain-of-Verification

Обновлено: 2025-11-15
- Добавлена интеграция с LLM для reasoning
- Реализован ReAct-цикл (Thought → Action → Observation)
- Добавлен реальный веб-поиск через SerpAPI
- Fallback на эвристики при отсутствии API
"""
import re
import logging
from typing import List, Dict, Tuple, Optional
from .base_agent import BaseAgent

# Импорты для новой функциональности
try:
    from ..llm import LLMClient, get_default_llm
    from ..tools import WebSearchTool
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    LLMClient = None

from ..config import settings

logger = logging.getLogger(__name__)


class VerifierAgent(BaseAgent):
    """Агент для проверки фактов с помощью внешних источников"""

    SOURCE_PRIORITY = {
        'academic': 4,        # Академические источники (приоритет выше)
        'peer_reviewed': 3,   # Рецензируемые публикации
        'government': 3,      # Правительственные источники
        'official': 2,        # Официальные отчёты
        'news': 1,           # Новостные источники
        'expert': 1,          # Экспертные источники
        'wikipedia': 1,       # Wikipedia
        'other': 0           # Прочие
    }

    def __init__(
        self,
        prompt_template_path: str = "prompts/verifier_prompt.md",
        llm_client: Optional[LLMClient] = None,
        search_tool: Optional[WebSearchTool] = None,
        use_react: bool = True
    ):
        """
        Инициализация VerifierAgent

        Args:
            prompt_template_path: Путь к промпт-шаблону
            llm_client: LLM клиент (если None, создаётся автоматически)
            search_tool: Инструмент веб-поиска (если None, создаётся автоматически)
            use_react: Использовать ReAct-цикл (если False, fallback на эвристики)
        """
        super().__init__(prompt_template_path)

        self.use_react = use_react and HAS_LLM

        # Инициализация LLM
        if self.use_react:
            try:
                self.llm = llm_client or get_default_llm()
                logger.info("VerifierAgent: LLM initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}. Falling back to heuristics")
                self.llm = None
                self.use_react = False
        else:
            self.llm = None

        # Инициализация веб-поиска
        try:
            self.search = search_tool or WebSearchTool(num_results=5)
            logger.info("VerifierAgent: WebSearchTool initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize WebSearchTool: {e}")
            self.search = None

        # Статистика
        self.verification_stats = {
            'total': 0,
            'supported': 0,
            'refuted': 0,
            'uncertain': 0,
            'conditional': 0
        }

    def process(
        self,
        claims: List[Dict],
        conflicts: List[Dict] = None,
        judgments: List[Dict] = None
    ) -> List[Dict]:
        """
        Проверка фактов в спорных тезисах

        Args:
            claims: Список всех тезисов
            conflicts: Список конфликтов (опционально)
            judgments: Результаты судейства (опционально)

        Returns:
            Список результатов проверки фактов
        """
        # Определяем, какие тезисы требуют проверки
        claims_to_verify = self._select_claims_for_verification(
            claims,
            conflicts,
            judgments
        )

        evidence_records = []

        for claim in claims_to_verify:
            # Генерируем вопросы для проверки
            questions = self._generate_verification_questions(claim)

            # Для каждого вопроса ищем ответ
            for question in questions:
                evidence = self._verify_question(claim['id'], question, claim)
                evidence_records.append(evidence)

        return evidence_records

    def _select_claims_for_verification(
        self,
        claims: List[Dict],
        conflicts: List[Dict] = None,
        judgments: List[Dict] = None
    ) -> List[Dict]:
        """
        Отбор тезисов, требующих проверки

        Args:
            claims: Все тезисы
            conflicts: Конфликты
            judgments: Результаты судейства

        Returns:
            Список тезисов для проверки
        """
        claims_to_verify = []
        verify_ids = set()

        # Добавляем конфликтующие тезисы
        if conflicts:
            for conflict in conflicts:
                verify_ids.add(conflict['A_id'])
                verify_ids.add(conflict['B_id'])

        # Добавляем неопределённые тезисы из судейства
        if judgments:
            for judgment in judgments:
                if judgment.get('winner') == 'uncertain':
                    # Извлекаем ID из pair_id
                    pair_id = judgment['pair_id']
                    # Ищем соответствующие тезисы
                    # (предполагается, что pair_id связан с A_id и B_id)

        # Добавляем тезисы с фактами, но без доказательств
        for claim in claims:
            if claim.get('facts') and not claim.get('evidence_inline'):
                verify_ids.add(claim['id'])

        # Собираем отобранные тезисы
        claims_index = {c['id']: c for c in claims}
        for claim_id in verify_ids:
            if claim_id in claims_index:
                claims_to_verify.append(claims_index[claim_id])

        return claims_to_verify

    def _generate_verification_questions(self, claim: Dict) -> List[str]:
        """
        Генерация вопросов для проверки тезиса

        Args:
            claim: Тезис для проверки

        Returns:
            Список вопросов для поиска
        """
        questions = []

        # Извлекаем факты
        facts_str = claim.get('facts', '')
        if not facts_str:
            # Если факты не указаны, пытаемся их извлечь из текста
            facts_str = self._extract_facts_from_text(claim['claim'])

        if not facts_str:
            # Общий вопрос о тезисе
            questions.append(f"Подтверждается ли утверждение: {claim['claim'][:100]}?")
            return questions

        # Разбиваем факты
        facts = [f.strip() for f in facts_str.split(';') if f.strip()]

        for fact in facts[:5]:  # Ограничиваем до 5 вопросов
            # Очищаем факт от префикса "F#:"
            clean_fact = re.sub(r'^F\d+:\s*', '', fact)

            # Формируем вопрос
            if re.search(r'\d+', clean_fact):
                # Факт содержит числа
                questions.append(f"Каково точное значение показателя '{clean_fact}'?")
            else:
                # Общий факт
                questions.append(f"Подтверждается ли факт: {clean_fact}?")

        # Если есть время/юрисдикция, добавляем контекстные вопросы
        if claim.get('scope.time'):
            questions.append(
                f"Какие данные доступны за период {claim['scope.time']}?"
            )

        return questions

    def _extract_facts_from_text(self, text: str) -> str:
        """
        Извлечение фактов из текста (если не указаны явно)

        Args:
            text: Текст тезиса

        Returns:
            Строка с фактами
        """
        facts = []

        # Ищем числа
        numbers = re.finditer(r'(\d+(?:[.,]\d+)?%?)', text)
        for idx, match in enumerate(numbers, 1):
            facts.append(f"F{idx}: {match.group(1)}")
            if idx >= 5:
                break

        return '; '.join(facts)

    def _verify_question(
        self,
        claim_id: str,
        question: str,
        claim: Dict
    ) -> Dict:
        """
        Проверка одного вопроса через ReAct-цикл

        Реализует ReAct (Reasoning and Acting):
        1. Thought (Рассуждение): Анализ вопроса и формирование плана
        2. Action (Действие): Веб-поиск
        3. Observation (Наблюдение): Анализ результатов поиска
        4. Synthesis (Синтез): Формирование ответа

        Args:
            claim_id: ID тезиса
            question: Вопрос для проверки
            claim: Полный объект тезиса

        Returns:
            Запись о проверке с полями:
            - claim_id, question, status, source, date, quote, notes
        """
        self.verification_stats['total'] += 1

        # Если ReAct доступен, используем его
        if self.use_react and self.llm and self.search:
            try:
                result = self._verify_with_react(claim_id, question, claim)
                # Обновляем статистику
                status = result.get('status', 'uncertain')
                if status in self.verification_stats:
                    self.verification_stats[status] += 1
                return result
            except Exception as e:
                logger.error(f"ReAct verification failed: {e}. Falling back to heuristics")
                # Fallback на эвристики

        # Fallback: веб-поиск без LLM
        if self.search:
            try:
                return self._verify_with_search_only(claim_id, question, claim)
            except Exception as e:
                logger.error(f"Search-only verification failed: {e}")

        # Полный fallback: возвращаем uncertain
        self.verification_stats['uncertain'] += 1
        return {
            'claim_id': claim_id,
            'question': question,
            'status': 'uncertain',
            'source': '',
            'date': '',
            'quote': '',
            'notes': 'Верификация недоступна (нет API ключей или инструментов)'
        }

    def _verify_with_react(
        self,
        claim_id: str,
        question: str,
        claim: Dict
    ) -> Dict:
        """
        Верификация с использованием ReAct-цикла

        ReAct Steps:
        1. Thought: Формирование стратегии поиска
        2. Action: Выполнение веб-поиска
        3. Observation: Анализ результатов
        4. Thought: Оценка достаточности информации
        5. Synthesis: Формирование финального ответа

        Args:
            claim_id: ID тезиса
            question: Вопрос для проверки
            claim: Полный объект тезиса

        Returns:
            Результат верификации
        """
        logger.info(f"ReAct verification for: {question[:50]}...")

        # ========================================
        # Step 1: THOUGHT - Формирование стратегии
        # ========================================
        thought_prompt = f"""You are a fact-checker. Analyze this question and determine the best search strategy.

Question: {question}
Claim context: {claim.get('claim', '')[:200]}

Think step-by-step:
1. What specific information do we need to find?
2. What search query would be most effective?
3. What kind of sources would be most authoritative?

Provide:
- search_query: The optimal search query (one line)
- reasoning: Brief explanation of your strategy (2-3 sentences)
"""

        thought_response = self.llm.generate(thought_prompt, temperature=0.3, max_tokens=300)
        thought_text = thought_response.text

        # Извлекаем search query
        search_query = self._extract_search_query(thought_text, question)

        logger.info(f"ReAct Thought: Query = '{search_query}'")

        # ========================================
        # Step 2: ACTION - Веб-поиск
        # ========================================
        search_results = self.search.search(search_query, num_results=5)

        if not search_results:
            return {
                'claim_id': claim_id,
                'question': question,
                'status': 'uncertain',
                'source': '',
                'date': '',
                'quote': '',
                'notes': f'No search results found for query: {search_query}'
            }

        # ========================================
        # Step 3: OBSERVATION - Анализ результатов
        # ========================================
        # Формируем текст с результатами поиска
        search_context = self._format_search_results(search_results)

        observation_prompt = f"""You are analyzing search results to verify a claim.

Original question: {question}
Claim: {claim.get('claim', '')[:300]}

Search results:
{search_context}

Analyze the search results:
1. Do they support, refute, or are neutral to the claim?
2. What is the quality and authority of the sources?
3. Are there any conditions or caveats?

Provide your analysis in this format:
Status: [supported/refuted/uncertain/conditional]
Confidence: [high/medium/low]
Key finding: [One sentence summary]
Best source: [Title and URL of most authoritative source]
Quote: [Relevant quote from the source if available]
Reasoning: [2-3 sentences explaining your conclusion]
"""

        observation_response = self.llm.generate(observation_prompt, temperature=0.2, max_tokens=500)
        observation_text = observation_response.text

        logger.info(f"ReAct Observation: {observation_text[:100]}...")

        # ========================================
        # Step 4: SYNTHESIS - Парсинг ответа
        # ========================================
        result = self._parse_verification_result(
            observation_text,
            search_results,
            claim_id,
            question
        )

        logger.info(f"ReAct Result: {result['status']} ({result.get('notes', '')[:50]}...)")

        return result

    def _extract_search_query(self, thought_text: str, fallback_question: str) -> str:
        """
        Извлечение search query из thought response

        Args:
            thought_text: Ответ от LLM с рассуждением
            fallback_question: Fallback если не удалось извлечь

        Returns:
            Search query
        """
        # Ищем строку с search_query
        import re
        patterns = [
            r'search[_ ]query:\s*(.+?)(?:\n|$)',
            r'query:\s*(.+?)(?:\n|$)',
            r'"(.+?)"',
        ]

        for pattern in patterns:
            match = re.search(pattern, thought_text, re.IGNORECASE)
            if match:
                query = match.group(1).strip().strip('"\'')
                if query and len(query) > 5:
                    return query

        # Fallback: используем вопрос напрямую
        return fallback_question

    def _format_search_results(self, search_results, max_results: int = 5) -> str:
        """
        Форматирование результатов поиска для промпта

        Args:
            search_results: Список SearchResult
            max_results: Максимум результатов для включения

        Returns:
            Отформатированная строка
        """
        formatted = ""
        for i, result in enumerate(search_results[:max_results], 1):
            formatted += f"\n[Result {i}] {result.title}\n"
            formatted += f"URL: {result.url}\n"
            formatted += f"Source type: {result.source_type}\n"
            if result.date:
                formatted += f"Date: {result.date}\n"
            formatted += f"Content: {result.snippet}\n"
            formatted += "-" * 50 + "\n"

        return formatted

    def _parse_verification_result(
        self,
        observation_text: str,
        search_results,
        claim_id: str,
        question: str
    ) -> Dict:
        """
        Парсинг результата верификации из observation

        Args:
            observation_text: Ответ от LLM с анализом
            search_results: Результаты поиска
            claim_id: ID тезиса
            question: Вопрос

        Returns:
            Словарь с результатом верификации
        """
        import re

        # Извлекаем status
        status_match = re.search(r'Status:\s*(\w+)', observation_text, re.IGNORECASE)
        status = status_match.group(1).lower() if status_match else 'uncertain'

        # Нормализуем status
        if status not in ['supported', 'refuted', 'uncertain', 'conditional']:
            status = 'uncertain'

        # Извлекаем quote
        quote_match = re.search(r'Quote:\s*(.+?)(?:\n[A-Z]|\n$)', observation_text, re.IGNORECASE | re.DOTALL)
        quote = quote_match.group(1).strip() if quote_match else ''

        # Извлекаем reasoning
        reasoning_match = re.search(r'Reasoning:\s*(.+?)(?:\n\n|$)', observation_text, re.IGNORECASE | re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ''

        # Находим лучший источник (используем первый академический или первый вообще)
        best_source = None
        for result in search_results:
            if result.source_type in ['academic', 'government', 'peer_reviewed']:
                best_source = result
                break

        if not best_source and search_results:
            best_source = search_results[0]

        # Формируем результат
        return {
            'claim_id': claim_id,
            'question': question,
            'status': status,
            'source': f"{best_source.title} ({best_source.url})" if best_source else '',
            'date': best_source.date if best_source else '',
            'quote': quote[:200] if quote else (best_source.snippet[:200] if best_source else ''),
            'notes': reasoning[:300] if reasoning else observation_text[:300]
        }

    def _verify_with_search_only(
        self,
        claim_id: str,
        question: str,
        claim: Dict
    ) -> Dict:
        """
        Верификация только с веб-поиском (без LLM)

        Использует эвристики для анализа результатов поиска.

        Args:
            claim_id: ID тезиса
            question: Вопрос
            claim: Объект тезиса

        Returns:
            Результат верификации
        """
        logger.info(f"Search-only verification for: {question[:50]}...")

        # Веб-поиск
        search_results = self.search.search(question, num_results=5)

        if not search_results:
            return {
                'claim_id': claim_id,
                'question': question,
                'status': 'uncertain',
                'source': '',
                'date': '',
                'quote': '',
                'notes': 'No search results found'
            }

        # Используем старую логику analyze_source_content
        best_source = self._select_best_source(search_results)
        status = self._analyze_source_content(question, best_source.snippet)

        return {
            'claim_id': claim_id,
            'question': question,
            'status': status,
            'source': f"{best_source.title} ({best_source.url})",
            'date': best_source.date or '',
            'quote': best_source.snippet[:200],
            'notes': f'Verified using search heuristics. Source type: {best_source.source_type}'
        }

    def _select_best_source(self, search_results):
        """
        Выбор лучшего источника по приоритету

        Args:
            search_results: Список SearchResult

        Returns:
            Лучший SearchResult
        """
        # Сортируем по приоритету source_type
        sorted_results = sorted(
            search_results,
            key=lambda r: self.SOURCE_PRIORITY.get(r.source_type, 0),
            reverse=True
        )
        return sorted_results[0] if sorted_results else search_results[0]

    def verify_with_search(
        self,
        claim_id: str,
        question: str,
        search_results: List[Dict]
    ) -> Dict:
        """
        Проверка вопроса с использованием результатов веб-поиска

        Args:
            claim_id: ID тезиса
            question: Вопрос для проверки
            search_results: Результаты веб-поиска, каждый с полями:
                - 'title': заголовок
                - 'url': URL источника
                - 'snippet': фрагмент текста
                - 'date': дата публикации
                - 'source_type': тип источника

        Returns:
            Запись о проверке с результатом
        """
        if not search_results:
            return {
                'claim_id': claim_id,
                'question': question,
                'status': 'uncertain',
                'source': '',
                'date': '',
                'quote': '',
                'notes': 'Источники не найдены'
            }

        # Сортируем источники по приоритету
        sorted_results = sorted(
            search_results,
            key=lambda x: self.SOURCE_PRIORITY.get(x.get('source_type', 'other'), 0),
            reverse=True
        )

        # Берём лучший источник
        best_source = sorted_results[0]

        # Анализируем содержимое
        status = self._analyze_source_content(question, best_source['snippet'])

        return {
            'claim_id': claim_id,
            'question': question,
            'status': status,
            'source': f"{best_source['title']} ({best_source['url']})",
            'date': best_source.get('date', ''),
            'quote': best_source['snippet'][:200],  # Первые 200 символов
            'notes': f"Тип источника: {best_source.get('source_type', 'other')}"
        }

    def _analyze_source_content(self, question: str, content: str) -> str:
        """
        Анализ содержимого источника для определения статуса

        Args:
            question: Вопрос для проверки
            content: Содержимое источника

        Returns:
            Статус: supported / refuted / uncertain / conditional
        """
        # Это упрощённая эвристика
        # В реальной системе нужен более сложный анализ (возможно, с LLM)

        content_lower = content.lower()
        question_lower = question.lower()

        # Извлекаем ключевые слова из вопроса
        question_words = set(re.findall(r'\w+', question_lower))
        question_words = {w for w in question_words if len(w) > 3}

        # Проверяем наличие ключевых слов в контенте
        matches = sum(1 for word in question_words if word in content_lower)

        if matches >= len(question_words) * 0.5:
            # Достаточно совпадений
            # Проверяем на отрицание
            negation_words = ['нет', 'не', 'отсутств', 'опроверг', 'ложн']
            has_negation = any(neg in content_lower for neg in negation_words)

            if has_negation:
                return 'refuted'
            else:
                # Проверяем на условность
                conditional_words = ['если', 'при условии', 'в случае', 'когда']
                has_conditional = any(cond in content_lower for cond in conditional_words)

                if has_conditional:
                    return 'conditional'
                else:
                    return 'supported'
        else:
            return 'uncertain'

    def generate_prompt_for_llm(
        self,
        claim: Dict,
        questions: List[str]
    ) -> str:
        """
        Генерация промпта для LLM для проверки фактов

        Args:
            claim: Тезис для проверки
            questions: Список вопросов

        Returns:
            Сформированный промпт
        """
        questions_str = '\n'.join(f"{i+1}. {q}" for i, q in enumerate(questions))

        return self.get_prompt() + f"""

## Тезис для проверки

**ID**: {claim['id']}
**Утверждение**: {claim['claim']}
**Факты**: {claim.get('facts', 'не указаны')}
**Источник**: {claim.get('origin', '')}

## Вопросы для проверки

{questions_str}

Для каждого вопроса выполни веб-поиск и заполни таблицу evidence.tsv согласно инструкциям.
"""

    def filter_by_status(
        self,
        evidence_records: List[Dict],
        allowed_statuses: List[str] = None
    ) -> List[Dict]:
        """
        Фильтрация записей по статусу

        Args:
            evidence_records: Все записи о проверке
            allowed_statuses: Разрешённые статусы (по умолчанию: supported, conditional)

        Returns:
            Отфильтрованные записи
        """
        if allowed_statuses is None:
            allowed_statuses = ['supported', 'conditional']

        return [
            record for record in evidence_records
            if record['status'] in allowed_statuses
        ]

    def get_verified_claims(
        self,
        claims: List[Dict],
        evidence_records: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Разделение тезисов на подтверждённые и неподтверждённые

        Args:
            claims: Все тезисы
            evidence_records: Результаты проверки

        Returns:
            Кортеж: (подтверждённые тезисы, неподтверждённые тезисы)
        """
        # Группируем записи по claim_id
        evidence_by_claim = {}
        for record in evidence_records:
            claim_id = record['claim_id']
            if claim_id not in evidence_by_claim:
                evidence_by_claim[claim_id] = []
            evidence_by_claim[claim_id].append(record)

        verified_claims = []
        unverified_claims = []

        for claim in claims:
            claim_id = claim['id']

            if claim_id not in evidence_by_claim:
                # Не проверялся - оставляем как есть
                verified_claims.append(claim)
                continue

            # Проверяем статусы всех проверок
            records = evidence_by_claim[claim_id]
            statuses = [r['status'] for r in records]

            # Если хоть одна проверка опровергла - тезис отклоняется
            if 'refuted' in statuses:
                unverified_claims.append(claim)
            # Если все проверки подтверждают или условные - тезис принимается
            elif all(s in ['supported', 'conditional'] for s in statuses):
                verified_claims.append(claim)
            # Если есть неопределённые - помечаем как требующий дополнительной проверки
            else:
                claim_copy = claim.copy()
                claim_copy['notes'] = (claim.get('notes', '') +
                                      ' [требует дополнительной проверки]')
                unverified_claims.append(claim_copy)

        return verified_claims, unverified_claims

    def get_verification_stats(self) -> Dict:
        """
        Получить статистику верификации

        Returns:
            Словарь со статистикой
        """
        stats = self.verification_stats.copy()

        # Добавляем процентные соотношения
        total = stats['total']
        if total > 0:
            stats['supported_pct'] = (stats['supported'] / total) * 100
            stats['refuted_pct'] = (stats['refuted'] / total) * 100
            stats['uncertain_pct'] = (stats['uncertain'] / total) * 100
            stats['conditional_pct'] = (stats['conditional'] / total) * 100
        else:
            stats['supported_pct'] = 0
            stats['refuted_pct'] = 0
            stats['uncertain_pct'] = 0
            stats['conditional_pct'] = 0

        stats['mode'] = 'ReAct' if self.use_react else 'Heuristics'

        return stats

    def print_stats(self):
        """Вывод статистики верификации"""
        stats = self.get_verification_stats()

        print("=" * 50)
        print("VerifierAgent Statistics")
        print("=" * 50)
        print(f"Mode: {stats['mode']}")
        print(f"\nTotal verifications: {stats['total']}")
        print(f"Supported: {stats['supported']} ({stats['supported_pct']:.1f}%)")
        print(f"Refuted: {stats['refuted']} ({stats['refuted_pct']:.1f}%)")
        print(f"Uncertain: {stats['uncertain']} ({stats['uncertain_pct']:.1f}%)")
        print(f"Conditional: {stats['conditional']} ({stats['conditional_pct']:.1f}%)")
        print("=" * 50)


# Пример использования
if __name__ == "__main__":
    verifier = VerifierAgent()

    # Загрузка данных
    claims = verifier.load_tsv('output/claims.tsv')
    conflicts = verifier.load_json('output/conflicts.json')

    # Проверка фактов
    evidence = verifier.process(claims, conflicts)

    # Сохранение результатов
    verifier.save_result(evidence, 'output/evidence.tsv', format='tsv')

    # Разделение на подтверждённые/неподтверждённые
    verified, unverified = verifier.get_verified_claims(claims, evidence)

    print(f"Проверено {len(evidence)} фактов")
    print(f"Подтверждённых тезисов: {len(verified)}")
    print(f"Неподтверждённых тезисов: {len(unverified)}")
