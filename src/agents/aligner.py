"""
Агент для семантического сопоставления и типизации отношений между тезисами (Шаг 4)

ОБНОВЛЕНО (Шаг 7):
- Добавлена LLM-based semantic matching вместо word overlap
- Интегрирован Concise Chain-of-Thought (CoT) для reasoning
- Сохранены эвристики как fallback
- 2-уровневая graceful degradation
"""
from typing import List, Dict, Tuple, Set, Optional
from .base_agent import BaseAgent
import re
import logging

# LLM imports (опциональные)
try:
    from ..llm import get_default_llm, LLMClient
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    get_default_llm = None
    LLMClient = None

from ..config import settings

logger = logging.getLogger(__name__)


class AlignerAgent(BaseAgent):
    """Агент для определения семантических отношений между тезисами"""

    def __init__(
        self,
        prompt_template_path: str = "prompts/aligner_prompt.md",
        llm_client: Optional['LLMClient'] = None,
        use_llm: Optional[bool] = None
    ):
        """
        Инициализация AlignerAgent

        Args:
            prompt_template_path: Путь к промпт-шаблону
            llm_client: LLM клиент (если None, создастся автоматически)
            use_llm: Использовать ли LLM для semantic matching (если None, берется из settings)
        """
        super().__init__(prompt_template_path)
        self.conflicts = []

        # Настройка режимов
        self.use_llm = use_llm if use_llm is not None else (HAS_LLM and settings.can_use_llm)

        # Уровень 1: LLM semantic matching (preferred)
        if self.use_llm and HAS_LLM:
            try:
                self.llm = llm_client or get_default_llm(temperature=0.2)
                logger.info("AlignerAgent: используется LLM для semantic matching")
            except Exception as e:
                logger.error(f"Не удалось инициализировать LLM: {e}")
                self.llm = None
                self.use_llm = False
        else:
            self.llm = None
            if not HAS_LLM:
                logger.warning("LLM недоступен (нет openai/anthropic)")

        # Уровень 2: Эвристики (fallback)
        logger.info("AlignerAgent: эвристики доступны как fallback")

        # Статистика
        self.alignment_stats = {
            'total_pairs': 0,
            'llm_analyzed': 0,
            'heuristic_analyzed': 0,
            'equivalent': 0,
            'refines': 0,
            'extends': 0,
            'contradicts': 0,
            'independent': 0
        }

    def process(self, pairs: List[Dict], claims: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Анализ пар тезисов и определение отношений

        Args:
            pairs: Список пар тезисов из файла pairs.tsv
            claims: Список тезисов из файла claims.tsv

        Returns:
            Кортеж: (обновлённые пары, список конфликтов)
        """
        # Создаём индекс тезисов для быстрого доступа
        claims_index = {claim['id']: claim for claim in claims}

        updated_pairs = []
        self.conflicts = []

        for pair in pairs:
            a_id = pair['A_id']
            b_id = pair['B_id']

            if a_id not in claims_index or b_id not in claims_index:
                continue

            claim_a = claims_index[a_id]
            claim_b = claims_index[b_id]

            # Анализируем отношение между тезисами
            analysis = self._analyze_pair(claim_a, claim_b, pair)

            # Обновляем запись пары
            updated_pair = {**pair, **analysis}
            updated_pairs.append(updated_pair)

            # Обновляем статистику
            self.alignment_stats['total_pairs'] += 1

            # Если есть конфликт, добавляем в список
            if analysis['relation'] == 'contradicts':
                self.conflicts.append({
                    'pair_id': pair['pair_id'],
                    'A_id': a_id,
                    'B_id': b_id,
                    'conflict_type': analysis['conflict_type'],
                    'description': analysis['rationale']
                })

        return updated_pairs, self.conflicts

    def _analyze_pair(self, claim_a: Dict, claim_b: Dict, pair: Dict) -> Dict:
        """
        Анализ отношения между двумя тезисами

        Args:
            claim_a: Первый тезис
            claim_b: Второй тезис
            pair: Информация о паре

        Returns:
            Словарь с результатами анализа
        """
        text_a = claim_a['claim']
        text_b = claim_b['claim']

        # Определяем тип отношения
        relation = self._determine_relation(text_a, text_b, claim_a, claim_b)

        # Определяем логический статус (NLI)
        nli = self._determine_nli(relation, text_a, text_b)

        # Формируем обоснование
        rationale = self._generate_rationale(relation, text_a, text_b, claim_a, claim_b)

        # Определяем тип конфликта (если есть)
        conflict_type = ''
        if relation == 'contradicts':
            conflict_type = self._classify_conflict(claim_a, claim_b)

        # Заметки об условиях
        condition_notes = self._extract_condition_notes(claim_a, claim_b)

        return {
            'relation': relation,
            'nli': nli,
            'rationale': rationale,
            'conflict_type': conflict_type,
            'condition_notes': condition_notes
        }

    def _determine_relation(
        self,
        text_a: str,
        text_b: str,
        claim_a: Dict,
        claim_b: Dict
    ) -> str:
        """
        Определение типа отношения между тезисами

        Использует 2-уровневую стратегию:
        1. LLM с Concise CoT (если доступно)
        2. Эвристики (fallback)

        Args:
            text_a: Текст первого тезиса
            text_b: Текст второго тезиса
            claim_a: Полный объект первого тезиса
            claim_b: Полный объект второго тезиса

        Returns:
            Тип отношения: equivalent, refines, extends, contradicts, independent
        """
        # Уровень 1: LLM semantic matching
        if self.use_llm and self.llm:
            try:
                return self._determine_relation_with_llm(text_a, text_b, claim_a, claim_b)
            except Exception as e:
                logger.error(f"Ошибка в LLM relation detection: {e}")
                logger.warning("Переключаемся на эвристики")

        # Уровень 2: Эвристики (fallback)
        return self._determine_relation_with_heuristics(text_a, text_b, claim_a, claim_b)

    def _determine_relation_with_llm(
        self,
        text_a: str,
        text_b: str,
        claim_a: Dict,
        claim_b: Dict
    ) -> str:
        """
        Определение отношения с использованием LLM + Concise CoT

        Args:
            text_a: Текст первого тезиса
            text_b: Текст второго тезиса
            claim_a: Полный объект первого тезиса
            claim_b: Полный объект второго тезиса

        Returns:
            Тип отношения
        """
        # Формируем Concise CoT промпт
        prompt = f"""Определи семантическое отношение между двумя утверждениями. Используй краткую цепочку рассуждений.

Утверждение A: {text_a}

Утверждение B: {text_b}

Дополнительный контекст:
- Факты A: {claim_a.get('facts', 'нет')}
- Факты B: {claim_b.get('facts', 'нет')}
- Условия A: {self._format_scope(claim_a)}
- Условия B: {self._format_scope(claim_b)}

Типы отношений:
- equivalent: утверждения выражают одно и то же (синонимы, перефразировки)
- refines: одно уточняет другое (добавляет детали, конкретизирует)
- extends: дополняют друг друга (разные аспекты одной темы)
- contradicts: противоречат друг другу (несовместимые утверждения)
- independent: не связаны по смыслу (разные темы)

Рассуждение (2-3 предложения):
1. Что общего между утверждениями?
2. В чем ключевое различие?
3. Какое отношение это означает?

Ответ (одно слово): [equivalent/refines/extends/contradicts/independent]"""

        response = self.llm.generate(prompt, temperature=0.2, max_tokens=300)

        # Парсим ответ LLM
        relation = self._parse_relation_from_llm(response.text)

        self.alignment_stats['llm_analyzed'] += 1
        self.alignment_stats[relation] += 1

        return relation

    def _determine_relation_with_heuristics(
        self,
        text_a: str,
        text_b: str,
        claim_a: Dict,
        claim_b: Dict
    ) -> str:
        """
        Fallback: определение отношения с использованием эвристик

        Args:
            text_a: Текст первого тезиса
            text_b: Текст второго тезиса
            claim_a: Полный объект первого тезиса
            claim_b: Полный объект второго тезиса

        Returns:
            Тип отношения
        """
        # Вычисляем схожесть по словам
        words_a = set(self._tokenize(text_a.lower()))
        words_b = set(self._tokenize(text_b.lower()))

        # Коэффициент Жаккара
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        jaccard = intersection / union if union > 0 else 0

        # Проверяем на противоречия в фактах
        if self._has_contradicting_facts(claim_a, claim_b):
            relation = 'contradicts'
        # Высокая схожесть - эквивалентность
        elif jaccard > 0.7:
            relation = 'equivalent'
        # Один текст существенно длиннее и содержит другой
        elif jaccard > 0.5:
            if len(words_a) > len(words_b) * 1.5 or len(words_b) > len(words_a) * 1.5:
                relation = 'refines'  # Один детализирует другой
            else:
                relation = 'extends'  # Оба дополняют
        # Средняя схожесть - дополнение
        elif jaccard > 0.3:
            relation = 'extends'
        # Низкая схожесть - независимые
        else:
            relation = 'independent'

        self.alignment_stats['heuristic_analyzed'] += 1
        self.alignment_stats[relation] += 1

        return relation

    def _format_scope(self, claim: Dict) -> str:
        """
        Форматирование scope для промпта

        Args:
            claim: Тезис

        Returns:
            Строка с условиями
        """
        parts = []
        if claim.get('scope.time'):
            parts.append(f"время: {claim['scope.time']}")
        if claim.get('scope.jurisdiction'):
            parts.append(f"область: {claim['scope.jurisdiction']}")
        if claim.get('scope.conditions'):
            parts.append(f"условия: {claim['scope.conditions']}")

        return ', '.join(parts) if parts else 'нет'

    def _parse_relation_from_llm(self, llm_response: str) -> str:
        """
        Парсинг типа отношения из ответа LLM

        Args:
            llm_response: Текст ответа от LLM

        Returns:
            Тип отношения
        """
        # Нормализуем ответ
        response_lower = llm_response.lower()

        # Ищем ключевые слова в порядке приоритета
        if 'equivalent' in response_lower:
            return 'equivalent'
        elif 'contradict' in response_lower:
            return 'contradicts'
        elif 'refine' in response_lower:
            return 'refines'
        elif 'extend' in response_lower:
            return 'extends'
        elif 'independent' in response_lower:
            return 'independent'

        # Если не нашли явного указания, берем последнее слово
        last_line = llm_response.strip().split('\n')[-1].lower()
        for relation in ['equivalent', 'contradicts', 'refines', 'extends', 'independent']:
            if relation in last_line:
                return relation

        # Fallback - independent
        logger.warning(f"Не удалось распарсить relation из LLM: {llm_response}")
        return 'independent'

    def _tokenize(self, text: str) -> List[str]:
        """
        Токенизация текста

        Args:
            text: Входной текст

        Returns:
            Список токенов
        """
        # Удаляем пунктуацию и разбиваем на слова
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        # Фильтруем короткие слова (предлоги и т.п.)
        return [t for t in tokens if len(t) > 2]

    def _has_contradicting_facts(self, claim_a: Dict, claim_b: Dict) -> bool:
        """
        Проверка на противоречащие факты

        Args:
            claim_a: Первый тезис
            claim_b: Второй тезис

        Returns:
            True, если есть противоречия в фактах
        """
        facts_a = self._extract_numbers(claim_a['claim'])
        facts_b = self._extract_numbers(claim_b['claim'])

        if not facts_a or not facts_b:
            return False

        # Проверяем, есть ли одинаковые числа с разным контекстом
        # Это упрощённая эвристика
        for num_a in facts_a:
            for num_b in facts_b:
                # Если числа близки, но не равны (потенциальное противоречие)
                if abs(num_a - num_b) / max(num_a, num_b) < 0.1 and num_a != num_b:
                    return True

        return False

    def _extract_numbers(self, text: str) -> List[float]:
        """
        Извлечение чисел из текста

        Args:
            text: Текст для анализа

        Returns:
            Список чисел
        """
        numbers = []
        # Находим числа (включая проценты и десятичные)
        matches = re.finditer(r'(\d+(?:[.,]\d+)?)', text)
        for match in matches:
            try:
                num_str = match.group(1).replace(',', '.')
                numbers.append(float(num_str))
            except ValueError:
                continue
        return numbers

    def _determine_nli(self, relation: str, text_a: str, text_b: str) -> str:
        """
        Определение логического статуса (NLI)

        Args:
            relation: Тип отношения
            text_a: Текст первого тезиса
            text_b: Текст второго тезиса

        Returns:
            Логический статус: entails, contradicts, neutral
        """
        if relation == 'contradicts':
            return 'contradicts'

        if relation == 'equivalent':
            return 'entails'

        if relation == 'refines':
            # Уточнение обычно влечёт базовое утверждение
            return 'entails'

        return 'neutral'

    def _generate_rationale(
        self,
        relation: str,
        text_a: str,
        text_b: str,
        claim_a: Dict,
        claim_b: Dict
    ) -> str:
        """
        Генерация обоснования выбранного отношения

        Args:
            relation: Тип отношения
            text_a: Текст первого тезиса
            text_b: Текст второго тезиса
            claim_a: Полный объект первого тезиса
            claim_b: Полный объект второго тезиса

        Returns:
            Текст обоснования
        """
        if relation == 'equivalent':
            return f"Тезисы {claim_a['id']} и {claim_b['id']} выражают одно утверждение разными словами"

        elif relation == 'refines':
            return f"Один тезис уточняет другой с дополнительными деталями"

        elif relation == 'extends':
            return f"Тезисы дополняют друг друга новыми аспектами"

        elif relation == 'contradicts':
            facts_a = self._extract_numbers(text_a)
            facts_b = self._extract_numbers(text_b)
            if facts_a and facts_b:
                return f"Противоречие в числовых данных: {facts_a} vs {facts_b}"
            else:
                return "Тезисы содержат противоречащие утверждения"

        else:  # independent
            return "Тезисы не связаны по смыслу"

    def _classify_conflict(self, claim_a: Dict, claim_b: Dict) -> str:
        """
        Классификация типа конфликта

        Args:
            claim_a: Первый тезис
            claim_b: Второй тезис

        Returns:
            Тип конфликта: true_conflict, apparent_conflict, pseudo_conflict
        """
        # Проверяем условия применимости
        scope_a = (
            claim_a.get('scope.time', ''),
            claim_a.get('scope.jurisdiction', ''),
            claim_a.get('scope.conditions', '')
        )
        scope_b = (
            claim_b.get('scope.time', ''),
            claim_b.get('scope.jurisdiction', ''),
            claim_b.get('scope.conditions', '')
        )

        # Если условия разные, это кажущийся конфликт
        if any(a != b and a and b for a, b in zip(scope_a, scope_b)):
            return 'apparent_conflict'

        # Проверяем терминологию
        if self._has_terminology_differences(claim_a['claim'], claim_b['claim']):
            return 'pseudo_conflict'

        # Иначе - истинный конфликт
        return 'true_conflict'

    def _has_terminology_differences(self, text_a: str, text_b: str) -> bool:
        """
        Проверка на различия в терминологии

        Args:
            text_a: Текст первого тезиса
            text_b: Текст второго тезиса

        Returns:
            True, если различия только терминологические
        """
        # Упрощённая эвристика: если тексты похожи по структуре,
        # но отличаются ключевыми терминами
        words_a = set(self._tokenize(text_a.lower()))
        words_b = set(self._tokenize(text_b.lower()))

        # Общие слова
        common = words_a & words_b
        # Если большинство слов общие, это может быть терминологическое различие
        return len(common) / len(words_a | words_b) > 0.6

    def _extract_condition_notes(self, claim_a: Dict, claim_b: Dict) -> str:
        """
        Извлечение заметок об условиях применимости

        Args:
            claim_a: Первый тезис
            claim_b: Второй тезис

        Returns:
            Заметки об условиях
        """
        notes = []

        # Собираем условия из обоих тезисов
        for claim, label in [(claim_a, 'A'), (claim_b, 'B')]:
            conditions = []
            if claim.get('scope.time'):
                conditions.append(f"время: {claim['scope.time']}")
            if claim.get('scope.jurisdiction'):
                conditions.append(f"область: {claim['scope.jurisdiction']}")
            if claim.get('scope.conditions'):
                conditions.append(f"условия: {claim['scope.conditions']}")

            if conditions:
                notes.append(f"{label}: {', '.join(conditions)}")

        return '; '.join(notes) if notes else ''

    def generate_conflicts_md(self) -> str:
        """
        Генерация Markdown документа с конфликтами

        Returns:
            Содержимое файла conflicts.md
        """
        if not self.conflicts:
            return "# Конфликты\n\nКонфликтов не обнаружено."

        md = "# Конфликты между тезисами\n\n"

        for conflict in self.conflicts:
            md += f"## Пара {conflict['pair_id']}\n\n"
            md += f"- **Тезисы**: {conflict['A_id']} vs {conflict['B_id']}\n"
            md += f"- **Тип конфликта**: {conflict['conflict_type']}\n"
            md += f"- **Описание**: {conflict['description']}\n\n"

        return md

    def generate_prompt_for_llm(self, claim_a: Dict, claim_b: Dict) -> str:
        """
        Генерация промпта для LLM для анализа пары

        Args:
            claim_a: Первый тезис
            claim_b: Второй тезис

        Returns:
            Сформированный промпт
        """
        return self.get_prompt() + f"""

## Пара для анализа

**Тезис A (ID: {claim_a['id']})**
{claim_a['claim']}

Факты: {claim_a.get('facts', '')}
Условия: time={claim_a.get('scope.time', '')}, jurisdiction={claim_a.get('scope.jurisdiction', '')}, conditions={claim_a.get('scope.conditions', '')}

**Тезис B (ID: {claim_b['id']})**
{claim_b['claim']}

Факты: {claim_b.get('facts', '')}
Условия: time={claim_b.get('scope.time', '')}, jurisdiction={claim_b.get('scope.jurisdiction', '')}, conditions={claim_b.get('scope.conditions', '')}

Выполни анализ согласно инструкциям выше.
"""

    def get_alignment_stats(self) -> Dict:
        """
        Получить статистику alignment

        Returns:
            Словарь со статистикой использования разных методов
        """
        stats = self.alignment_stats.copy()

        total = stats['total_pairs']
        if total > 0:
            stats['llm_pct'] = (stats['llm_analyzed'] / total) * 100
            stats['heuristic_pct'] = (stats['heuristic_analyzed'] / total) * 100

            # Процентное распределение отношений
            stats['equivalent_pct'] = (stats['equivalent'] / total) * 100
            stats['refines_pct'] = (stats['refines'] / total) * 100
            stats['extends_pct'] = (stats['extends'] / total) * 100
            stats['contradicts_pct'] = (stats['contradicts'] / total) * 100
            stats['independent_pct'] = (stats['independent'] / total) * 100

        # Информация о режиме
        if self.use_llm and self.llm:
            stats['mode'] = 'LLM + Concise CoT + Heuristics'
        else:
            stats['mode'] = 'Heuristics only'

        return stats


# Пример использования
if __name__ == "__main__":
    aligner = AlignerAgent()

    # Загрузка данных
    pairs = aligner.load_tsv('output/pairs.tsv')
    claims = aligner.load_tsv('output/claims.tsv')

    # Анализ пар
    updated_pairs, conflicts = aligner.process(pairs, claims)

    # Сохранение результатов
    aligner.save_result(updated_pairs, 'output/pairs_aligned.tsv', format='tsv')

    # Сохранение конфликтов
    conflicts_md = aligner.generate_conflicts_md()
    aligner.save_result(conflicts_md, 'output/conflicts.md', format='md')

    print(f"Проанализировано {len(updated_pairs)} пар")
    print(f"Обнаружено {len(conflicts)} конфликтов")
