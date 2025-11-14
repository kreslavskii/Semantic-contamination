"""
Агент для семантического сопоставления и типизации отношений между тезисами (Шаг 4)
"""
from typing import List, Dict, Tuple, Set
from .base_agent import BaseAgent
import re


class AlignerAgent(BaseAgent):
    """Агент для определения семантических отношений между тезисами"""

    def __init__(self, prompt_template_path: str = "prompts/aligner_prompt.md"):
        super().__init__(prompt_template_path)
        self.conflicts = []

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

        Args:
            text_a: Текст первого тезиса
            text_b: Текст второго тезиса
            claim_a: Полный объект первого тезиса
            claim_b: Полный объект второго тезиса

        Returns:
            Тип отношения: equivalent, refines, extends, contradicts, independent
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
            return 'contradicts'

        # Высокая схожесть - эквивалентность
        if jaccard > 0.7:
            return 'equivalent'

        # Один текст существенно длиннее и содержит другой
        if jaccard > 0.5:
            if len(words_a) > len(words_b) * 1.5:
                return 'refines'  # A детализирует B
            elif len(words_b) > len(words_a) * 1.5:
                return 'refines'  # B детализирует A

        # Средняя схожесть - дополнение
        if jaccard > 0.3:
            return 'extends'

        # Низкая схожесть - независимые
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
