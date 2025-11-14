"""
Агент для фактчекинга через веб-поиск (Шаг 6) - Chain-of-Verification
"""
import re
from typing import List, Dict, Tuple
from .base_agent import BaseAgent


class VerifierAgent(BaseAgent):
    """Агент для проверки фактов с помощью внешних источников"""

    SOURCE_PRIORITY = {
        'peer_reviewed': 3,  # Рецензируемые публикации
        'official': 2,       # Официальные отчёты
        'expert': 1,         # Экспертные источники
        'other': 0          # Прочие
    }

    def __init__(self, prompt_template_path: str = "prompts/verifier_prompt.md"):
        super().__init__(prompt_template_path)

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
        Проверка одного вопроса (заглушка для реальной реализации с веб-поиском)

        В реальной реализации здесь должен быть вызов веб-поиска и анализ результатов.

        Args:
            claim_id: ID тезиса
            question: Вопрос для проверки
            claim: Полный объект тезиса

        Returns:
            Запись о проверке
        """
        # Это упрощённая заглушка
        # В реальной системе здесь должен быть веб-поиск и анализ источников

        return {
            'claim_id': claim_id,
            'question': question,
            'status': 'uncertain',  # supported / refuted / uncertain / conditional
            'source': '',
            'date': '',
            'quote': '',
            'notes': 'Требуется ручная проверка с веб-поиском'
        }

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
