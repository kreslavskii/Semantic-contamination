"""
Агент для извлечения тезисов из текстовых документов (Шаг 1)

ОБНОВЛЕНО (Шаг 6):
- Добавлена LLM-based extraction вместо regex
- Интегрирован SelfCheckGPT для детекции hallucinations
- Multiple sampling для consistency checking
- Сохранены эвристики как fallback
"""
import os
import re
import logging
from typing import List, Dict, Optional, Tuple
from collections import Counter
from .base_agent import BaseAgent

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


class ExtractorAgent(BaseAgent):
    """Агент для извлечения содержательных тезисов из документов"""

    def __init__(
        self,
        prompt_template_path: str = "prompts/extractor_prompt.md",
        llm_client: Optional['LLMClient'] = None,
        use_llm: Optional[bool] = None,
        use_selfcheck: Optional[bool] = None,
        selfcheck_samples: Optional[int] = None
    ):
        """
        Инициализация ExtractorAgent

        Args:
            prompt_template_path: Путь к промпт-шаблону
            llm_client: LLM клиент (если None, создастся автоматически)
            use_llm: Использовать ли LLM для extraction (если None, берется из settings)
            use_selfcheck: Использовать ли SelfCheckGPT (если None, берется из settings)
            selfcheck_samples: Количество samples для SelfCheck (если None, берется из settings)
        """
        super().__init__(prompt_template_path)
        self.claim_counter = 1

        # Настройка режимов
        self.use_llm = use_llm if use_llm is not None else (HAS_LLM and settings.can_use_llm)
        self.use_selfcheck = use_selfcheck if use_selfcheck is not None else settings.USE_SELFCHECK
        self.selfcheck_samples = selfcheck_samples if selfcheck_samples is not None else settings.SELFCHECK_SAMPLES

        # Уровень 1: LLM extraction (preferred)
        if self.use_llm and HAS_LLM:
            try:
                self.llm = llm_client or get_default_llm(temperature=0.3)
                logger.info("ExtractorAgent: используется LLM для extraction")

                if self.use_selfcheck:
                    logger.info(f"ExtractorAgent: SelfCheckGPT включен ({self.selfcheck_samples} samples)")
            except Exception as e:
                logger.error(f"Не удалось инициализировать LLM: {e}")
                self.llm = None
                self.use_llm = False
                self.use_selfcheck = False
        else:
            self.llm = None
            if not HAS_LLM:
                logger.warning("LLM недоступен (нет openai/anthropic)")

        # Уровень 2: Эвристики (fallback)
        logger.info("ExtractorAgent: эвристики доступны как fallback")

        # Статистика
        self.extraction_stats = {
            'total_claims': 0,
            'llm_extracted': 0,
            'heuristic_extracted': 0,
            'selfcheck_verified': 0,
            'hallucination_flagged': 0
        }

    def process(self, documents: List[Dict[str, str]]) -> List[Dict]:
        """
        Извлечение тезисов из документов

        Args:
            documents: Список документов, каждый с полями:
                - 'path': путь к файлу
                - 'name': имя документа
                - 'content': содержимое (опционально)

        Returns:
            Список тезисов в виде словарей
        """
        all_claims = []

        for doc in documents:
            content = doc.get('content')
            if not content:
                # Загружаем содержимое из файла
                with open(doc['path'], 'r', encoding='utf-8') as f:
                    content = f.read()

            doc_name = doc.get('name', os.path.basename(doc['path']))
            claims = self._extract_from_document(content, doc_name)
            all_claims.extend(claims)

        # Обновляем общую статистику
        self.extraction_stats['total_claims'] = len(all_claims)

        return all_claims

    def _extract_from_document(self, content: str, doc_name: str) -> List[Dict]:
        """
        Извлечение тезисов из одного документа

        Args:
            content: Содержимое документа
            doc_name: Название документа

        Returns:
            Список тезисов
        """
        # Разбиваем на секции
        sections = self._split_into_sections(content)

        claims = []
        section_number = 1

        for section in sections:
            # Пропускаем слишком короткие секции
            if len(section['content'].split()) < 20:
                continue

            # Извлекаем тезисы из секции
            section_claims = self._extract_claims_from_section(
                section['content'],
                doc_name,
                section['title'],
                section_number
            )
            claims.extend(section_claims)
            section_number += 1

        return claims

    def _split_into_sections(self, content: str) -> List[Dict]:
        """
        Разбивка документа на секции по заголовкам

        Args:
            content: Содержимое документа

        Returns:
            Список секций с заголовками
        """
        # Ищем заголовки Markdown
        lines = content.split('\n')
        sections = []
        current_section = {'title': '', 'content': ''}

        for line in lines:
            # Проверка на заголовок
            if line.startswith('#'):
                # Сохраняем предыдущую секцию
                if current_section['content'].strip():
                    sections.append(current_section)

                # Начинаем новую секцию
                title = line.lstrip('#').strip()
                current_section = {'title': title, 'content': ''}
            else:
                current_section['content'] += line + '\n'

        # Добавляем последнюю секцию
        if current_section['content'].strip():
            sections.append(current_section)

        # Если нет заголовков, весь документ - одна секция
        if not sections:
            sections = [{'title': 'Main', 'content': content}]

        return sections

    def _extract_claims_from_section(
        self,
        section_content: str,
        doc_name: str,
        section_title: str,
        section_number: int
    ) -> List[Dict]:
        """
        Извлечение тезисов из секции

        Использует 2-уровневую стратегию:
        1. LLM extraction + SelfCheckGPT (если доступно)
        2. Эвристики (fallback)

        Args:
            section_content: Содержимое секции
            doc_name: Название документа
            section_title: Заголовок секции
            section_number: Номер секции

        Returns:
            Список тезисов
        """
        # Уровень 1: LLM extraction
        if self.use_llm and self.llm:
            try:
                return self._extract_with_llm(
                    section_content,
                    doc_name,
                    section_title,
                    section_number
                )
            except Exception as e:
                logger.error(f"Ошибка в LLM extraction: {e}")
                logger.warning("Переключаемся на эвристики")

        # Уровень 2: Эвристики (fallback)
        return self._extract_with_heuristics(
            section_content,
            doc_name,
            section_title,
            section_number
        )

    def _extract_with_llm(
        self,
        section_content: str,
        doc_name: str,
        section_title: str,
        section_number: int
    ) -> List[Dict]:
        """
        Извлечение тезисов с использованием LLM + SelfCheckGPT

        Args:
            section_content: Содержимое секции
            doc_name: Название документа
            section_title: Заголовок секции
            section_number: Номер секции

        Returns:
            Список тезисов с проверкой consistency
        """
        claims = []

        # Промпт для извлечения тезисов
        extraction_prompt = f"""Извлеки ключевые фактические утверждения из следующего текста.

Документ: {doc_name}
Раздел: {section_title}

Текст:
{section_content[:2000]}  # Ограничиваем длину

Инструкции:
1. Извлеки 2-5 ключевых фактических утверждений
2. Каждое утверждение должно быть самодостаточным
3. Включай конкретные факты, числа, даты
4. Избегай общих утверждений

Формат ответа (каждое утверждение с новой строки):
1. [Первое утверждение]
2. [Второе утверждение]
...
"""

        # Генерируем множественные samples для SelfCheck
        samples = []
        num_samples = self.selfcheck_samples if self.use_selfcheck else 1

        for i in range(num_samples):
            try:
                # Temperature немного выше для разнообразия samples
                temp = 0.3 if i == 0 else 0.5
                response = self.llm.generate(extraction_prompt, temperature=temp, max_tokens=500)
                extracted_claims = self._parse_llm_claims(response.text)
                samples.append(extracted_claims)
            except Exception as e:
                logger.error(f"Ошибка в sample {i+1}: {e}")
                continue

        if not samples:
            raise ValueError("Не удалось получить ни одного sample")

        # Берем первый sample как основу
        base_claims = samples[0]

        # Если SelfCheck включен, проверяем consistency
        if self.use_selfcheck and len(samples) > 1:
            for claim_text in base_claims:
                # Проверяем consistency с другими samples
                consistency_score = self._calculate_consistency(claim_text, samples)

                # Создаем claim с метаданными
                claim = self._create_claim_with_selfcheck(
                    claim_text,
                    doc_name,
                    section_title,
                    section_number,
                    consistency_score
                )
                claims.append(claim)

                self.extraction_stats['llm_extracted'] += 1
                if consistency_score >= 0.7:
                    self.extraction_stats['selfcheck_verified'] += 1
                else:
                    self.extraction_stats['hallucination_flagged'] += 1
        else:
            # Без SelfCheck, просто создаем claims
            for claim_text in base_claims:
                claim = self._create_claim(
                    claim_text,
                    doc_name,
                    section_title,
                    section_number,
                    1  # para_number
                )
                claims.append(claim)
                self.extraction_stats['llm_extracted'] += 1

        return claims

    def _extract_with_heuristics(
        self,
        section_content: str,
        doc_name: str,
        section_title: str,
        section_number: int
    ) -> List[Dict]:
        """
        Fallback: извлечение с использованием эвристик

        Args:
            section_content: Содержимое секции
            doc_name: Название документа
            section_title: Заголовок секции
            section_number: Номер секции

        Returns:
            Список тезисов
        """
        claims = []

        # Разбиваем на абзацы
        paragraphs = [p.strip() for p in section_content.split('\n\n') if p.strip()]

        for para_idx, paragraph in enumerate(paragraphs, 1):
            # Пропускаем слишком короткие абзацы
            if len(paragraph.split()) < 10:
                continue

            # Проверяем, содержит ли абзац содержательные утверждения
            if self._is_substantive(paragraph):
                claim = self._create_claim(
                    paragraph,
                    doc_name,
                    section_title,
                    section_number,
                    para_idx
                )
                claims.append(claim)
                self.extraction_stats['heuristic_extracted'] += 1

        return claims

    def _is_substantive(self, text: str) -> bool:
        """
        Проверка, является ли текст содержательным

        Args:
            text: Проверяемый текст

        Returns:
            True, если текст содержательный
        """
        # Исключаем вводные фразы
        intro_patterns = [
            r'^(введение|заключение|summary|conclusion)',
            r'^(например|к примеру|for example)',
            r'^(таким образом|thus|therefore)',
        ]

        text_lower = text.lower()
        for pattern in intro_patterns:
            if re.match(pattern, text_lower):
                return False

        # Проверяем наличие фактов (числа, даты, имена)
        has_numbers = bool(re.search(r'\d+', text))
        has_caps = bool(re.search(r'[А-ЯA-Z][а-яa-z]+', text))

        # Содержательный текст обычно содержит факты
        return has_numbers or has_caps

    def _create_claim(
        self,
        text: str,
        doc_name: str,
        section_title: str,
        section_number: int,
        para_number: int
    ) -> Dict:
        """
        Создание записи тезиса

        Args:
            text: Текст тезиса
            doc_name: Название документа
            section_title: Заголовок секции
            section_number: Номер секции
            para_number: Номер абзаца

        Returns:
            Словарь с полями тезиса
        """
        claim_id = f"C{self.claim_counter:03d}"
        self.claim_counter += 1

        # Извлекаем факты (простая эвристика)
        facts = self._extract_facts(text)

        # Формируем origin
        origin = f"{doc_name}"
        if section_title:
            origin += f", раздел '{section_title}'"
        origin += f" (секция {section_number}, абзац {para_number})"

        return {
            'id': claim_id,
            'claim': text[:500],  # Ограничиваем длину
            'evidence_inline': '',
            'scope.time': '',
            'scope.jurisdiction': '',
            'scope.conditions': '',
            'facts': '; '.join(facts) if facts else '',
            'origin': origin,
            'notes': ''
        }

    def _parse_llm_claims(self, llm_response: str) -> List[str]:
        """
        Парсинг тезисов из ответа LLM

        Args:
            llm_response: Текст ответа от LLM

        Returns:
            Список извлеченных тезисов
        """
        claims = []

        # Ищем пронумерованные элементы
        lines = llm_response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Убираем номер в начале (1., 2., 1), 2), -)
            match = re.match(r'^[\d\-\*]+[\.\)]\s*(.+)$', line)
            if match:
                claim_text = match.group(1).strip()
                if len(claim_text.split()) >= 5:  # Минимум 5 слов
                    claims.append(claim_text)

        return claims

    def _calculate_consistency(
        self,
        claim_text: str,
        all_samples: List[List[str]]
    ) -> float:
        """
        Вычисление consistency score для SelfCheckGPT

        Метод: Подсчет, в скольких samples встречается похожее утверждение

        Args:
            claim_text: Текст тезиса для проверки
            all_samples: Список всех samples (каждый sample = список тезисов)

        Returns:
            Consistency score (0-1)
        """
        if len(all_samples) <= 1:
            return 1.0  # Если только один sample, считаем consistent

        # Подсчитываем, в скольких samples есть похожий claim
        matches = 0

        # Извлекаем ключевые слова из claim
        claim_keywords = self._extract_keywords(claim_text)

        for sample in all_samples:
            # Проверяем, есть ли в этом sample похожий claim
            for sample_claim in sample:
                sample_keywords = self._extract_keywords(sample_claim)

                # Вычисляем overlap ключевых слов
                overlap = len(claim_keywords & sample_keywords)
                total = len(claim_keywords | sample_keywords)

                if total > 0 and (overlap / total) >= 0.5:
                    matches += 1
                    break  # Нашли похожий в этом sample

        # Consistency = доля samples с похожим claim
        return matches / len(all_samples)

    def _extract_keywords(self, text: str) -> set:
        """
        Извлечение ключевых слов из текста для сравнения

        Args:
            text: Текст для анализа

        Returns:
            Множество ключевых слов (lowercase)
        """
        # Удаляем стоп-слова и берем существенные слова
        stop_words = {
            'и', 'в', 'на', 'с', 'по', 'для', 'к', 'о', 'от', 'из',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'
        }

        words = re.findall(r'\w+', text.lower())
        keywords = {w for w in words if w not in stop_words and len(w) > 3}

        return keywords

    def _create_claim_with_selfcheck(
        self,
        text: str,
        doc_name: str,
        section_title: str,
        section_number: int,
        consistency_score: float
    ) -> Dict:
        """
        Создание claim с SelfCheck метаданными

        Args:
            text: Текст тезиса
            doc_name: Название документа
            section_title: Заголовок секции
            section_number: Номер секции
            consistency_score: Score consistency (0-1)

        Returns:
            Словарь с полями тезиса + SelfCheck метаданные
        """
        claim = self._create_claim(text, doc_name, section_title, section_number, 1)

        # Добавляем SelfCheck метаданные в notes
        if consistency_score < 0.7:
            hallucination_warning = f"⚠️ LOW CONSISTENCY ({consistency_score:.2f}) - possible hallucination"
            claim['notes'] = hallucination_warning
        else:
            claim['notes'] = f"✓ Verified by SelfCheck (consistency: {consistency_score:.2f})"

        return claim

    def _extract_facts(self, text: str) -> List[str]:
        """
        Извлечение фактов из текста (упрощённая версия)

        Args:
            text: Текст для анализа

        Returns:
            Список идентификаторов фактов
        """
        facts = []
        fact_counter = 1

        # Ищем числа с контекстом
        numbers = re.finditer(r'(\d+(?:[.,]\d+)?%?)', text)
        for match in numbers:
            facts.append(f"F{fact_counter}: {match.group(1)}")
            fact_counter += 1
            if fact_counter > 5:  # Ограничиваем количество
                break

        return facts

    def generate_prompt_for_llm(self, section_content: str, doc_name: str) -> str:
        """
        Генерация промпта для LLM для извлечения тезисов

        Args:
            section_content: Содержимое секции
            doc_name: Название документа

        Returns:
            Сформированный промпт
        """
        return self.get_prompt(
            document_name=doc_name,
            document_content=section_content
        ) + f"\n\n## Документ для анализа\n\n{section_content}"

    def get_extraction_stats(self) -> Dict:
        """
        Получить статистику extraction

        Returns:
            Словарь со статистикой использования разных методов
        """
        stats = self.extraction_stats.copy()

        total = stats['total_claims']
        if total > 0:
            stats['llm_pct'] = (stats['llm_extracted'] / total) * 100
            stats['heuristic_pct'] = (stats['heuristic_extracted'] / total) * 100

            if stats['selfcheck_verified'] > 0:
                stats['verification_rate'] = (stats['selfcheck_verified'] / stats['llm_extracted']) * 100
                stats['hallucination_rate'] = (stats['hallucination_flagged'] / stats['llm_extracted']) * 100

        # Информация о режиме
        if self.use_llm and self.llm:
            if self.use_selfcheck:
                stats['mode'] = f'LLM + SelfCheck({self.selfcheck_samples} samples) + Heuristics'
            else:
                stats['mode'] = 'LLM + Heuristics'
        else:
            stats['mode'] = 'Heuristics only'

        return stats


# Пример использования
if __name__ == "__main__":
    extractor = ExtractorAgent()

    # Пример документов
    documents = [
        {'path': 'data/doc1.md', 'name': 'Документ A'},
        {'path': 'data/doc2.md', 'name': 'Документ B'},
    ]

    # Извлечение тезисов
    claims = extractor.process(documents)

    # Сохранение результата
    extractor.save_result(claims, 'output/claims.tsv', format='tsv')

    print(f"Извлечено {len(claims)} тезисов")
