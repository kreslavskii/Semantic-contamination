"""
Агент для извлечения тезисов из текстовых документов (Шаг 1)
"""
import os
import re
from typing import List, Dict
from .base_agent import BaseAgent


class ExtractorAgent(BaseAgent):
    """Агент для извлечения содержательных тезисов из документов"""

    def __init__(self, prompt_template_path: str = "prompts/extractor_prompt.md"):
        super().__init__(prompt_template_path)
        self.claim_counter = 1

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

        Это упрощённая версия. В полной реализации здесь должен быть
        вызов LLM с промптом для извлечения тезисов.

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
