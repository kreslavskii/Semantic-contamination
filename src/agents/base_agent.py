"""
Базовый класс для всех агентов системы сравнения и слияния текстов
"""
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import json


class BaseAgent(ABC):
    """Абстрактный базовый класс для агентов"""

    def __init__(self, prompt_template_path: str):
        """
        Инициализация агента

        Args:
            prompt_template_path: Путь к файлу с шаблоном промпта
        """
        self.prompt_template = self._load_prompt_template(prompt_template_path)

    def _load_prompt_template(self, path: str) -> str:
        """
        Загрузка шаблона промпта из файла

        Args:
            path: Путь к файлу с промптом

        Returns:
            Содержимое промпта
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл промпта не найден: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Основной метод обработки данных агентом

        Args:
            input_data: Входные данные для обработки

        Returns:
            Результат обработки
        """
        pass

    def get_prompt(self, **kwargs) -> str:
        """
        Формирование промпта с подстановкой параметров

        Args:
            **kwargs: Параметры для подстановки в шаблон

        Returns:
            Сформированный промпт
        """
        prompt = self.prompt_template
        for key, value in kwargs.items():
            placeholder = f"{{{{{key}}}}}"
            prompt = prompt.replace(placeholder, str(value))
        return prompt

    def save_result(self, data: Any, output_path: str, format: str = 'tsv'):
        """
        Сохранение результата в файл

        Args:
            data: Данные для сохранения
            output_path: Путь к выходному файлу
            format: Формат файла ('tsv', 'json', 'md')
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if format == 'tsv':
            self._save_tsv(data, output_path)
        elif format == 'json':
            self._save_json(data, output_path)
        elif format == 'md':
            self._save_md(data, output_path)
        else:
            raise ValueError(f"Неподдерживаемый формат: {format}")

    def _save_tsv(self, data: List[Dict], output_path: str):
        """Сохранение в TSV формате"""
        if not data:
            return

        with open(output_path, 'w', encoding='utf-8') as f:
            # Заголовки
            headers = list(data[0].keys())
            f.write('\t'.join(headers) + '\n')

            # Данные
            for row in data:
                values = [str(row.get(h, '')) for h in headers]
                f.write('\t'.join(values) + '\n')

    def _save_json(self, data: Any, output_path: str):
        """Сохранение в JSON формате"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_md(self, data: str, output_path: str):
        """Сохранение в Markdown формате"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(data)

    def load_tsv(self, path: str) -> List[Dict]:
        """
        Загрузка данных из TSV файла

        Args:
            path: Путь к TSV файлу

        Returns:
            Список словарей с данными
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл не найден: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            return []

        headers = lines[0].strip().split('\t')
        data = []

        for line in lines[1:]:
            if line.strip():
                values = line.strip().split('\t')
                row = dict(zip(headers, values))
                data.append(row)

        return data

    def load_json(self, path: str) -> Any:
        """
        Загрузка данных из JSON файла

        Args:
            path: Путь к JSON файлу

        Returns:
            Загруженные данные
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл не найден: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
