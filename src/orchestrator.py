"""
Оркестратор процесса сравнения и слияния текстов
Координирует работу всех агентов
"""
import os
import json
from typing import List, Dict, Optional
from agents import ExtractorAgent, AlignerAgent, JudgeAgent, VerifierAgent


class TextMergeOrchestrator:
    """Главный оркестратор процесса"""

    def __init__(
        self,
        input_dir: str = 'data',
        output_dir: str = 'output',
        prompts_dir: str = 'prompts'
    ):
        """
        Инициализация оркестратора

        Args:
            input_dir: Директория с входными документами
            output_dir: Директория для результатов
            prompts_dir: Директория с шаблонами промптов
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.prompts_dir = prompts_dir

        # Создаём директории
        os.makedirs(output_dir, exist_ok=True)

        # Инициализируем агентов
        self.extractor = ExtractorAgent(f"{prompts_dir}/extractor_prompt.md")
        self.aligner = AlignerAgent(f"{prompts_dir}/aligner_prompt.md")
        self.judge = JudgeAgent(f"{prompts_dir}/judge_prompt.md")
        self.verifier = VerifierAgent(f"{prompts_dir}/verifier_prompt.md")

    def run_pipeline(
        self,
        documents: List[Dict[str, str]],
        skip_steps: List[int] = None
    ) -> Dict:
        """
        Запуск полного пайплайна обработки

        Args:
            documents: Список документов для обработки
            skip_steps: Список номеров шагов, которые нужно пропустить

        Returns:
            Словарь с результатами всех этапов
        """
        skip_steps = skip_steps or []
        results = {}

        print("=== Начало обработки ===\n")

        # Шаг 1: Экстракция тезисов
        if 1 not in skip_steps:
            print("Шаг 1: Экстракция тезисов...")
            claims = self._step1_extract_claims(documents)
            results['claims'] = claims
            print(f"✓ Извлечено {len(claims)} тезисов\n")
        else:
            print("Шаг 1: Пропущен")
            claims = self._load_claims()
            results['claims'] = claims

        # Шаг 2: Нормализация терминов (TODO: реализовать)
        # Шаг 3: Формирование пар (TODO: реализовать или создать вручную)

        # Шаг 4: Семантическое сопоставление
        if 4 not in skip_steps:
            print("Шаг 4: Семантическое сопоставление...")
            pairs = self._load_or_create_pairs(claims)
            aligned_pairs, conflicts = self._step4_align_pairs(pairs, claims)
            results['pairs'] = aligned_pairs
            results['conflicts'] = conflicts
            print(f"✓ Проанализировано {len(aligned_pairs)} пар")
            print(f"✓ Обнаружено {len(conflicts)} конфликтов\n")
        else:
            print("Шаг 4: Пропущен")
            aligned_pairs = self._load_pairs()
            conflicts = []
            results['pairs'] = aligned_pairs
            results['conflicts'] = conflicts

        # Шаг 5: Парное судейство
        if 5 not in skip_steps:
            print("Шаг 5: Парное судейство...")
            judgments = self._step5_judge_pairs(aligned_pairs, claims)
            results['judgments'] = judgments
            print(f"✓ Оценено {len(judgments)} пар\n")
        else:
            print("Шаг 5: Пропущен")
            judgments = self._load_judgments()
            results['judgments'] = judgments

        # Шаг 6: Фактчекинг
        if 6 not in skip_steps:
            print("Шаг 6: Фактчекинг...")
            evidence = self._step6_verify_facts(claims, conflicts, judgments)
            results['evidence'] = evidence

            # Разделяем на подтверждённые/неподтверждённые
            verified, unverified = self.verifier.get_verified_claims(claims, evidence)
            results['verified_claims'] = verified
            results['unverified_claims'] = unverified
            print(f"✓ Проверено {len(evidence)} фактов")
            print(f"✓ Подтверждённых тезисов: {len(verified)}")
            print(f"✓ Неподтверждённых тезисов: {len(unverified)}\n")
        else:
            print("Шаг 6: Пропущен")
            evidence = []
            results['evidence'] = evidence

        print("=== Обработка завершена ===\n")

        # Генерируем отчёт
        self._generate_report(results)

        return results

    def _step1_extract_claims(self, documents: List[Dict]) -> List[Dict]:
        """Шаг 1: Экстракция тезисов"""
        claims = self.extractor.process(documents)
        self.extractor.save_result(
            claims,
            f"{self.output_dir}/claims.tsv",
            format='tsv'
        )
        return claims

    def _step4_align_pairs(
        self,
        pairs: List[Dict],
        claims: List[Dict]
    ) -> tuple:
        """Шаг 4: Семантическое сопоставление"""
        aligned_pairs, conflicts = self.aligner.process(pairs, claims)

        # Сохраняем пары
        self.aligner.save_result(
            aligned_pairs,
            f"{self.output_dir}/pairs_aligned.tsv",
            format='tsv'
        )

        # Сохраняем конфликты
        conflicts_md = self.aligner.generate_conflicts_md()
        self.aligner.save_result(
            conflicts_md,
            f"{self.output_dir}/conflicts.md",
            format='md'
        )

        return aligned_pairs, conflicts

    def _step5_judge_pairs(
        self,
        pairs: List[Dict],
        claims: List[Dict]
    ) -> List[Dict]:
        """Шаг 5: Парное судейство"""
        judgments = self.judge.process(pairs, claims, double_check=True)
        self.judge.save_result(
            judgments,
            f"{self.output_dir}/judgments.tsv",
            format='tsv'
        )
        return judgments

    def _step6_verify_facts(
        self,
        claims: List[Dict],
        conflicts: List[Dict],
        judgments: List[Dict]
    ) -> List[Dict]:
        """Шаг 6: Фактчекинг"""
        evidence = self.verifier.process(claims, conflicts, judgments)
        self.verifier.save_result(
            evidence,
            f"{self.output_dir}/evidence.tsv",
            format='tsv'
        )
        return evidence

    def _load_claims(self) -> List[Dict]:
        """Загрузка тезисов из файла"""
        path = f"{self.output_dir}/claims.tsv"
        if os.path.exists(path):
            return self.extractor.load_tsv(path)
        return []

    def _load_pairs(self) -> List[Dict]:
        """Загрузка пар из файла"""
        path = f"{self.output_dir}/pairs_aligned.tsv"
        if os.path.exists(path):
            return self.aligner.load_tsv(path)
        return []

    def _load_judgments(self) -> List[Dict]:
        """Загрузка судейства из файла"""
        path = f"{self.output_dir}/judgments.tsv"
        if os.path.exists(path):
            return self.judge.load_tsv(path)
        return []

    def _load_or_create_pairs(self, claims: List[Dict]) -> List[Dict]:
        """
        Загрузка или создание пар тезисов

        Если файл pairs.tsv существует, загружаем его.
        Иначе создаём простой набор пар для всех тезисов.
        """
        path = f"{self.output_dir}/pairs.tsv"
        if os.path.exists(path):
            return self.aligner.load_tsv(path)

        # Создаём простые пары: каждый с каждым
        pairs = []
        pair_id = 1
        for i, claim_a in enumerate(claims):
            for claim_b in claims[i+1:]:
                pairs.append({
                    'pair_id': f"P{pair_id:03d}",
                    'A_id': claim_a['id'],
                    'B_id': claim_b['id'],
                    'matched_keys': '',
                    'scope_overlap': ''
                })
                pair_id += 1

        # Сохраняем
        self.aligner.save_result(pairs, path, format='tsv')
        return pairs

    def _generate_report(self, results: Dict):
        """Генерация итогового отчёта"""
        report_path = f"{self.output_dir}/report.md"

        report = "# Отчёт о работе системы сравнения и слияния текстов\n\n"

        report += "## Статистика\n\n"
        report += f"- **Тезисов извлечено**: {len(results.get('claims', []))}\n"
        report += f"- **Пар проанализировано**: {len(results.get('pairs', []))}\n"
        report += f"- **Конфликтов обнаружено**: {len(results.get('conflicts', []))}\n"
        report += f"- **Пар оценено судьёй**: {len(results.get('judgments', []))}\n"
        report += f"- **Фактов проверено**: {len(results.get('evidence', []))}\n"

        if 'verified_claims' in results:
            report += f"- **Подтверждённых тезисов**: {len(results['verified_claims'])}\n"
            report += f"- **Неподтверждённых тезисов**: {len(results['unverified_claims'])}\n"

        report += "\n## Файлы результатов\n\n"
        report += "- `claims.tsv` - Извлечённые тезисы\n"
        report += "- `pairs_aligned.tsv` - Пары с определёнными отношениями\n"
        report += "- `conflicts.md` - Список конфликтов\n"
        report += "- `judgments.tsv` - Результаты судейства\n"
        report += "- `evidence.tsv` - Результаты фактчекинга\n"

        report += "\n## Следующие шаги\n\n"
        report += "1. Проверьте файл `conflicts.md` и разрешите конфликты\n"
        report += "2. Просмотрите `evidence.tsv` и добавьте недостающие источники\n"
        report += "3. Используйте `prompt_01.md` или `prompt_02.md` для финального синтеза\n"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"Отчёт сохранён: {report_path}")

    def generate_prompts_for_llm(self, results: Dict) -> Dict[str, str]:
        """
        Генерация промптов для LLM для каждого этапа

        Args:
            results: Результаты работы пайплайна

        Returns:
            Словарь промптов для каждого этапа
        """
        prompts = {}

        # Промпты для экстракции (для примера одного документа)
        if 'claims' in results and results['claims']:
            first_claim = results['claims'][0]
            doc_name = first_claim['origin'].split(',')[0]
            prompts['extraction'] = self.extractor.generate_prompt_for_llm(
                section_content="[Содержимое документа]",
                doc_name=doc_name
            )

        # Промпты для алигнера (для примера одной пары)
        if 'pairs' in results and results['pairs'] and 'claims' in results:
            claims_index = {c['id']: c for c in results['claims']}
            first_pair = results['pairs'][0]
            if first_pair['A_id'] in claims_index and first_pair['B_id'] in claims_index:
                claim_a = claims_index[first_pair['A_id']]
                claim_b = claims_index[first_pair['B_id']]
                prompts['alignment'] = self.aligner.generate_prompt_for_llm(claim_a, claim_b)

        # Промпты для судьи (для примера одной пары)
        if 'judgments' in results and 'claims' in results:
            claims_index = {c['id']: c for c in results['claims']}
            pairs_index = {p['pair_id']: p for p in results.get('pairs', [])}

            if results['judgments']:
                first_judgment = results['judgments'][0]
                pair_id = first_judgment['pair_id']
                if pair_id in pairs_index:
                    pair = pairs_index[pair_id]
                    claim_a = claims_index.get(pair['A_id'])
                    claim_b = claims_index.get(pair['B_id'])
                    if claim_a and claim_b:
                        prompts['judging'] = self.judge.generate_prompt_for_llm(
                            claim_a['claim'],
                            claim_b['claim'],
                            randomize=True
                        )

        # Промпты для верификатора (для примера одного тезиса)
        if 'verified_claims' in results or 'claims' in results:
            claims = results.get('verified_claims') or results['claims']
            if claims:
                first_claim = claims[0]
                questions = self.verifier._generate_verification_questions(first_claim)
                prompts['verification'] = self.verifier.generate_prompt_for_llm(
                    first_claim,
                    questions
                )

        # Сохраняем промпты
        prompts_path = f"{self.output_dir}/generated_prompts.json"
        with open(prompts_path, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)

        print(f"Промпты для LLM сохранены: {prompts_path}")

        return prompts


# Пример использования
if __name__ == "__main__":
    # Инициализация оркестратора
    orchestrator = TextMergeOrchestrator()

    # Определение документов для обработки
    documents = [
        {'path': 'data/doc1.md', 'name': 'Документ A'},
        {'path': 'data/doc2.md', 'name': 'Документ B'},
    ]

    # Запуск пайплайна
    # skip_steps можно использовать для пропуска уже выполненных шагов
    results = orchestrator.run_pipeline(documents, skip_steps=[])

    # Генерация промптов для LLM
    prompts = orchestrator.generate_prompts_for_llm(results)

    print("\n=== Работа завершена успешно ===")
