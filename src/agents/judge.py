"""
Агент для слепого парного судейства тезисов по критериям (Шаг 5)

ОБНОВЛЕНО (Шаг 5):
- Интегрирован PairRM (llm-blender/PairRM) для SOTA парного ранжирования
- Добавлена LLM поддержка для сложных случаев
- Сохранены эвристики как final fallback
- 3-уровневая graceful degradation
"""
import random
import logging
from typing import List, Dict, Tuple, Optional
from .base_agent import BaseAgent

# PairRM imports (опциональные)
try:
    from ..models import PairRMRanker
    HAS_PAIRRM = True
except ImportError:
    HAS_PAIRRM = False
    PairRMRanker = None

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


class JudgeAgent(BaseAgent):
    """Агент для оценки качества тезисов по критериям"""

    CRITERIA = {
        'C1': 'Корректность фактов и непротиворечивость',
        'C2': 'Полнота ответа по вопросу',
        'C3': 'Логическая связность',
        'C4': 'Экономия языка',
        'C5': 'Проверяемость и источники'
    }

    def __init__(
        self,
        prompt_template_path: str = "prompts/judge_prompt.md",
        pairrm_ranker: Optional['PairRMRanker'] = None,
        llm_client: Optional['LLMClient'] = None,
        use_pairrm: Optional[bool] = None,
        use_llm_tiebreaker: bool = False
    ):
        """
        Инициализация JudgeAgent

        Args:
            prompt_template_path: Путь к промпт-шаблону
            pairrm_ranker: PairRM ranker (если None, создастся автоматически)
            llm_client: LLM клиент для tiebreaker (если None, создастся автоматически)
            use_pairrm: Использовать ли PairRM (если None, берется из settings)
            use_llm_tiebreaker: Использовать ли LLM для разрешения сложных случаев
        """
        super().__init__(prompt_template_path)

        # Настройка режимов
        self.use_pairrm = use_pairrm if use_pairrm is not None else settings.USE_PAIRRM
        self.use_llm_tiebreaker = use_llm_tiebreaker and HAS_LLM

        # Уровень 1: PairRM (preferred)
        if self.use_pairrm and HAS_PAIRRM:
            try:
                self.ranker = pairrm_ranker or PairRMRanker()
                logger.info("JudgeAgent: используется PairRM для ранжирования")
            except Exception as e:
                logger.error(f"Не удалось инициализировать PairRM: {e}")
                self.ranker = None
                self.use_pairrm = False
        else:
            self.ranker = None
            if not HAS_PAIRRM:
                logger.warning("PairRM недоступен (нет transformers/torch)")

        # Уровень 2: LLM tiebreaker (optional)
        if self.use_llm_tiebreaker:
            try:
                self.llm = llm_client or get_default_llm(temperature=0.2)
                logger.info("JudgeAgent: LLM tiebreaker включен")
            except Exception as e:
                logger.warning(f"LLM tiebreaker недоступен: {e}")
                self.llm = None
                self.use_llm_tiebreaker = False
        else:
            self.llm = None

        # Уровень 3: Эвристики (fallback)
        logger.info("JudgeAgent: эвристики доступны как fallback")

        # Статистика
        self.judgment_stats = {
            'total_pairs': 0,
            'pairrm_used': 0,
            'llm_used': 0,
            'heuristic_used': 0,
            'ties': 0,
            'uncertain': 0
        }

    def process(
        self,
        pairs: List[Dict],
        claims: List[Dict],
        double_check: bool = True
    ) -> List[Dict]:
        """
        Оценка пар тезисов по критериям

        Args:
            pairs: Список пар тезисов (с отношениями от Aligner)
            claims: Список всех тезисов
            double_check: Выполнять ли двойной прогон (A vs B, затем B vs A)

        Returns:
            Список результатов судейства
        """
        # Создаём индекс тезисов
        claims_index = {claim['id']: claim for claim in claims}

        judgments = []

        for pair in pairs:
            # Пропускаем независимые пары
            if pair.get('relation') == 'independent':
                continue

            a_id = pair['A_id']
            b_id = pair['B_id']

            if a_id not in claims_index or b_id not in claims_index:
                continue

            claim_a = claims_index[a_id]
            claim_b = claims_index[b_id]

            # Первый прогон
            judgment1 = self._judge_pair(claim_a, claim_b, pair, order='AB')

            if double_check:
                # Второй прогон с обратным порядком
                judgment2 = self._judge_pair(claim_b, claim_a, pair, order='BA')

                # Проверяем согласованность
                if judgment1['winner'] != self._reverse_winner(judgment2['winner']):
                    # Результаты расходятся
                    judgment = self._resolve_disagreement(judgment1, judgment2, pair)
                else:
                    # Результаты согласуются
                    judgment = judgment1
            else:
                judgment = judgment1

            judgments.append(judgment)

        return judgments

    def _judge_pair(
        self,
        claim_a: Dict,
        claim_b: Dict,
        pair: Dict,
        order: str = 'AB'
    ) -> Dict:
        """
        Оценка одной пары тезисов

        Args:
            claim_a: Первый тезис (Кандидат-1)
            claim_b: Второй тезис (Кандидат-2)
            pair: Информация о паре
            order: Порядок оценки ('AB' или 'BA')

        Returns:
            Результат судейства
        """
        # Анонимизируем тезисы
        candidate_1 = claim_a['claim']
        candidate_2 = claim_b['claim']

        # Оцениваем по каждому критерию
        scores = {}
        for criterion_id, criterion_name in self.CRITERIA.items():
            score = self._evaluate_criterion(
                candidate_1,
                candidate_2,
                claim_a,
                claim_b,
                criterion_id
            )
            scores[criterion_id] = score

        # Определяем победителя
        winner = self._determine_winner(scores, claim_a['id'], claim_b['id'])

        # Формируем заметки
        notes = self._generate_notes(scores, claim_a, claim_b)

        return {
            'pair_id': pair['pair_id'],
            'winner': winner,
            'C1': scores['C1'],
            'C2': scores['C2'],
            'C3': scores['C3'],
            'C4': scores['C4'],
            'C5': scores['C5'],
            'notes': notes,
            'order': order
        }

    def _evaluate_criterion(
        self,
        candidate_1: str,
        candidate_2: str,
        claim_a: Dict,
        claim_b: Dict,
        criterion_id: str
    ) -> str:
        """
        Оценка пары по одному критерию

        Использует 3-уровневую стратегию:
        1. PairRM (если доступен) - SOTA модель
        2. LLM tiebreaker (если включен и PairRM дал tie)
        3. Эвристики (fallback)

        Args:
            candidate_1: Текст первого кандидата
            candidate_2: Текст второго кандидата
            claim_a: Полный объект первого тезиса
            claim_b: Полный объект второго тезиса
            criterion_id: Идентификатор критерия (C1-C5)

        Returns:
            Оценка: 'A+' (первый лучше), 'B+' (второй лучше), 'tie' (паритет)
        """
        # Уровень 1: PairRM
        if self.use_pairrm and self.ranker:
            try:
                result = self._evaluate_with_pairrm(
                    candidate_1, candidate_2, claim_a, claim_b, criterion_id
                )

                # Если PairRM дал уверенный результат, используем его
                if result != 'tie' or not self.use_llm_tiebreaker:
                    self.judgment_stats['pairrm_used'] += 1
                    return result

                # Если tie и есть LLM tiebreaker, используем его
                if self.use_llm_tiebreaker and self.llm:
                    logger.debug(f"PairRM дал tie для {criterion_id}, используем LLM tiebreaker")
                    llm_result = self._evaluate_with_llm(
                        candidate_1, candidate_2, criterion_id
                    )
                    if llm_result != 'tie':
                        self.judgment_stats['llm_used'] += 1
                        return llm_result

                self.judgment_stats['pairrm_used'] += 1
                return result

            except Exception as e:
                logger.error(f"Ошибка в PairRM оценке: {e}")
                logger.warning("Переключаемся на эвристики")

        # Уровень 2/3: Эвристики (fallback)
        self.judgment_stats['heuristic_used'] += 1

        if criterion_id == 'C1':  # Корректность фактов
            return self._evaluate_correctness(claim_a, claim_b)

        elif criterion_id == 'C2':  # Полнота
            return self._evaluate_completeness(claim_a, claim_b)

        elif criterion_id == 'C3':  # Логическая связность
            return self._evaluate_coherence(claim_a, claim_b)

        elif criterion_id == 'C4':  # Экономия языка
            return self._evaluate_brevity(claim_a, claim_b)

        elif criterion_id == 'C5':  # Проверяемость
            return self._evaluate_verifiability(claim_a, claim_b)

        return 'tie'

    def _evaluate_with_pairrm(
        self,
        candidate_1: str,
        candidate_2: str,
        claim_a: Dict,
        claim_b: Dict,
        criterion_id: str
    ) -> str:
        """
        Оценка с использованием PairRM

        Args:
            candidate_1: Текст первого кандидата
            candidate_2: Текст второго кандидата
            claim_a: Полный объект первого тезиса
            claim_b: Полный объект второго тезиса
            criterion_id: Идентификатор критерия

        Returns:
            Результат: 'A+', 'B+', или 'tie'
        """
        # Формируем instruction для PairRM на основе критерия
        criterion_name = self.CRITERIA[criterion_id]
        instruction = f"Сравните два текста по критерию: {criterion_name}. Какой текст лучше?"

        # Вызываем PairRM
        result = self.ranker.compare(
            text_a=candidate_1,
            text_b=candidate_2,
            instruction=instruction,
            tie_threshold=0.1  # Порог для определения паритета
        )

        # Конвертируем результат PairRM в формат JudgeAgent
        if result.winner == 'A':
            return 'A+'
        elif result.winner == 'B':
            return 'B+'
        else:
            return 'tie'

    def _evaluate_with_llm(
        self,
        candidate_1: str,
        candidate_2: str,
        criterion_id: str
    ) -> str:
        """
        LLM tiebreaker для сложных случаев

        Args:
            candidate_1: Текст первого кандидата
            candidate_2: Текст второго кандидата
            criterion_id: Идентификатор критерия

        Returns:
            Результат: 'A+', 'B+', или 'tie'
        """
        criterion_name = self.CRITERIA[criterion_id]

        prompt = f"""Сравни два текста по критерию: {criterion_name}

Текст А: {candidate_1}

Текст Б: {candidate_2}

Ответь кратко (одно слово):
- "A" если текст А лучше
- "B" если текст Б лучше
- "tie" если паритет

Ответ:"""

        try:
            response = self.llm.generate(prompt, temperature=0.2, max_tokens=10)
            answer = response.text.strip().lower()

            if 'a' in answer and 'b' not in answer:
                return 'A+'
            elif 'b' in answer:
                return 'B+'
            else:
                return 'tie'

        except Exception as e:
            logger.error(f"Ошибка в LLM tiebreaker: {e}")
            return 'tie'

    def _evaluate_correctness(self, claim_a: Dict, claim_b: Dict) -> str:
        """Оценка корректности фактов"""
        # Проверяем наличие фактов
        facts_a = claim_a.get('facts', '')
        facts_b = claim_b.get('facts', '')

        # Проверяем наличие доказательств
        evidence_a = claim_a.get('evidence_inline', '')
        evidence_b = claim_b.get('evidence_inline', '')

        score_a = (1 if facts_a else 0) + (1 if evidence_a else 0)
        score_b = (1 if facts_b else 0) + (1 if evidence_b else 0)

        if score_a > score_b:
            return 'A+'
        elif score_b > score_a:
            return 'B+'
        return 'tie'

    def _evaluate_completeness(self, claim_a: Dict, claim_b: Dict) -> str:
        """Оценка полноты"""
        # Полнота оценивается по длине и детализации
        len_a = len(claim_a['claim'].split())
        len_b = len(claim_b['claim'].split())

        # Проверяем наличие условий применимости
        scope_a = sum([
            1 if claim_a.get('scope.time') else 0,
            1 if claim_a.get('scope.jurisdiction') else 0,
            1 if claim_a.get('scope.conditions') else 0
        ])
        scope_b = sum([
            1 if claim_b.get('scope.time') else 0,
            1 if claim_b.get('scope.jurisdiction') else 0,
            1 if claim_b.get('scope.conditions') else 0
        ])

        score_a = len_a + scope_a * 10
        score_b = len_b + scope_b * 10

        if score_a > score_b * 1.2:
            return 'A+'
        elif score_b > score_a * 1.2:
            return 'B+'
        return 'tie'

    def _evaluate_coherence(self, claim_a: Dict, claim_b: Dict) -> str:
        """Оценка логической связности"""
        # Проверяем наличие связующих слов
        coherence_markers = [
            'потому что', 'так как', 'поэтому', 'следовательно',
            'в результате', 'из-за', 'вследствие', 'если', 'то'
        ]

        text_a = claim_a['claim'].lower()
        text_b = claim_b['claim'].lower()

        score_a = sum(1 for marker in coherence_markers if marker in text_a)
        score_b = sum(1 for marker in coherence_markers if marker in text_b)

        if score_a > score_b:
            return 'A+'
        elif score_b > score_a:
            return 'B+'
        return 'tie'

    def _evaluate_brevity(self, claim_a: Dict, claim_b: Dict) -> str:
        """Оценка экономии языка"""
        # Краткость при сохранении смысла
        len_a = len(claim_a['claim'])
        len_b = len(claim_b['claim'])

        # Более короткий текст лучше (при прочих равных)
        if len_a < len_b * 0.8:
            return 'A+'
        elif len_b < len_a * 0.8:
            return 'B+'
        return 'tie'

    def _evaluate_verifiability(self, claim_a: Dict, claim_b: Dict) -> str:
        """Оценка проверяемости"""
        # Наличие фактов, источников, конкретных данных
        facts_a = len(claim_a.get('facts', '').split(';')) if claim_a.get('facts') else 0
        facts_b = len(claim_b.get('facts', '').split(';')) if claim_b.get('facts') else 0

        evidence_a = 1 if claim_a.get('evidence_inline') else 0
        evidence_b = 1 if claim_b.get('evidence_inline') else 0

        score_a = facts_a + evidence_a * 2
        score_b = facts_b + evidence_b * 2

        if score_a > score_b:
            return 'A+'
        elif score_b > score_a:
            return 'B+'
        return 'tie'

    def _determine_winner(
        self,
        scores: Dict[str, str],
        id_a: str,
        id_b: str
    ) -> str:
        """
        Определение победителя по агрегации критериев

        Args:
            scores: Оценки по критериям
            id_a: ID первого тезиса
            id_b: ID второго тезиса

        Returns:
            Победитель: 'A', 'B', или 'tie'
        """
        # Подсчитываем победы
        wins_a = sum(1 for score in scores.values() if score == 'A+')
        wins_b = sum(1 for score in scores.values() if score == 'B+')

        # Приоритет критериям (C1 > C2 > C5)
        priority_criteria = ['C1', 'C2', 'C5']
        priority_wins_a = sum(1 for c in priority_criteria if scores.get(c) == 'A+')
        priority_wins_b = sum(1 for c in priority_criteria if scores.get(c) == 'B+')

        # Решение
        if priority_wins_a > priority_wins_b:
            return 'A'
        elif priority_wins_b > priority_wins_a:
            return 'B'
        elif wins_a > wins_b:
            return 'A'
        elif wins_b > wins_a:
            return 'B'
        else:
            return 'tie'

    def _generate_notes(
        self,
        scores: Dict[str, str],
        claim_a: Dict,
        claim_b: Dict
    ) -> str:
        """
        Генерация заметок о судействе

        Args:
            scores: Оценки по критериям
            claim_a: Первый тезис
            claim_b: Второй тезис

        Returns:
            Текст заметок
        """
        notes = []

        # Анализируем по каждому критерию
        for criterion_id, score in scores.items():
            if score == 'A+':
                notes.append(f"{criterion_id}: Кандидат-1 лучше")
            elif score == 'B+':
                notes.append(f"{criterion_id}: Кандидат-2 лучше")

        if not notes:
            return "Паритет по всем критериям"

        return "; ".join(notes)

    def _reverse_winner(self, winner: str) -> str:
        """
        Инверсия победителя для сравнения с обратным порядком

        Args:
            winner: Победитель в прямом порядке

        Returns:
            Инвертированный победитель
        """
        if winner == 'A':
            return 'B'
        elif winner == 'B':
            return 'A'
        return 'tie'

    def _resolve_disagreement(
        self,
        judgment1: Dict,
        judgment2: Dict,
        pair: Dict
    ) -> Dict:
        """
        Разрешение противоречий между двумя прогонами

        Args:
            judgment1: Результат первого прогона
            judgment2: Результат второго прогона
            pair: Информация о паре

        Returns:
            Итоговое решение
        """
        # Если результаты противоречат друг другу, отмечаем как неопределённость
        return {
            'pair_id': pair['pair_id'],
            'winner': 'uncertain',
            'C1': 'uncertain',
            'C2': 'uncertain',
            'C3': 'uncertain',
            'C4': 'uncertain',
            'C5': 'uncertain',
            'notes': f"Противоречие между прогонами: {judgment1['winner']} vs {judgment2['winner']}",
            'order': 'both'
        }

    def generate_prompt_for_llm(
        self,
        candidate_1: str,
        candidate_2: str,
        randomize: bool = True
    ) -> str:
        """
        Генерация промпта для LLM для судейства

        Args:
            candidate_1: Текст первого кандидата
            candidate_2: Текст второго кандидата
            randomize: Рандомизировать ли порядок кандидатов

        Returns:
            Сформированный промпт
        """
        # Рандомизация порядка
        if randomize and random.choice([True, False]):
            candidate_1, candidate_2 = candidate_2, candidate_1
            order_note = "(порядок случайный: инвертирован)"
        else:
            order_note = "(порядок случайный: прямой)"

        return self.get_prompt() + f"""

## Кандидаты для сравнения {order_note}

**Кандидат-1:**
{candidate_1}

**Кандидат-2:**
{candidate_2}

Выполни оценку согласно критериям выше.
"""

    def get_judgment_stats(self) -> Dict:
        """
        Получить статистику судейства

        Returns:
            Словарь со статистикой использования разных методов
        """
        stats = self.judgment_stats.copy()

        # Подсчет процентов
        total_evaluations = (
            stats['pairrm_used'] +
            stats['llm_used'] +
            stats['heuristic_used']
        )

        if total_evaluations > 0:
            stats['pairrm_pct'] = (stats['pairrm_used'] / total_evaluations) * 100
            stats['llm_pct'] = (stats['llm_used'] / total_evaluations) * 100
            stats['heuristic_pct'] = (stats['heuristic_used'] / total_evaluations) * 100

        # Информация о режиме
        if self.use_pairrm and self.ranker:
            stats['mode'] = 'PairRM + Heuristics'
            if self.use_llm_tiebreaker:
                stats['mode'] = 'PairRM + LLM + Heuristics'
        else:
            stats['mode'] = 'Heuristics only'

        # Информация о модели PairRM
        if self.ranker:
            pairrm_stats = self.ranker.get_stats()
            stats['pairrm_model_stats'] = pairrm_stats

        return stats


# Пример использования
if __name__ == "__main__":
    judge = JudgeAgent()

    # Загрузка данных
    pairs = judge.load_tsv('output/pairs_aligned.tsv')
    claims = judge.load_tsv('output/claims.tsv')

    # Судейство
    judgments = judge.process(pairs, claims, double_check=True)

    # Сохранение результатов
    judge.save_result(judgments, 'output/judgments.tsv', format='tsv')

    print(f"Оценено {len(judgments)} пар")
