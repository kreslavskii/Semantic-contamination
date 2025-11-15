# План имплементации Tier 1 компонентов

**Дата начала:** 2025-11-15
**Статус:** В процессе
**Основание:** Анализ ANALYSIS_REASONING_AGENTS.md

---

## Цель

Превратить прототип с эвристиками в полнофункциональную систему с real LLM reasoning, веб-верификацией, SOTA ранжированием и детекцией галлюцинаций.

---

## Фаза 1: Базовая инфраструктура (Шаги 1-3)

### Шаг 1: Обновление зависимостей и конфигурации
**Приоритет:** КРИТИЧЕСКИЙ
**Оценка:** 1-2 часа

**Задачи:**
- [ ] Обновить `requirements.txt` с новыми библиотеками
- [ ] Создать `.env.example` для API ключей
- [ ] Создать `config.py` для централизованной конфигурации
- [ ] Обновить `.gitignore` для исключения `.env`

**Зависимости:**
```
# LLM APIs
openai>=1.0.0
anthropic>=0.3.0

# Reasoning & Orchestration
langchain>=0.1.0
langchain-community>=0.0.38
langchain-openai>=0.0.5

# Models
transformers>=4.36.0
torch>=2.0.0
sentence-transformers>=2.2.0

# Web search
google-search-results>=2.4.2  # SerpAPI

# Hallucination detection
selfcheckgpt>=0.1.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
tiktoken>=0.5.0

# Existing
pandas>=2.0.0
numpy>=1.24.0
regex>=2023.0.0
python-dateutil>=2.8.0
```

**Обоснование:**
- Нужна централизованная конфигурация для управления API ключами
- `.env` файл для безопасного хранения credentials
- Все зависимости должны быть совместимы по версиям

---

### Шаг 2: Базовая LLM интеграция
**Приоритет:** КРИТИЧЕСКИЙ
**Оценка:** 2-3 часа

**Задачи:**
- [ ] Создать `src/llm/llm_client.py` с абстракцией для LLM API
- [ ] Поддержка OpenAI и Anthropic с единым интерфейсом
- [ ] Добавить error handling и retry logic
- [ ] Добавить rate limiting и cost tracking

**Файлы для создания:**
- `src/llm/__init__.py`
- `src/llm/llm_client.py`
- `src/llm/config.py`

**Обоснование:**
- Единая абстракция позволяет легко менять провайдеров LLM
- Централизованный error handling упрощает отладку
- Cost tracking критичен для контроля расходов

---

### Шаг 3: Конфигурация и константы
**Приоритет:** КРИТИЧЕСКИЙ
**Оценка:** 1 час

**Задачи:**
- [ ] Создать `src/config.py` с константами и настройками
- [ ] Настроить логирование
- [ ] Создать utility функции для работы с путями

**Обоснование:**
- Централизованная конфигурация упрощает поддержку
- Логирование критично для debugging production issues

---

## Фаза 2: Имплементация агентов (Шаги 4-7)

### Шаг 4: VerifierAgent + ReAct + Web Search
**Приоритет:** КРИТИЧЕСКИЙ
**Оценка:** 6-8 часов

**Текущая проблема:**
- `VerifierAgent._verify_question()` (строка 174-204) — заглушка
- Возвращает `status: 'uncertain'` для всех запросов
- Нет реального веб-поиска

**Решение:**
1. Интегрировать LangChain + SerpAPI для веб-поиска
2. Реализовать ReAct-цикл (Thought → Action → Observation)
3. Заменить заглушку на real verification logic

**Задачи:**
- [ ] Создать `src/tools/web_search.py` с SerpAPI wrapper
- [ ] Обновить `VerifierAgent.__init__()` для инициализации LLM и tools
- [ ] Реализовать `_verify_question()` с ReAct-циклом
- [ ] Добавить `_parse_verification_result()` для извлечения статуса
- [ ] Обновить `verify_with_search()` для использования нового механизма
- [ ] Добавить fallback на старую логику при отсутствии API ключей

**Код-структура:**
```python
class VerifierAgent(BaseAgent):
    def __init__(self, prompt_template_path, llm_client=None, search_tool=None):
        super().__init__(prompt_template_path)
        self.llm = llm_client or get_default_llm()
        self.search = search_tool or WebSearchTool()

    def _verify_question(self, claim_id, question, claim):
        # ReAct цикл
        thought = self._generate_thought(question)
        action = self._decide_action(thought)
        observation = self._execute_action(action)
        result = self._synthesize_result(thought, observation)
        return result
```

**Обоснование:**
- ReAct показал высокую эффективность для задач с внешними инструментами
- Структурированный цикл рассуждения повышает надёжность
- Fallback на эвристики обеспечивает работу без API

**Метрики успеха:**
- Статус verification != 'uncertain' в >80% случаев
- Найдены релевантные источники в >70% случаев

---

### Шаг 5: JudgeAgent + PairRM
**Приоритет:** КРИТИЧЕСКИЙ
**Оценка:** 4-6 часов

**Текущая проблема:**
- `JudgeAgent._evaluate_criterion()` использует примитивные эвристики
- `_determine_winner()` — простой подсчет побед
- Нет использования SOTA моделей

**Решение:**
1. Интегрировать PairRM (llm-blender/PairRM) для парного ранжирования
2. Заменить эвристики на model-based scoring
3. Добавить hybrid approach: PairRM + LLM для edge cases

**Задачи:**
- [ ] Создать `src/models/pairrm_ranker.py` с PairRM wrapper
- [ ] Обновить `JudgeAgent.__init__()` для загрузки PairRM
- [ ] Реализовать `_judge_with_pairrm()` для парного ранжирования
- [ ] Обновить `_judge_pair()` для использования PairRM
- [ ] Добавить `_judge_with_llm()` для сложных случаев
- [ ] Сохранить эвристики как fallback при отсутствии моделей

**Код-структура:**
```python
class JudgeAgent(BaseAgent):
    def __init__(self, prompt_template_path, use_pairrm=True, llm_client=None):
        super().__init__(prompt_template_path)
        self.pairrm = PairRMRanker() if use_pairrm else None
        self.llm = llm_client or get_default_llm()

    def _judge_pair(self, claim_a, claim_b, pair, order='AB'):
        if self.pairrm:
            # PairRM inference
            return self._judge_with_pairrm(claim_a, claim_b, pair)
        elif self.llm:
            # LLM-based judging
            return self._judge_with_llm(claim_a, claim_b, pair)
        else:
            # Fallback на эвристики
            return self._judge_with_heuristics(claim_a, claim_b, pair)
```

**Обоснование:**
- PairRM специально обучена для парного ранжирования (bidirectional attention)
- Локальный inference без API-вызовов снижает стоимость
- Hybrid подход обеспечивает robustness

**Метрики успеха:**
- Correlation с human judgments >0.75
- Согласованность при double-check >85%

---

### Шаг 6: ExtractorAgent + SelfCheckGPT
**Приоритет:** ВЫСОКИЙ
**Оценка:** 4-5 часов

**Текущая проблема:**
- `ExtractorAgent._extract_claims_from_section()` использует эвристики
- Нет детекции галлюцинаций при извлечении
- `_is_substantive()` — примитивная проверка

**Решение:**
1. Добавить LLM-based extraction вместо эвристик
2. Интегрировать SelfCheckGPT для детекции галлюцинаций
3. Multiple sampling для проверки консистентности

**Задачи:**
- [ ] Обновить `ExtractorAgent.__init__()` для инициализации LLM и SelfCheck
- [ ] Реализовать `_extract_with_llm()` для LLM-based extraction
- [ ] Добавить `_check_hallucination()` с SelfCheckGPT
- [ ] Обновить `_extract_claims_from_section()` для использования LLM + SelfCheck
- [ ] Добавить конфигурацию для количества samples (по умолчанию 3-5)
- [ ] Сохранить эвристики как fast fallback

**Код-структура:**
```python
class ExtractorAgent(BaseAgent):
    def __init__(self, prompt_template_path, llm_client=None, use_selfcheck=True):
        super().__init__(prompt_template_path)
        self.llm = llm_client or get_default_llm()
        self.selfcheck = SelfCheckNLI() if use_selfcheck else None

    def _extract_claims_from_section(self, section_content, ...):
        # 1. LLM extraction
        claims = self._extract_with_llm(section_content, ...)

        # 2. SelfCheck для фильтрации галлюцинаций
        if self.selfcheck:
            verified_claims = self._filter_hallucinations(claims, section_content)
            return verified_claims

        return claims

    def _filter_hallucinations(self, claims, source_text):
        # Multiple sampling
        samples = [self._extract_with_llm(source_text) for _ in range(3)]

        verified = []
        for claim in claims:
            score = self.selfcheck.predict(
                sentences=[claim['claim']],
                sampled_passages=[s['claim'] for s in samples]
            )
            if score[0] < 0.5:  # low hallucination
                verified.append(claim)

        return verified
```

**Обоснование:**
- LLM лучше понимает семантику, чем regex
- SelfCheckGPT использует консистентность как прокси для фактичности
- Multiple sampling критичен для детекции галлюцинаций

**Метрики успеха:**
- Precision извлечения >0.85 (меньше false positives)
- Recall >0.80 (не пропускаем важные тезисы)
- Hallucination rate <10%

---

### Шаг 7: AlignerAgent улучшения
**Приоритет:** СРЕДНИЙ
**Оценка:** 2-3 часа

**Задачи:**
- [ ] Добавить LLM-based semantic matching вместо Jaccard
- [ ] Улучшить `_determine_relation()` с использованием LLM
- [ ] Добавить Concise CoT в промпты
- [ ] Сохранить Jaccard как fast approximation

**Обоснование:**
- LLM лучше понимает семантические отношения
- CoT улучшает reasoning quality
- Jaccard остается для быстрой preliminary filtering

---

## Фаза 3: Конфигурация и документация (Шаги 8-10)

### Шаг 8: Обновление конфигурации
**Приоритет:** КРИТИЧЕСКИЙ
**Оценка:** 1-2 часа

**Задачи:**
- [ ] Создать `.env.example` с примерами API ключей
- [ ] Создать `src/config.py` с настройками
- [ ] Добавить валидацию конфигурации
- [ ] Документировать все настройки

**Пример `.env.example`:**
```bash
# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Search API
SERPAPI_API_KEY=...

# Model Settings
DEFAULT_LLM_PROVIDER=openai  # openai | anthropic
DEFAULT_MODEL=gpt-4-turbo-preview
MAX_TOKENS=2000
TEMPERATURE=0.7

# Agent Settings
USE_PAIRRM=true
USE_SELFCHECK=true
SELFCHECK_SAMPLES=3

# Cost Limits
MAX_COST_PER_RUN=10.0  # USD
```

---

### Шаг 9: Обновление документации
**Приоритет:** ВЫСОКИЙ
**Оценка:** 2-3 часа

**Задачи:**
- [ ] Обновить README.md с новыми возможностями
- [ ] Создать SETUP.md с инструкциями по установке
- [ ] Обновить примеры использования
- [ ] Документировать конфигурационные опции
- [ ] Создать TROUBLESHOOTING.md

**Структура новой документации:**
- README.md — обзор и quick start
- SETUP.md — детальная установка
- USAGE.md — примеры использования
- CONFIG.md — описание конфигурации
- TROUBLESHOOTING.md — частые проблемы

---

### Шаг 10: Тестирование и примеры
**Приоритет:** ВЫСОКИЙ
**Оценка:** 3-4 часа

**Задачи:**
- [ ] Создать `tests/` директорию
- [ ] Написать unit tests для каждого агента
- [ ] Создать integration test для полного пайплайна
- [ ] Добавить example scripts в `examples/`
- [ ] Создать golden dataset для regression testing

**Тесты:**
- `tests/test_verifier.py` — тесты для VerifierAgent
- `tests/test_judge.py` — тесты для JudgeAgent
- `tests/test_extractor.py` — тесты для ExtractorAgent
- `tests/test_pipeline.py` — интеграционные тесты

**Примеры:**
- `examples/basic_usage.py` — простой пример
- `examples/custom_config.py` — кастомная конфигурация
- `examples/fallback_mode.py` — работа без API

---

## Оценка трудозатрат

| Фаза | Шаги | Оценка часов |
|------|------|--------------|
| **Фаза 1: Инфраструктура** | 1-3 | 4-6 |
| **Фаза 2: Агенты** | 4-7 | 16-22 |
| **Фаза 3: Документация** | 8-10 | 6-9 |
| **ИТОГО** | 1-10 | **26-37 часов** |

**Ожидаемые результаты:**
- ✅ Working solution вместо прототипа
- ✅ Реальная веб-верификация фактов
- ✅ SOTA парное ранжирование
- ✅ Детекция галлюцинаций
- ✅ Полная документация
- ✅ Тесты и примеры

---

## Порядок выполнения

**Неделя 1:**
- День 1-2: Шаги 1-3 (инфраструктура)
- День 3-4: Шаг 4 (VerifierAgent)
- День 5: Шаг 5 (JudgeAgent, начало)

**Неделя 2:**
- День 1: Шаг 5 (JudgeAgent, завершение)
- День 2-3: Шаг 6 (ExtractorAgent)
- День 4: Шаг 7 (AlignerAgent)
- День 5: Шаги 8-10 (документация и тесты)

---

## Риски и митигация

| Риск | Вероятность | Митигация |
|------|-------------|-----------|
| API rate limits | Средняя | Добавить rate limiting, retry logic |
| High API costs | Средняя | Cost tracking, лимиты, кэширование |
| Model unavailability | Низкая | Fallback на эвристики |
| Сложность интеграции | Средняя | Модульный подход, изоляция компонентов |

---

## Критерии успеха

**Технические:**
- [ ] Все агенты работают с LLM API
- [ ] VerifierAgent успешно выполняет веб-поиск
- [ ] JudgeAgent использует PairRM
- [ ] ExtractorAgent фильтрует галлюцинации
- [ ] Все тесты проходят
- [ ] Документация полная

**Качественные:**
- [ ] Verification accuracy >80%
- [ ] Judging consistency >85%
- [ ] Extraction hallucination rate <10%
- [ ] End-to-end pipeline работает без ошибок

**Операционные:**
- [ ] Setup time <15 минут
- [ ] Clear error messages
- [ ] Comprehensive logging
- [ ] Cost tracking работает

---

## Следующие шаги после завершения

После успешной имплементации Tier 1:
1. Собрать метрики качества на реальных данных
2. Оценить необходимость Tier 2 компонентов:
   - LangGraph для оркестрации
   - CoVe для верификации
   - CCoT для промптов
3. Подготовить план для Tier 2 имплементации
