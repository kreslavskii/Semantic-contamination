# Semantic Contamination - Система сравнения и слияния текстов

Проект для сравнения и слияния до четырёх текстов на общую тему с использованием многоагентного подхода и методов LLM-as-a-Judge.

## Описание

Система реализует процесс структурированного анализа и слияния текстовых документов через последовательные этапы обработки. Основные возможности:

- **Экстракция тезисов** из документов с выделением фактов и условий применимости
- **Семантическое сопоставление** тезисов с определением типов отношений
- **Парное судейство** по критериям качества (корректность, полнота, связность и др.)
- **Фактчекинг** через веб-поиск (Chain-of-Verification)
- **Синтез** финального согласованного документа

## Структура проекта

```
Semantic-contamination/
├── src/
│   └── agents/
│       ├── __init__.py
│       ├── base_agent.py      # Базовый класс для агентов
│       ├── extractor.py       # Шаг 1: Экстракция тезисов
│       ├── aligner.py         # Шаг 4: Семантическое сопоставление
│       ├── judge.py           # Шаг 5: Парное судейство
│       └── verifier.py        # Шаг 6: Фактчекинг
├── prompts/
│   ├── extractor_prompt.md    # Шаблон промпта для экстрактора
│   ├── aligner_prompt.md      # Шаблон промпта для алигнера
│   ├── judge_prompt.md        # Шаблон промпта для судьи
│   └── verifier_prompt.md     # Шаблон промпта для верификатора
├── data/                      # Входные документы
├── output/                    # Результаты работы
│   ├── claims.tsv            # Извлечённые тезисы
│   ├── pairs_aligned.tsv     # Пары с отношениями
│   ├── judgments.tsv         # Результаты судейства
│   ├── evidence.tsv          # Результаты фактчекинга
│   └── conflicts.md          # Список конфликтов
├── prompt_01.md              # Промпт для синтеза (вариант 1)
├── prompt_02.md              # Промпт для синтеза (вариант 2)
├── requirements.txt
└── README.md
```

## Установка

```bash
# Клонирование репозитория
git clone <repository-url>
cd Semantic-contamination

# Установка зависимостей
pip install -r requirements.txt
```

## Использование

### Шаг 1: Экстракция тезисов

```python
from src.agents import ExtractorAgent

extractor = ExtractorAgent()

documents = [
    {'path': 'data/doc1.md', 'name': 'Документ A'},
    {'path': 'data/doc2.md', 'name': 'Документ B'},
]

claims = extractor.process(documents)
extractor.save_result(claims, 'output/claims.tsv', format='tsv')
```

### Шаг 4: Семантическое сопоставление

```python
from src.agents import AlignerAgent

aligner = AlignerAgent()

# Предполагается, что у вас есть pairs.tsv (можно создать вручную или через Blocking)
pairs = aligner.load_tsv('output/pairs.tsv')
claims = aligner.load_tsv('output/claims.tsv')

updated_pairs, conflicts = aligner.process(pairs, claims)
aligner.save_result(updated_pairs, 'output/pairs_aligned.tsv', format='tsv')

# Сохранение конфликтов
conflicts_md = aligner.generate_conflicts_md()
aligner.save_result(conflicts_md, 'output/conflicts.md', format='md')
```

### Шаг 5: Парное судейство

```python
from src.agents import JudgeAgent

judge = JudgeAgent()

pairs = judge.load_tsv('output/pairs_aligned.tsv')
claims = judge.load_tsv('output/claims.tsv')

judgments = judge.process(pairs, claims, double_check=True)
judge.save_result(judgments, 'output/judgments.tsv', format='tsv')
```

### Шаг 6: Фактчекинг

```python
from src.agents import VerifierAgent

verifier = VerifierAgent()

claims = verifier.load_tsv('output/claims.tsv')
conflicts = verifier.load_json('output/conflicts.json')

evidence = verifier.process(claims, conflicts)
verifier.save_result(evidence, 'output/evidence.tsv', format='tsv')

# Разделение на подтверждённые/неподтверждённые
verified, unverified = verifier.get_verified_claims(claims, evidence)
```

## Реализованные компоненты

### ✅ Пункт 1: Экстракция тезисов (Extractor)
- Извлечение содержательных утверждений из документов
- Выделение фактов, условий применимости
- Формирование структурированных записей тезисов

### ✅ Пункт 4: Семантическое сопоставление (Aligner/NLI)
- Определение типов отношений: equivalent, refines, extends, contradicts, independent
- Классификация логического статуса (NLI)
- Типизация конфликтов: true_conflict, apparent_conflict, pseudo_conflict

### ✅ Пункт 5: Парное судейство (Judge)
- Оценка по 5 критериям: корректность, полнота, связность, экономия языка, проверяемость
- Слепое судейство с анонимизацией и рандомизацией
- Двойной прогон для проверки устойчивости

### ✅ Пункт 6: Фактчекинг (Verifier, CoVe)
- Генерация целевых вопросов для проверки
- Иерархия авторитетности источников
- Классификация статусов: supported, refuted, uncertain, conditional

### ✅ Шаблоны промптов
- Детальные инструкции для каждого агента
- Примеры форматов вывода
- Критерии и правила обработки

## Форматы данных

### claims.tsv
```
id	claim	evidence_inline	scope.time	scope.jurisdiction	scope.conditions	facts	origin	notes
C001	Утверждение...	Источник...	2023	AI/NLP		F1: 15%	Документ A, раздел 2
```

### pairs_aligned.tsv
```
pair_id	A_id	B_id	matched_keys	scope_overlap	relation	nli	rationale	conflict_type	condition_notes
P001	C001	C015	термин1, термин2		equivalent	neutral	Тезисы выражают...
```

### judgments.tsv
```
pair_id	winner	C1	C2	C3	C4	C5	notes
P007	A	A+	tie	A+	B+	A+	Кандидат-1 лучше по корректности...
```

### evidence.tsv
```
claim_id	question	status	source	date	quote
C017	Вопрос?	supported	Название (URL)	2023-01-01	Цитата из источника...
```

## Методология

Проект основан на современных подходах:

- **LLM-as-a-Judge** - использование LLM для парного сравнения
- **Chain-of-Verification (CoVe)** - целевая верификация фактов
- **Multi-agent Debate** - многокритериальный анализ через агентов
- **Toulmin Model** - структурирование тезисов с обоснованием

## Документация

Подробная документация о процессе:
- `Contaminatin_Metod_1.md` - полное описание 9 шагов процесса
- `prompt_01.md`, `prompt_02.md` - промпты для финального синтеза

## Авторы

Проект разработан для исследований в области обработки естественного языка и работы с LLM.

## Лицензия

[Укажите лицензию проекта]
