# Запуск

Команды выполняются из корня репозитория. Нужен Python 3.11.

## Внешний сервер модели

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Задайте `LLM_BASE_URL`, `LLM_MODEL` и `SUBMISSION_ID`. Если сервер требует авторизацию, также задайте `LLM_API_KEY`. Проверяющие могут подставить свой Chat Completions API.

## Локальная модель на Apple Silicon Mac

Вместо предыдущего шага можно подготовить локальную среду и фиксированные веса:

```bash
python3 setup_local.py
.venv/bin/python local_server.py
```

Сервер работает в отдельном терминале. Для остановки нажмите Ctrl-C. Для остальных команд:

```bash
export LLM_BASE_URL=http://127.0.0.1:8765/v1
export LLM_MODEL=Qwen/Qwen3.5-4B
export SUBMISSION_ID=ermolaev-fedor-llm-rag
```

## Данные и индекс

```bash
.venv/bin/python prepare_assets.py
.venv/bin/python prepare_corpus.py --pdf data/tfkp_2025.pdf --output data/corpus.jsonl
.venv/bin/python prepare_questions.py --source data/tfkp_15_qa.md --corpus data/corpus.jsonl --questions dev_questions.jsonl --gold dev_gold.jsonl --own-questions own_questions.jsonl --own-gold own_gold.jsonl
.venv/bin/python build_index.py --corpus data/corpus.jsonl --output index --config config.json
```

PDF и токенизатор скачиваются до предсказания. Корпус, индекс и веса в репозиторий не включены.

## Предсказание

```bash
.venv/bin/python predict.py --index index --questions dev_questions.jsonl --output answers.jsonl --trace trace.jsonl --config config.json
.venv/bin/python evaluate.py --questions dev_questions.jsonl --gold dev_gold.jsonl --answers answers.jsonl --trace trace.jsonl --corpus data/corpus.jsonl
```

Сохранённый итоговый прогон проверяется отдельно:

```bash
.venv/bin/python evaluate.py --questions dev_questions.jsonl --gold dev_gold.jsonl --answers dev_answers.jsonl --trace dev_trace.jsonl --corpus data/corpus.jsonl --review rationale_review.json
```

## Пять экспериментов

```bash
.venv/bin/python run_experiments.py --index index --questions dev_questions.jsonl --gold dev_gold.jsonl --corpus data/corpus.jsonl --config config.json --output results
```

Ответы, трассы и автоматические оценки сохраняются в `results`. Прогресс печатается после каждого вопроса. Параметры генерации одинаковы во всех пяти конфигурациях.

Сохранённые метки `rationale_review.json` привязаны к SHA256 прежних ответов. Для нового прогона обоснования нужно прочитать заново; старые метки автоматически не применяются. В опубликованной таблице учтена содержательная проверка сохранённых ответов.
