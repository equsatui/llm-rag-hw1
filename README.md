# ДЗ 1 — RAG по ТФКП

Ермолаев Фёдор Андреевич, М05-604б.

Python 3.11. Генерация — Qwen3.5-4B через OpenAI-совместимый Chat Completions API. Поиск выполняется на CPU: BM25 и символьный TF-IDF объединяются через RRF. Простой RAG использует BM25; итоговая система добавляет символьный поиск и соседние фрагменты. На вопрос выполняется один вызов модели, без повторных попыток. Модель отдельно отмечает истинность каждого варианта, затем код собирает требуемый JSON. Синтаксические дефекты JSON исправляет готовая библиотека `json_repair`; типы, ID, ссылки и лимиты проверяются после неё.

Измерения выполнены на Apple M4 Pro с 48 ГБ общей памяти: сборка индекса 0.38 с, пик RSS 175 MiB; пик выделения MLX при генерации 3.57 GiB.

## Подготовка и запуск

```bash
python -m pip install -r requirements.txt
python prepare_assets.py
python prepare_corpus.py --pdf data/tfkp_2025.pdf --output data/corpus.jsonl
python prepare_questions.py --source data/tfkp_15_qa.md --corpus data/corpus.jsonl --questions dev_questions.jsonl --gold dev_gold.jsonl
python build_index.py --corpus data/corpus.jsonl --output index --config config.json
```

Задайте `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` при необходимости авторизации и `SUBMISSION_ID`. Например, для локального запуска:

```bash
export LLM_BASE_URL=http://127.0.0.1:8765/v1
export LLM_MODEL=Qwen/Qwen3.5-4B
export SUBMISSION_ID=ermolaev-fedor-llm-rag
python predict.py --index index --questions dev_questions.jsonl --output answers.jsonl --trace trace.jsonl --config config.json
```

Все зависимости и файлы токенизатора загружаются до предсказания. При ответах код обращается только к `LLM_BASE_URL`. Во время сборки индекса вызовов LLM нет. В индекс не входят вопросы и эталоны. Пути в конфигурации отсчитываются от самого `config.json`.

## Локальная модель на Mac

```bash
python setup_local.py
.venv/bin/python local_server.py
```

Скрипт создаёт собственную Python 3.11 среду и скачивает фиксированную MLX-квантизацию Qwen3.5-4B. Сервер слушает только `127.0.0.1:8765`, кэш между вопросами отключён. Для внешнего endpoint эти файлы и MLX не нужны. Подробные команды — в `RUN.md`.

## Проверка и эксперименты

```bash
python evaluate.py --questions dev_questions.jsonl --gold dev_gold.jsonl --answers dev_answers.jsonl --trace dev_trace.jsonl --corpus data/corpus.jsonl --review rationale_review.json
python run_experiments.py --index index --questions dev_questions.jsonl --gold dev_gold.jsonl --corpus data/corpus.jsonl --config config.json --output results
```

В `results` сохраняются пять конфигураций и таблица `experiments.csv`: без контекста, простой RAG, итоговая система, без символьного поиска, без соседних фрагментов. Все используют одинаковые вопросы и параметры генерации. В итоговую сдачу входят `dev_answers.jsonl`, `dev_trace.jsonl`, `experiments.csv` и `report.md`.

`prepare_questions.py` пересобирает открытый набор из выданного `tfkp_15_qa.md`; пример с путями приведён в `RUN.md`. Восемь собственных вопросов и их эталоны находятся в `own_questions.jsonl` и `own_gold.jsonl`.

Индекс, корпус, веса, окружение, кэши и большие журналы исключены через `.gitignore`. Формулы восстанавливаются из глифов и геометрии PDF; сложные вложенные формулы могут содержать ошибки, ограничения разобраны в `report.md`.

## Материалы

- [Горяйнов, Половинкин. Лекции по ТФКП, 2025](https://new.mipt.ru/upload/mathematics/study/uchebniki/%D0%9B_%D0%A2%D0%A4%D0%9A%D0%9F_%D0%93%D0%BE%D1%80%D1%8F%D0%B9%D0%BD%D0%BE%D0%B2_%D0%9F%D0%BE%D0%BB%D0%BE%D0%B2%D0%B8%D0%BD%D0%BA%D0%B8%D0%BD_2025.pdf)
- `6_RAG_Evals_MuSiQue.ipynb` из материалов курса: BM25, RRF, символьный TF-IDF, сборка контекста и сравнение конфигураций
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [MLX-квантизация](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit), [MLX LM](https://github.com/ml-explore/mlx-lm)

Смысловые метки в `rationale_review.json` относятся только к сохранённым ответам с указанными SHA256. При новом endpoint или изменении ответов нужна новая проверка обоснований.
