# ДЗ 1 — RAG по ТФКП

Ермолаев Фёдор Андреевич, М05-604б.

Ответы по учебнику Горяйнова и Половинкина с обоснованиями и ссылками на исходные блоки. Поиск на CPU: BM25 и символьный TF-IDF объединяются через RRF, к контексту добавляются соседние фрагменты. Qwen3.5-4B отвечает за один вызов через Chat Completions API.

## Эксперименты и результаты

На 15 открытых вопросах итоговая система выбрала правильные наборы в **10 случаях**, **Q ≈ 0.590**.

| Файл | Содержание |
|---|---|
| [experiments.csv](experiments.csv) | Метрики пяти конфигураций: без контекста, BM25, итоговая система и две абляции |
| [report.md](report.md) | Метод, сравнение, выводы абляций, три ошибки и ресурсы |
| [run_experiments.py](run_experiments.py) | Код запуска всех пяти конфигураций |
| [dev_answers.jsonl](dev_answers.jsonl) | Сохранённые ответы итоговой системы |
| [dev_trace.jsonl](dev_trace.jsonl) | Расход токенов, время, вызовы и найденные блоки |

Для остальных конфигураций опубликованы сводные метрики. Папка `results/` с ответами и журналами создаётся при повторном запуске.

## Запуск

Нужен Python 3.11. Команды ниже выполняются в активной среде из корня репозитория.

```bash
python -m pip install -r requirements.txt
python prepare_assets.py
python prepare_corpus.py --pdf data/tfkp_2025.pdf --output data/corpus.jsonl
python build_index.py --corpus data/corpus.jsonl --output index --config config.json
```

Задайте `LLM_BASE_URL`, `LLM_MODEL` и `SUBMISSION_ID`; при авторизации сервера — `LLM_API_KEY`.

```bash
python predict.py --index index --questions dev_questions.jsonl --output answers.jsonl --trace trace.jsonl --config config.json
```

[RUN.md](RUN.md) содержит настройку среды, запуск локальной Qwen на Apple Silicon Mac, подготовку вопросов, проверку ответов и команду всех экспериментов.

Модели и зависимости скачиваются до предсказания. Во время ответов сеть используется только для заданного сервера. При сборке индекса LLM не вызывается, вопросы и эталоны не индексируются.

## Данные и настройки

- [Открытые вопросы](dev_questions.jsonl) и [эталоны](dev_gold.jsonl).
- [Восемь новых вопросов](own_questions.jsonl) и [эталоны](own_gold.jsonl), по два на категорию.
- [config.json](config.json) — поиск и контекст; [runtime.json](runtime.json) — ревизии модели и параметры генерации.
- [prepare_questions.py](prepare_questions.py) пересобирает вопросы из [выданного QA](data/tfkp_15_qa.md) и корпуса.
- [rationale_review.json](rationale_review.json) — метки проверки обоснований сохранённых ответов. Для новых ответов нужна новая проверка.

Корпус, индекс, веса, окружение и большие журналы не включены. Сложные формулы PDF иногда извлекаются с ошибками.

## Ресурсы и источники

Замеры: Apple M4 Pro, 48 ГБ общей памяти. Сборка индекса — 0.38 с, пик RSS — 175 MiB; пик выделения MLX при генерации — 3.57 GiB.

- [Горяйнов, Половинкин. Лекции по ТФКП, 2025](https://new.mipt.ru/upload/mathematics/study/uchebniki/%D0%9B_%D0%A2%D0%A4%D0%9A%D0%9F_%D0%93%D0%BE%D1%80%D1%8F%D0%B9%D0%BD%D0%BE%D0%B2_%D0%9F%D0%BE%D0%BB%D0%BE%D0%B2%D0%B8%D0%BD%D0%BA%D0%B8%D0%BD_2025.pdf)
- Семинарский `6_RAG_Evals_MuSiQue.ipynb`: BM25, RRF, символьный TF-IDF и сборка контекста
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [MLX-квантизация](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit), [MLX LM](https://github.com/ml-explore/mlx-lm)
