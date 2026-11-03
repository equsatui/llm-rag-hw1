import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import joblib
import json_repair

from rag import count_prompt, load_config, load_tokenizer, make_messages, pack_context, read_jsonl, retrieve


def validate_question(question):
    if not isinstance(question.get("id"), str) or not question["id"] or not isinstance(question.get("question"), str) or not question["question"].strip():
        raise ValueError("Некорректный вопрос")
    options = question.get("options")
    if not isinstance(options, list) or not 4 <= len(options) <= 6:
        raise ValueError("Нужно от 4 до 6 вариантов")
    if any(not isinstance(o.get("id"), str) or not o["id"] or not isinstance(o.get("text"), str) or not o["text"].strip() for o in options):
        raise ValueError("Некорректные варианты")
    if len({o["id"] for o in options}) != len(options):
        raise ValueError("Повторяющиеся ID вариантов")


def validate_response(data, question, shown, no_context=False):
    if not isinstance(data, dict) or set(data) != {"answer_ids", "abstain", "rationale", "citations"}:
        raise ValueError("Некорректные поля JSON")
    if type(data["abstain"]) is not bool or not isinstance(data["rationale"], str) or not data["rationale"].strip() or len(data["rationale"]) > 1200:
        raise ValueError("Некорректный отказ или обоснование")
    options = {o["id"] for o in question["options"]}
    for key, allowed in [("answer_ids", options), ("citations", shown)]:
        values = data[key]
        if not isinstance(values, list) or any(not isinstance(x, str) for x in values) or len(values) != len(set(values)) or not set(values) <= allowed:
            raise ValueError("Некорректные " + key)
    if len(data["citations"]) > 4:
        raise ValueError("Слишком много ссылок")
    if data["abstain"] and (data["answer_ids"] or data["citations"]):
        raise ValueError("При отказе списки должны быть пустыми")
    if not data["abstain"] and (not data["answer_ids"] or (not no_context and not data["citations"])):
        raise ValueError("Неполный ответ")
    return {"id": question["id"], **data}


def collect_decisions(data, question):
    if not isinstance(data, dict) or not isinstance(data.get("checks"), dict):
        raise ValueError("Нужны решения для всех вариантов")
    checks = data["checks"]
    if set(checks) != {o["id"] for o in question["options"]} or any(type(v) is not bool for v in checks.values()):
        raise ValueError("Некорректные решения по вариантам")
    selected = [o["id"] for o in question["options"] if checks[o["id"]]]
    return {"answer_ids": selected, "abstain": not selected, "rationale": data.get("rationale"),
            "citations": data.get("citations") if selected else []}


def parse_json(content, trace, purpose):
    if not isinstance(content, str):
        raise ValueError("Endpoint вернул нетекстовый ответ")
    if content.strip().startswith("```"):
        content = "\n".join(content.strip().splitlines()[1:-1])
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        trace.setdefault("json_repairs", []).append(purpose)
        return json_repair.loads(content)


def complete(messages, question_id, tokenizer, cfg, trace, purpose):
    runtime = cfg["runtime_settings"]
    estimated = count_prompt(messages, tokenizer, cfg)
    maximum = runtime["generation"]["max_tokens"]
    if trace["llm_calls"] >= 4 or estimated + maximum + 64 > 8192 or trace["input_tokens"] + estimated + 64 > 16384 or trace["output_tokens"] + maximum > 4096:
        raise ValueError("Исчерпан токенный бюджет")
    payload = {"model": os.environ.get("LLM_MODEL", runtime["model"]), "messages": messages,
               **runtime["generation"], "chat_template_kwargs": {"enable_thinking": runtime["thinking"]}, "stream": False}
    headers = {"Content-Type": "application/json", "X-Question-ID": question_id, "X-Submission-ID": os.environ["SUBMISSION_ID"]}
    if os.environ.get("LLM_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["LLM_API_KEY"]
    request = urllib.request.Request(os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions",
                                     data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers, method="POST")
    trace["llm_calls"] += 1
    event = {"purpose": purpose, "input_tokens_before_request": estimated, "max_output_tokens": maximum}
    trace["requests"].append(event)
    try:
        with urllib.request.urlopen(request, timeout=runtime["timeout_seconds"]) as response:
            result = json.load(response)
        if not isinstance(result, dict) or not isinstance(result.get("usage"), dict):
            raise ValueError("Некорректный объект ответа endpoint")
        usage = result["usage"]
        ni, no = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if type(ni) is not int or type(no) is not int or ni < 0 or no < 0:
            raise ValueError("Endpoint не вернул проверяемый usage")
    except (urllib.error.URLError, TimeoutError, ValueError):
        trace.update(input_tokens=None, output_tokens=None)
        raise
    event.update(input_tokens=ni, output_tokens=no)
    trace["input_tokens"] += ni
    trace["output_tokens"] += no
    if ni + maximum > 8192 or trace["input_tokens"] > 16384 or trace["output_tokens"] > 4096 or no > maximum:
        raise ValueError("Endpoint превысил токенный бюджет")
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("Endpoint вернул некорректный choices")
    if not isinstance(choices[0].get("message"), dict):
        raise ValueError("Endpoint вернул некорректное message")
    if choices[0].get("finish_reason") == "length":
        raise ValueError("Ответ обрезан по лимиту токенов")
    return result["choices"][0]["message"]["content"]


def predict_one(question, index, tokenizer, cfg):
    started = time.perf_counter()
    trace = {"id": question["id"], "llm_calls": 0, "input_tokens": 0, "output_tokens": 0,
             "elapsed_ms": 0, "retrieved_block_ids": [], "status": "started", "requests": [],
             "configuration": cfg["name"], "model_revision": cfg["runtime_settings"]["model_revision"],
             "model_repository": cfg["runtime_settings"]["local_model"],
             "prompt_budget_tokenizer_revision": cfg["runtime_settings"]["prompt_budget_tokenizer_revision"]}
    try:
        parts, ranked = [], []
        no_context = cfg["name"] == "no_context"
        if not no_context:
            order, ranked = retrieve(question, index, cfg)
            parts = pack_context(question, order, index, tokenizer, cfg)
        shown = {p["block_id"] for p in parts}
        trace["retrieved_block_ids"] = list(dict.fromkeys(p["block_id"] for p in parts))
        trace["candidate_block_ids"] = [list(dict.fromkeys(p["block_id"] for p in index["chunks"][i]["parts"])) for i in ranked]
        content = complete(make_messages(question, parts, no_context), question["id"], tokenizer, cfg, trace, "answer")
        trace["response_text"] = content
        answer = validate_response(collect_decisions(parse_json(content, trace, "answer"), question), question, shown, no_context)
        trace["status"] = "ok"
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, TypeError, IndexError) as error:
        trace["status"] = "error"
        trace["error_type"] = type(error).__name__
        trace["error"] = str(error) if isinstance(error, ValueError) else type(error).__name__
        answer = {"id": question["id"], "answer_ids": [], "abstain": True, "rationale": "Не удалось получить корректный ответ модели.", "citations": []}
    trace["elapsed_ms"] = (time.perf_counter() - started) * 1000
    return answer, trace


def run_predictions(index, questions, output, trace_path, cfg):
    if not os.environ.get("LLM_BASE_URL") or not os.environ.get("SUBMISSION_ID"):
        raise ValueError("Нужны LLM_BASE_URL и SUBMISSION_ID")
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Повторяющиеся ID вопросов")
    for question in questions:
        validate_question(question)
    tokenizer = load_tokenizer(cfg["tokenizer_dir"])
    for path in [output, trace_path]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    answers, traces = [], []
    with Path(output).open("w", encoding="utf-8") as answer_file, Path(trace_path).open("w", encoding="utf-8") as trace_file:
        for i, question in enumerate(questions, 1):
            answer, trace = predict_one(question, index, tokenizer, cfg)
            answer_file.write(json.dumps(answer, ensure_ascii=False) + "\n")
            trace_file.write(json.dumps(trace, ensure_ascii=False) + "\n")
            answer_file.flush()
            trace_file.flush()
            answers.append(answer)
            traces.append(trace)
            print(f"{cfg['name']}: {i}/{len(questions)}, {trace['status']}, {trace['elapsed_ms'] / 1000:.1f} с", flush=True)
    return answers, traces


def main():
    parser = argparse.ArgumentParser()
    for name in ["index", "questions", "output", "trace", "config"]:
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    index = joblib.load(Path(args.index) / "index.joblib")
    if index["build_settings"] != {"chunk_chars": cfg["chunk_chars"]}:
        raise ValueError("Индекс собран с другими настройками чанкинга")
    run_predictions(index, read_jsonl(args.questions), args.output, args.trace, cfg)


if __name__ == "__main__":
    main()
