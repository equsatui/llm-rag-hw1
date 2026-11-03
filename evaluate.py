import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


CATEGORIES = ("conditions", "comparison", "computation", "insufficient")
ANSWER_FIELDS = {"id", "answer_ids", "abstain", "rationale", "citations"}
TRACE_FIELDS = {
    "id", "llm_calls", "input_tokens", "output_tokens", "elapsed_ms",
    "retrieved_block_ids",
}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Повторяющийся JSON-ключ: {key}")
        result[key] = value
    return result


def load_jsonl(path, allow_invalid=False):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line, object_pairs_hook=unique_object)
                if not isinstance(record, dict):
                    raise ValueError("Строка должна содержать JSON-объект")
            except (ValueError, TypeError) as error:
                if not allow_invalid:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                record = {"_invalid": str(error), "_line": line_number}
            records.append(record)
    return records


def string_list(value):
    return (
        isinstance(value, list)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
        and len(value) == len(set(value))
    )


def validate_questions(questions):
    seen = set()
    for question in questions:
        if set(question) != {"id", "question", "options"}:
            raise ValueError("Вопрос должен содержать только id, question, options")
        question_id = question["id"]
        if not isinstance(question_id, str) or not question_id.strip() or question_id in seen:
            raise ValueError(f"Пустой или повторный ID вопроса: {question_id!r}")
        seen.add(question_id)
        if not isinstance(question["question"], str) or not question["question"].strip():
            raise ValueError(f"{question_id}: пустой текст вопроса")
        options = question["options"]
        if not isinstance(options, list) or not 4 <= len(options) <= 6:
            raise ValueError(f"{question_id}: требуется 4–6 вариантов")
        if any(not isinstance(option, dict) or set(option) != {"id", "text"} for option in options):
            raise ValueError(f"{question_id}: неверный формат вариантов")
        if not string_list([option["id"] for option in options]):
            raise ValueError(f"{question_id}: пустые или повторные ID вариантов")
        if any(not isinstance(option["text"], str) or not option["text"].strip() for option in options):
            raise ValueError(f"{question_id}: пустой текст варианта")
    if not questions:
        raise ValueError("Набор вопросов пуст")


def validate_answer(answer, question, source_ids, no_context=False):
    errors = []
    if not isinstance(answer, dict) or set(answer) != ANSWER_FIELDS:
        return ["Неверный набор полей ответа"]
    if answer["id"] != question["id"]:
        errors.append("ID ответа не совпадает с вопросом")
    answer_ids = answer["answer_ids"]
    citations = answer["citations"]
    if not string_list(answer_ids):
        errors.append("answer_ids должен быть списком уникальных непустых строк")
    elif not set(answer_ids) <= {option["id"] for option in question["options"]}:
        errors.append("Неизвестный ID варианта")
    if type(answer["abstain"]) is not bool:
        errors.append("abstain должен быть boolean")
    if not isinstance(answer["rationale"], str) or not answer["rationale"].strip():
        errors.append("Пустое обоснование")
    elif len(answer["rationale"]) > 1200:
        errors.append("Обоснование длиннее 1200 символов")
    if not string_list(citations):
        errors.append("citations должен быть списком уникальных непустых строк")
    else:
        if len(citations) > 4:
            errors.append("Более четырёх ссылок")
        if not set(citations) <= source_ids:
            errors.append("Неизвестный исходный block_id")
    if type(answer["abstain"]) is bool and isinstance(answer_ids, list) and isinstance(citations, list):
        if answer["abstain"] and (answer_ids or citations):
            errors.append("При отказе оба списка должны быть пустыми")
        if not answer["abstain"] and (not answer_ids or (not citations and not no_context)):
            errors.append("Без отказа нужны варианты и ссылки")
    return errors


def validate_gold(gold, questions, source_ids):
    by_id = {}
    required = {"id", "category", "answer_ids", "abstain", "evidence_sets", "rationale"}
    question_map = {question["id"]: question for question in questions}
    for record in gold:
        if set(record) != required:
            raise ValueError("Неверные поля эталона")
        question_id = record["id"]
        if not isinstance(question_id, str) or question_id not in question_map or question_id in by_id:
            raise ValueError(f"Неизвестный или повторный ID эталона: {question_id!r}")
        category = record["category"]
        if category not in CATEGORIES:
            raise ValueError(f"{question_id}: неизвестная категория")
        if type(record["abstain"]) is not bool or record["abstain"] != (category == "insufficient"):
            raise ValueError(f"{question_id}: категория не согласуется с abstain")
        evidence_sets = record["evidence_sets"]
        if not isinstance(evidence_sets, list):
            raise ValueError(f"{question_id}: evidence_sets должен быть списком")
        if record["abstain"] and (record["answer_ids"] or evidence_sets):
            raise ValueError(f"{question_id}: эталон отказа должен иметь пустые списки")
        if not record["abstain"] and not evidence_sets:
            raise ValueError(f"{question_id}: отсутствуют эталонные источники")
        seen_sets = set()
        for evidence in evidence_sets:
            if not string_list(evidence) or not 1 <= len(evidence) <= 4 or not set(evidence) <= source_ids:
                raise ValueError(f"{question_id}: неверный набор источников {evidence!r}")
            key = frozenset(evidence)
            if key in seen_sets:
                raise ValueError(f"{question_id}: повторный эталонный набор")
            seen_sets.add(key)
        answer = {key: record[key] for key in ("id", "answer_ids", "abstain", "rationale")}
        answer["citations"] = evidence_sets[0] if evidence_sets else []
        errors = validate_answer(answer, question_map[question_id], source_ids)
        if errors:
            raise ValueError(f"{question_id}: неверный эталон: {'; '.join(errors)}")
        by_id[question_id] = record
    if set(by_id) != set(question_map):
        raise ValueError("В эталоне должны быть все вопросы ровно по одному разу")
    if {record["category"] for record in gold} != set(CATEGORIES):
        raise ValueError("Для Q нужны все четыре категории")
    return by_id


def citation_f1(citations, evidence_sets):
    predicted = set(citations)
    scores = []
    for evidence in evidence_sets:
        reference = set(evidence)
        denominator = len(predicted) + len(reference)
        scores.append(2 * len(predicted & reference) / denominator if denominator else 1.0)
    return max(scores, default=0.0)


def group_records(records, question_ids, name):
    grouped = {}
    file_errors = []
    for number, record in enumerate(records, 1):
        question_id = record.get("id") if isinstance(record, dict) else None
        if not isinstance(question_id, str) or question_id not in question_ids:
            file_errors.append(f"{name}, строка {number}: неизвестный ID или неверный JSON")
            continue
        grouped.setdefault(question_id, []).append(record)
    return grouped, file_errors


def validate_trace(trace, source_ids):
    errors = []
    limits = []
    if "status" in trace and trace["status"] != "ok":
        errors.append("Техническая ошибка при получении ответа")
    if trace.get("error") or trace.get("error_type"):
        errors.append("В trace записана техническая ошибка")
    if not TRACE_FIELDS <= set(trace):
        return ["Отсутствуют обязательные поля trace"], [], False
    for field in ("llm_calls", "input_tokens", "output_tokens"):
        if type(trace[field]) is not int or trace[field] < 0:
            errors.append(f"{field} должен быть неотрицательным целым числом")
    elapsed = trace["elapsed_ms"]
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        errors.append("elapsed_ms должен быть конечным неотрицательным числом")
    retrieved = trace["retrieved_block_ids"]
    if not string_list(retrieved) or not set(retrieved) <= source_ids:
        errors.append("Неверные retrieved_block_ids")
    if trace.get("usage_verified") is False:
        errors.append("Расход не подтверждён ответом endpoint")
    for field, maximum in (("llm_calls", 4), ("input_tokens", 16384), ("output_tokens", 4096)):
        value = trace[field]
        if type(value) is int and value > maximum:
            limits.append(f"{field} > {maximum}")
    if not errors and trace["llm_calls"] == 0 and (trace["input_tokens"] or trace["output_tokens"]):
        errors.append("При нуле вызовов расход токенов должен быть нулевым")
    requests = trace.get("requests")
    per_request_verified = False
    if requests is not None:
        if not isinstance(requests, list) or len(requests) != trace["llm_calls"]:
            errors.append("requests не соответствует числу вызовов")
        else:
            valid_requests = True
            for request in requests:
                fields = ("input_tokens", "output_tokens", "max_output_tokens")
                if not isinstance(request, dict) or any(type(request.get(field)) is not int or request[field] < 0 for field in fields):
                    errors.append("Неполный или неверный расход отдельного запроса")
                    valid_requests = False
                    continue
                if request["input_tokens"] + request["max_output_tokens"] > 8192:
                    limits.append("Вход + допустимый выход запроса > 8192")
                if request["output_tokens"] > request["max_output_tokens"]:
                    limits.append("Фактический выход превышает допустимый выход запроса")
                if request.get("usage_verified") is False:
                    errors.append("Расход отдельного запроса не подтверждён")
            if valid_requests:
                for field in ("input_tokens", "output_tokens"):
                    if sum(request[field] for request in requests) != trace[field]:
                        errors.append(f"Сумма requests не совпадает с {field}")
                per_request_verified = True
    elif trace["llm_calls"] == 0:
        per_request_verified = True
    return errors, limits, per_request_verified


def evaluate_run(questions, gold, answers, trace, corpus=None, no_context=False, reference_cost=None):
    validate_questions(questions)
    if corpus is None:
        if not no_context:
            raise ValueError("Для строгой проверки block_id требуется корпус")
        source_ids = set()
        for record in gold:
            for evidence in record.get("evidence_sets", []):
                source_ids.update(evidence)
    else:
        source_ids = set()
        for block in corpus:
            block_id = block.get("block_id")
            if not isinstance(block_id, str) or not block_id or block_id in source_ids:
                raise ValueError("Пустой или повторный block_id корпуса")
            source_ids.add(block_id)
    gold_map = validate_gold(gold, questions, source_ids)
    question_ids = {question["id"] for question in questions}
    answer_map, answer_file_errors = group_records(answers, question_ids, "answers")
    trace_map, trace_file_errors = group_records(trace, question_ids, "trace")
    file_errors = answer_file_errors + trace_file_errors
    by_category = {category: [] for category in CATEGORIES}
    details = []
    usage = []
    latencies = []
    efficiency_eligible = not trace_file_errors
    all_request_limits_verified = True
    for question in questions:
        question_id = question["id"]
        reference = gold_map[question_id]
        errors = []
        limits = []
        candidates = answer_map.get(question_id, [])
        answer = candidates[0] if len(candidates) == 1 else None
        if answer is None:
            errors.append("Нет ответа или повторный ID ответа")
        else:
            errors.extend(validate_answer(answer, question, source_ids, no_context))
        traces = trace_map.get(question_id, [])
        current_trace = traces[0] if len(traces) == 1 else None
        if current_trace is None:
            errors.append("Нет trace или повторный ID trace")
            efficiency_eligible = False
            all_request_limits_verified = False
        else:
            trace_errors, limits, request_verified = validate_trace(current_trace, source_ids)
            errors.extend(trace_errors)
            all_request_limits_verified &= request_verified
            if limits:
                efficiency_eligible = False
            usage_valid = all(type(current_trace.get(field)) is int and current_trace[field] >= 0
                              for field in ("llm_calls", "input_tokens", "output_tokens"))
            usage_valid &= current_trace.get("usage_verified") is not False
            if usage_valid and current_trace["llm_calls"] == 0:
                usage_valid = current_trace["input_tokens"] == current_trace["output_tokens"] == 0
            requests = current_trace.get("requests")
            if requests is not None:
                if not isinstance(requests, list) or len(requests) != current_trace.get("llm_calls"):
                    usage_valid = False
                elif any(not isinstance(request, dict) or request.get("usage_verified") is False or
                         any(type(request.get(field)) is not int or request[field] < 0
                             for field in ("input_tokens", "output_tokens", "max_output_tokens"))
                         for request in requests):
                    usage_valid = False
                elif usage_valid:
                    usage_valid = all(sum(request[field] for request in requests) == current_trace[field]
                                      for field in ("input_tokens", "output_tokens"))
            if usage_valid:
                usage.append(current_trace)
            else:
                efficiency_eligible = False
            elapsed = current_trace.get("elapsed_ms")
            if type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0:
                latencies.append(elapsed)
        if file_errors:
            errors.append("Нарушен формат файла: есть лишние или неразбираемые записи")
        valid = not errors and not limits
        correct = bool(valid and answer["abstain"] == reference["abstain"] and set(answer["answer_ids"]) == set(reference["answer_ids"]))
        f1 = citation_f1(answer["citations"], reference["evidence_sets"]) if valid and not reference["abstain"] else None
        score = 0.0
        if correct:
            score = 1.0 if reference["abstain"] else 0.8 + 0.2 * f1
        by_category[reference["category"]].append(score)
        details.append({"id": question_id, "category": reference["category"], "valid": valid,
                        "answer_correct": correct, "citation_f1": f1, "score": None if no_context else score,
                        "errors": errors, "limit_violations": limits})
    count = len(questions)
    complete_usage = len(usage) == count and not trace_file_errors
    means = {}
    for field in ("input_tokens", "output_tokens", "llm_calls"):
        means[f"mean_{field}"] = sum(record[field] for record in usage) / count if complete_usage else None
    mean_cost = sum(record["input_tokens"] + 4 * record["output_tokens"] + 256 * record["llm_calls"] for record in usage) / count if complete_usage else None
    category_scores = {category: statistics.mean(scores) for category, scores in by_category.items()}
    quality = None if no_context else statistics.mean(category_scores.values())
    efficiency_eligible &= all_request_limits_verified
    efficiency = None
    final_score = None
    if reference_cost is not None:
        if type(reference_cost) not in (int, float) or not math.isfinite(reference_cost) or reference_cost < 0:
            raise ValueError("reference_cost должен быть конечным неотрицательным числом")
        efficiency = min(1.0, reference_cost / max(mean_cost, 1)) if efficiency_eligible and mean_cost is not None else 0.0
        if quality is not None:
            final_score = 75 * quality + 25 * quality * efficiency
    return {
        "n_questions": count, "n_valid": sum(detail["valid"] for detail in details),
        "answer_accuracy": sum(detail["answer_correct"] for detail in details) / count,
        "Q": quality, "category_scores": None if no_context else category_scores,
        **means, "mean_cost": mean_cost,
        "median_latency_ms": statistics.median(latencies) if len(latencies) == count and not trace_file_errors else None,
        "efficiency_eligible": efficiency_eligible and complete_usage,
        "per_request_limits_verified": all_request_limits_verified,
        "efficiency": efficiency, "final_score": final_score,
        "rationale_semantics_checked": False,
        "file_errors": file_errors, "per_question": details,
    }


def apply_rationale_review(result, review, answer_path):
    digest = hashlib.sha256(Path(answer_path).read_bytes()).hexdigest()
    if digest != review["answers_sha256"]:
        raise ValueError("Смысловая проверка относится к другим ответам")
    known = {row["id"] for row in result["per_question"]}
    if not set(review["errors"]) <= known:
        raise ValueError("Неизвестный ID в смысловой проверке")
    if result["Q"] is None:
        return result
    result["Q_before_rationale_review"] = result["Q"]
    for row in result["per_question"]:
        if row["id"] in review["errors"]:
            row["score"] = 0.0
            row["rationale_error"] = review["errors"][row["id"]]
    result["category_scores"] = {category: statistics.mean(row["score"] for row in result["per_question"] if row["category"] == category) for category in CATEGORIES}
    result["Q"] = statistics.mean(result["category_scores"].values())
    result["rationale_semantics_checked"] = True
    if result["efficiency"] is not None:
        result["final_score"] = 75 * result["Q"] + 25 * result["Q"] * result["efficiency"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--corpus")
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--reference-cost", type=float)
    parser.add_argument("--review")
    parser.add_argument("--review-name", default="final")
    args = parser.parse_args()
    corpus_path = Path(args.corpus) if args.corpus else Path(args.gold).resolve().parent / "data" / "corpus.jsonl"
    corpus = load_jsonl(corpus_path) if corpus_path.exists() else None
    if args.corpus and corpus is None:
        raise FileNotFoundError(corpus_path)
    result = evaluate_run(
        load_jsonl(args.questions), load_jsonl(args.gold),
        load_jsonl(args.answers, allow_invalid=True), load_jsonl(args.trace, allow_invalid=True),
        corpus, args.no_context, args.reference_cost,
    )
    if args.review:
        review = json.loads(Path(args.review).read_text(encoding="utf-8"))
        result = apply_rationale_review(result, review[args.review_name], args.answers)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
