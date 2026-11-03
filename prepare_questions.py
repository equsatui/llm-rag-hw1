import argparse
import json
from pathlib import Path
import re
import sys

from evaluate import validate_gold, validate_questions


CATEGORY_NAMES = {
    "условия": "conditions",
    "сопоставление": "comparison",
    "применение": "computation",
    "недостаточность данных": "insufficient",
}


def parse_source(text):
    questions = []
    gold = []
    matches = list(re.finditer(r"^## (\d+)\. (.+)$", text, re.MULTILINE))
    if [int(match[1]) for match in matches] != list(range(1, 16)):
        raise ValueError("В исходнике ожидаются ровно 15 пронумерованных вопросов")
    for position, match in enumerate(matches):
        section = text[match.end():matches[position + 1].start() if position + 1 < len(matches) else len(text)]
        category = re.search(r"\*\*Тип:\*\* ([^.]+)\.", section)
        question = re.search(r"\*\*Вопрос\.\*\*\s*(.*?)\n(?=- \*\*)", section, re.DOTALL)
        options = re.findall(r"^- \*\*([A-Z])\.\*\* (.+)$", section, re.MULTILINE)
        answer = re.search(r"\*\*Ответ:\*\* ([^\n]+)", section)
        source = re.search(r"\*\*Источник:\*\* ([^\n]+)", section)
        if not all((category, question, options, answer, source)):
            raise ValueError(f"Вопрос {match[1]}: не удалось разобрать исходник")
        answer_text = answer[1].rstrip(".").strip()
        abstain = answer_text == "Отказ"
        answer_ids = [] if abstain else [part.strip() for part in answer_text.split(",")]
        rationale = section[answer.end():source.start()].strip()
        question_id = f"q_{int(match[1]):03d}"
        questions.append({"id": question_id, "question": question[1].strip(),
                          "options": [{"id": option_id, "text": value} for option_id, value in options]})
        gold.append({"id": question_id, "category": CATEGORY_NAMES[category[1]],
                     "answer_ids": answer_ids, "abstain": abstain,
                     "evidence_sets": [], "rationale": rationale})
    validate_questions(questions)
    return questions, gold


def read_corpus(path):
    blocks = []
    seen = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            block = json.loads(line)
            if not isinstance(block, dict) or not {"block_id", "page", "section", "text"} <= set(block):
                raise ValueError("Неверная строка корпуса")
            if not isinstance(block["block_id"], str) or not block["block_id"] or block["block_id"] in seen:
                raise ValueError("Пустой или повторный block_id")
            if type(block["page"]) is not int or block["page"] < 1:
                raise ValueError("page должен быть положительным целым числом")
            if not isinstance(block["text"], str) or not block["text"].strip():
                raise ValueError("Пустой текст блока")
            seen.add(block["block_id"])
            blocks.append(block)
    return blocks


def normalized(text):
    return re.sub(r"\s+", " ", text).casefold().replace("ё", "е")


def find_fact(corpus, page, anchor):
    matches = [block for block in corpus if block["page"] == page and normalized(anchor) in normalized(block["text"])]
    if len(matches) != 1:
        raise ValueError(f"Источник PDF с. {page}, {anchor!r}: найдено блоков {len(matches)}, требуется ручная сверка")
    return matches[0]["block_id"]


def bind_gold(gold, corpus):
    facts = {name: find_fact(corpus, page, anchor) for name, (page, anchor) in FACTS.items()}
    for name, block_id in facts.items():
        page, anchor = FACTS[name]
        print(json.dumps({"fact": name, "page": page, "anchor": anchor, "block_id": block_id}, ensure_ascii=False), file=sys.stderr)
    for number, reference in enumerate(gold, 1):
        if not reference["abstain"]:
            reference["evidence_sets"] = [[facts[name] for name in group] for group in DEV_EVIDENCE[number]]
        print(json.dumps({"id": reference["id"], "evidence_sets": reference["evidence_sets"]}, ensure_ascii=False), file=sys.stderr)
    return facts


def own_data(facts):
    questions = []
    gold = []
    for number, (category, question, options, answers, evidence, rationale) in enumerate(OWN_DATA, 1):
        question_id = f"own_{number:03d}"
        questions.append({"id": question_id, "question": question,
                          "options": [{"id": chr(65 + i), "text": option} for i, option in enumerate(options)]})
        gold.append({"id": question_id, "category": category, "answer_ids": answers,
                     "abstain": category == "insufficient",
                     "evidence_sets": [[facts[name] for name in group] for group in evidence],
                     "rationale": rationale})
    return questions, gold


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--own-questions")
    parser.add_argument("--own-gold")
    args = parser.parse_args()
    if bool(args.own_questions) != bool(args.own_gold):
        parser.error("--own-questions и --own-gold указываются вместе")
    outputs = [args.questions, args.gold] + ([args.own_questions, args.own_gold] if args.own_questions else [])
    paths = [Path(path).resolve() for path in outputs]
    if len(set(paths)) != len(paths) or set(paths) & {Path(args.source).resolve(), Path(args.corpus).resolve()}:
        parser.error("Исходники и выходные файлы должны иметь разные пути")
    questions, gold = parse_source(Path(args.source).read_text(encoding="utf-8"))
    corpus = read_corpus(args.corpus)
    facts = bind_gold(gold, corpus)
    source_ids = {block["block_id"] for block in corpus}
    validate_gold(gold, questions, source_ids)
    own_questions, own_gold = own_data(facts)
    validate_questions(own_questions)
    validate_gold(own_gold, own_questions, source_ids)
    write_jsonl(args.questions, questions)
    write_jsonl(args.gold, gold)
    if args.own_questions:
        write_jsonl(args.own_questions, own_questions)
        write_jsonl(args.own_gold, own_gold)


FACTS = {
    "cr": (11, "Теорема 2.1."),
    "holomorphic": (11, "Определение 2.2."),
    "wirtinger": (12, "Наглядный способ выделить"),
    "laplace": (13, "Система уравнений Коши—Римана (2.3) обладает"),
    "harmonic": (13, "Определение 2.3."),
    "zero_derivative": (15, "Теорема 2.4."),
    "inverse": (15, "Теорема 2.5"),
    "power_radius": (17, "Теорема 3.1."),
    "power_properties": (18, "(ii) Если |z| > R"),
    "geometric": (17, "Рассмотрим один простой, но важный"),
    "exponential": (20, "Экспонента."),
    "trigonometric": (21, "Тригонометрические функции."),
    "log_values": (22, "Первое уравнение имеет единственное решение"),
    "log_branch": (22, "Допустим теперь, что в некоторой области"),
    "complex_power": (23, "Комплексные степени."),
    "primitive": (26, "Теорема 4.1."),
    "circle_integral": (28, "Наконец, рассмотрим в качестве γ окружность"),
    "exception_point": (30, "Теорема 4.4."),
    "local_primitive": (31, "Замечание 4.1."),
    "cauchy_conditions": (31, "Теорема 5.1"),
    "cauchy_holomorphic": (32, "является голоморфной в"),
    "cauchy_derivatives": (33, "Следствие 5.1."),
    "derivative_formula": (34, "Теорема 5.3."),
    "morera": (35, "Теорема 5.4"),
    "taylor": (36, "Теорема 6.1."),
    "taylor_unique": (37, "Замечание 6.1."),
    "entire_examples": (37, "К целым функциям относятся"),
    "growth": (37, "Теорема 6.2."),
    "identity": (38, "Теорема 6.3"),
}
DEV_EVIDENCE = {
    1: [["cr", "holomorphic"]],
    2: [["wirtinger"]],
    3: [["cr", "harmonic"]],
    4: [["primitive", "circle_integral", "local_primitive"]],
    5: [["exception_point"]],
    6: [["cauchy_conditions", "cauchy_derivatives"],
        ["cauchy_conditions", "cauchy_holomorphic", "cauchy_derivatives"]],
    7: [["exponential", "derivative_formula"]],
    8: [["morera"]],
    9: [["cr", "laplace", "derivative_formula"],
        ["cr", "harmonic", "laplace", "derivative_formula"]],
    10: [["taylor", "taylor_unique"]],
    11: [["growth"]],
    12: [["trigonometric", "entire_examples", "identity"]],
}
OWN_DATA = [
    (
        "conditions",
        "Пусть f голоморфна в области D и имеет непрерывную производную, z₀ ∈ D, w₀=f(z₀), f′(z₀)≠0. Какие выводы даёт теорема об обратной функции?",
        ["В некоторых окрестностях z₀ и w₀ существует голоморфная обратная функция g.",
         "Для этой обратной функции g′(w₀)=1/f′(z₀).",
         "f обязательно взаимно однозначна сразу во всей D.",
         "Для применения сформулированной теоремы условие f′(z₀)≠0 можно убрать."],
        ["A", "B"], [["inverse"]],
        "Теорема 2.5 даёт локальную обратимость при ненулевой производной и формулу g′(w₀)=1/f′(z₀). Глобальная однолистность не следует: z² имеет ненулевую производную в 1, но одинаковые значения в 1 и −1. При нулевой производной гипотеза теоремы нарушена.",
    ),
    (
        "conditions",
        "Степенной ряд ∑ₙ₌₀∞ aₙzⁿ имеет конечный радиус сходимости R>0. Какие свойства гарантирует теорема о степенном ряде?",
        ["Ряд сходится абсолютно и равномерно в каждом круге |z|≤r<R.",
         "Сумма голоморфна при |z|<R и допускает почленное дифференцирование там.",
         "Ряд обязательно сходится в каждой точке окружности |z|=R.",
         "При |z|>R ряд обязательно сходится условно."],
        ["A", "B"], [["power_radius", "power_properties"]],
        "Теорема 3.1 утверждает абсолютную и равномерную сходимость на меньших замкнутых кругах и голоморфность суммы с почленным дифференцированием внутри круга сходимости. Вне него ряд расходится; всеобщей сходимости на границе теорема не обещает.",
    ),
    (
        "comparison",
        "D состоит из двух непересекающихся открытых кругов. Функция f голоморфна на D, и f′=0 всюду на D. Какие утверждения верны?",
        ["Во всём D f обязательно равна одной и той же константе.",
         "На каждом из двух кругов f постоянна.",
         "Константы на двух кругах могут различаться.",
         "Нулевая производная означает f≡0."],
        ["B", "C"], [["zero_derivative"]],
        "Теорему 2.4 применяем к каждому кругу как к отдельной области. Она даёт постоянство на каждом из них. Связности объединения нет, поэтому допустима, например, функция 0 на одном круге и 1 на другом. Нулевую константу теорема не требует.",
    ),
    (
        "comparison",
        "В D=ℂ∖(−∞,0] выбрана ветвь L(z)=ln|z|+i arg z с −π<arg z<π. Какие утверждения о ней и многозначном логарифме верны?",
        ["L голоморфна в D и L′(z)=1/z.",
         "Множество Ln{z} в каждой точке z∈D состоит только из L(z).",
         "L(z)+2πi тоже является непрерывной ветвью логарифма в D.",
         "Равенство L(z₁z₂)=L(z₁)+L(z₂) верно всегда, когда все три аргумента принадлежат D."],
        ["A", "C"], [["log_values", "log_branch"]],
        "На с. 22 ветвь L голоморфна и имеет производную 1/z, а все значения логарифма отличаются на 2πik. Поэтому добавление 2πi сохраняет ветвь. Для z₁=z₂=exp(3πi/4) получаем L(z₁z₂)=−πi/2, но L(z₁)+L(z₂)=3πi/2; этот контрпример построен для вопроса.",
    ),
    (
        "computation",
        "Найдите сумму геометрического ряда ∑ₙ₌₀∞ (i/2)ⁿ.",
        ["(4+2i)/5", "(4−2i)/5", "2i", "Ряд расходится"],
        ["A"], [["geometric"]],
        "Поскольку |i/2|=1/2<1, сумма равна 1/(1−i/2). Домножая числитель и знаменатель на 1+i/2, получаем (1+i/2)/(5/4)=(4+2i)/5.",
    ),
    (
        "computation",
        "Комплексная степень iⁱ понимается как множество значений exp(i Ln{i}), без выбора ветви логарифма. Какие из следующих чисел принадлежат этому множеству?",
        ["exp(−π/2)", "exp(3π/2)", "−exp(−π/2)", "i"],
        ["A", "B"], [["log_values", "complex_power"]],
        "Все логарифмы i равны i(π/2+2πk), k∈ℤ. Значит, значения iⁱ равны exp(−π/2−2πk), то есть положительны и вещественны. A получается при k=0, B при k=−1; отрицательное число и i не подходят.",
    ),
    (
        "insufficient",
        "Функция F голоморфна на ℂ, причём F′(z)=2z для всех z. Чему точно равно F(0)? Выберите одно значение, обязательно следующее из данных; если оно не определяется, верните отказ.",
        ["0", "1", "−1", "2"],
        [], [],
        "Не задана константа интегрирования. Из (F(z)−z²)′=0 и теоремы 2.4 следует F(z)=z²+C. Функции z² и z²+1 удовлетворяют условию, но имеют разные значения в нуле. Нужна нормировка, например значение F в одной точке.",
    ),
    (
        "insufficient",
        "L — некоторая непрерывная ветвь логарифма в круге |z−1|<1/2, то есть exp(L(z))=z. Нормировка ветви не задана. Чему точно равно L(1)? Требуется одно определённое значение; если данных недостаточно, верните отказ.",
        ["0", "2πi", "−2πi", "πi"],
        [], [],
        "Не выбрана нормировка ветви. Описанную на с. 22 ветвь с L(1)=0 можно ограничить данным кругом, а прибавление 2πik даёт другие непрерывные ветви. Поэтому в точке 1 возможны 0, 2πi, −2πi и другие 2πik. Одно значение без дополнительного условия не определяется.",
    ),
]


if __name__ == "__main__":
    main()
