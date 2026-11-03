import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import resource
import statistics
import sys
import time

import pymupdf


EXTENSIONS = {
    0: "(", 1: ")", 2: "[", 3: "]", 8: "{", 9: "}", 12: "|",
    16: "(", 17: ")", 18: "(", 19: ")", 20: "[", 21: "]",
    26: "{", 27: "}", 30: "/", 32: "(", 33: ")", 40: "{", 41: "}",
    48: "(", 49: ")", 50: "[", 51: "]", 52: "[", 53: "]",
    56: "{", 57: "}", 58: "{", 59: "}", 60: "{", 61: "}",
    80: "∑", 81: "∏", 82: "∫", 88: "∑", 89: "∏", 90: "∫", 91: "∪",
    98: "^", 101: "~", 110: "{", 111: "}", 112: "√", 114: "√", 115: "√",
    0xF8EB: "(", 0xF8EC: "", 0xF8ED: "", 0xF8F6: ")", 0xF8F7: "", 0xF8F8: "",
}


def decode_glyph(char, font):
    if font.startswith("F") and font[1:].isdigit():
        if char == "\x01":
            return "´"
        if char == "\x16":
            return "—"
        if char == "\x15":
            return "–"
        if char == "\x9f":
            return "§"
        if 160 <= ord(char) <= 255:
            return char.encode("latin1").decode("cp1251")
    if font.startswith("CMEX"):
        return EXTENSIONS.get(ord(char), char)
    return char


def combine(items, text, y=None):
    return {
        "text": text,
        "x0": min(t["x0"] for t in items), "x1": max(t["x1"] for t in items),
        "y0": min(t["y0"] for t in items), "y1": max(t["y1"] for t in items),
        "y": statistics.median(t["y"] for t in items) if y is None else y,
        "size": max(t["size"] for t in items), "font": "", "large": False,
    }


def render_line(items, baseline=None):
    if not items:
        return ""
    size = max(t["size"] for t in items)
    if baseline is None:
        baseline = Counter(round(t["y"], 1) for t in items if t["size"] >= .85 * size).most_common(1)[0][0]
    main = [t for t in items if t["size"] >= .85 * size or abs(t["y"] - baseline) < .5]
    scripts = [t for t in items if t not in main]
    main.sort(key=lambda t: (t["x0"], t["y"]))
    groups = []
    for t in sorted(scripts, key=lambda t: (t["y"], t["x0"])):
        neighbours = [g for g in groups if abs(t["y"] - g[-1]["y"]) < .6 * size
                      and -.5 <= t["x0"] - g[-1]["x1"] < .6 * size]
        if neighbours:
            min(neighbours, key=lambda g: abs(t["x0"] - g[-1]["x1"])).append(t)
        else:
            groups.append([t])
    suffixes = {}
    for group in groups:
        g = combine(group, "")
        candidates = [t for t in main if t["x0"] - (.8 * size if t["large"] else .5) <= g["x0"] and g["x0"] - t["x1"] < .7 * size]
        candidates = [t for t in candidates if abs(g["y"] - t["y"]) < (2.2 if t["large"] else 1.1) * size]
        if candidates:
            parent = min(candidates, key=lambda t: abs(g["x0"] - t["x1"]))
            marker = "^" if g["y"] < parent["y"] else "_"
            suffixes.setdefault(id(parent), []).append(marker + "{" + render_line(group) + "}")
        else:
            main.extend(group)
    parts = []
    previous = None
    for t in sorted(main, key=lambda t: (t["x0"], t["y"])):
        if previous and t["x0"] - previous["x1"] > .17 * min(t["size"], previous["size"]):
            parts.append(" ")
        parts.append(t["text"] + "".join(suffixes.get(id(t), [])))
        previous = t
    return "".join(parts).strip()


def page_atoms(page, counts):
    atoms = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                for char in span["chars"]:
                    text = decode_glyph(char["c"], span["font"])
                    if text.isspace() or not text:
                        continue
                    x0, y0, x1, y1 = char["bbox"]
                    atoms.append({"text": text, "x0": x0, "x1": x1, "y0": y0, "y1": y1,
                                  "y": char["origin"][1], "size": span["size"], "font": span["font"],
                                  "large": span["font"].startswith("CMEX")})
    normal = [t for t in atoms if not t["large"] and t["size"] > 8]
    baselines = Counter(round(t["y"], 1) for t in normal)
    for t in atoms:
        if t["large"]:
            candidates = [(n, y) for y, n in baselines.items() if 0 <= y - t["y"] < 2 * t["size"]]
            if candidates:
                offset = 1.36 if t["text"] == "∫" else 1.05 if t["text"] in "∑∏" else 1.41
                target = t["y"] + offset * t["size"]
                t["y"] = min(candidates, key=lambda pair: (abs(pair[1] - target), -pair[0]))[1]
    rules = []
    for drawing in page.get_drawings():
        if (drawing.get("width") or 0) > 1:
            continue
        for item in drawing["items"]:
            if item[0] == "l" and abs(item[1].y - item[2].y) < .1:
                left, right = sorted([item[1].x, item[2].x])
                if right - left > 1:
                    rules.append((left, right, item[1].y))
    fractions = []
    bars = []
    for left, right, y in rules:
        inside = [t for t in atoms if not t["font"].startswith("F") and left - .4 <= (t["x0"] + t["x1"]) / 2 <= right + .4]
        upper = [t for t in inside if -.55 * t["size"] < t["y"] - y < -.05 * t["size"]]
        lower = [t for t in inside if .45 * t["size"] < t["y"] - y < 1.8 * t["size"]]
        if upper and lower:
            fractions.append((left, right, y))
        else:
            below = [t for t in inside if .35 * t["size"] < t["y"] - y < 1.1 * t["size"]]
            if below:
                baseline = statistics.median(t["y"] for t in below)
                base_size = max(t["size"] for t in below)
                below.extend(t for t in inside if t not in below and t["size"] < .85 * base_size and abs(t["y"] - baseline) < .8 * base_size)
                bars.append((left, right, y, below))
    for left, right, y, below in bars:
        below = [t for t in below if t in atoms]
        if below:
            bar = combine(below, "\\overline{" + render_line(below) + "}")
            atoms = [t for t in atoms if t not in below] + [bar]
            counts["overbars"] += 1
    for left, right, y in sorted(fractions, key=lambda r: r[1] - r[0]):
        inside = [t for t in atoms if not t["font"].startswith("F") and left - .4 <= (t["x0"] + t["x1"]) / 2 <= right + .4]
        upper = [t for t in inside if -2.5 * t["size"] < t["y"] - y < -.05 * t["size"]]
        lower = [t for t in inside if .15 * t["size"] < t["y"] - y < 2 * t["size"]]
        near_upper = [t for t in upper if y - t["y"] < .6 * t["size"]]
        near_lower = [t for t in lower if t["y"] - y < 1.8 * t["size"]]
        if not near_upper or not near_lower:
            continue
        upper_size = max(t["size"] for t in near_upper)
        lower_size = max(t["size"] for t in near_lower)
        top = Counter(round(t["y"], 2) for t in near_upper if t["size"] >= .85 * upper_size).most_common(1)[0][0]
        bottom = Counter(round(t["y"], 2) for t in near_lower if t["size"] >= .85 * lower_size).most_common(1)[0][0]
        upper = [t for t in upper if abs(t["y"] - top) < .9 * t["size"]]
        lower = [t for t in lower if abs(t["y"] - bottom) < .9 * t["size"]]
        used = upper + lower
        frac = combine(used, "\\frac{" + render_line(upper, top) + "}{" + render_line(lower, bottom) + "}", y + .25 * max(t["size"] for t in used))
        atoms = [t for t in atoms if t not in used] + [frac]
        counts["fractions"] += 1
    return atoms


def page_rows(page, counts):
    atoms = page_atoms(page, counts)
    size = statistics.median(t["size"] for t in atoms)
    main = [t for t in atoms if t["size"] >= .85 * size]
    scripts = [t for t in atoms if t["size"] < .85 * size]
    rows = []
    for t in sorted(main, key=lambda t: (t["y"], t["x0"])):
        if rows and abs(t["y"] - statistics.median(x["y"] for x in rows[-1])) < 2:
            rows[-1].append(t)
        else:
            rows.append([t])
    for t in scripts:
        operators = [(row, op) for row in rows for op in row if op["text"] in {"∫", "∑", "∏"}
                     and op["x0"] - .8 * size <= (t["x0"] + t["x1"]) / 2 <= op["x1"] + .8 * size
                     and ((op["text"] == "∫" and 3 < t["y"] - op["y"] < 2.5 * size)
                          or (op["text"] != "∫" and 2 < abs(t["y"] - op["y"]) < 2.2 * size))]
        if operators:
            row, op = min(operators, key=lambda pair: abs((pair[1]["x0"] + pair[1]["x1"] - t["x0"] - t["x1"]) / 2))
            row.append(t)
        elif rows:
            row = min(rows, key=lambda r: abs(t["y"] - statistics.median(x["y"] for x in r)))
            if abs(t["y"] - statistics.median(x["y"] for x in row)) < 18:
                row.append(t)
            else:
                rows.append([t])
        else:
            rows.append([t])
    result = []
    for row in rows:
        baseline = statistics.median(t["y"] for t in row if t["size"] >= .85 * max(x["size"] for x in row))
        text = render_line(row, baseline)
        if baseline < .065 * page.rect.height:
            counts["header_rows"] += 1
            continue
        if text:
            result.append((baseline, min(t["x0"] for t in row), text))
    return sorted(result)


def extract(pdf):
    counts = Counter()
    records = []
    section = ""
    with pymupdf.open(pdf) as document:
        for page_number, page in enumerate(document, 1):
            rows = page_rows(page, counts)
            left = min((x for y, x, text in rows if re.search("[А-Яа-я]{3}", text)), default=page.rect.width * .1)
            paragraphs = []
            lines = []
            current_section = section
            for y, x, text in rows:
                heading = re.match(r"^§\s*\d+\.", text)
                start = (re.match(r"^(Теорема|Определение|Следствие|Замечание|Доказательство)\b", text)
                         or (6 < x - left < 20 and len(re.findall("[А-Яа-я]", text)) > 12 and not re.match(r"^\([ivx]+\)", text))
                         or heading)
                if lines and start:
                    paragraphs.append((current_section, "\n".join(lines)))
                    lines = []
                if heading:
                    section = text
                if not lines:
                    current_section = section
                lines.append(text)
            if lines:
                paragraphs.append((current_section, "\n".join(lines)))
            block = 0
            for block_section, text in paragraphs:
                text = re.sub(r"([А-Яа-я])[-­]\n(?=[а-я])", r"\1", text)
                text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
                text = re.sub(r"\s+", " ", text).strip().replace("̸ =", "≠")
                pieces = []
                while len(text) > 2200:
                    cut = text.rfind(" ", 1000, 2201)
                    if cut < 0:
                        cut = 2200
                    pieces.append(text[:cut])
                    text = text[cut:].strip()
                if text:
                    pieces.append(text)
                for piece in pieces:
                    block += 1
                    records.append({"block_id": f"p{page_number:04d}_b{block:04d}", "page": page_number,
                                    "section": block_section, "text": piece})
            counts["pages"] += 1
    counts["blocks"] = len(records)
    counts["characters"] = sum(len(r["text"]) for r in records)
    return records, dict(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.pdf.resolve() == args.output.resolve():
        parser.error("PDF and output must be different files")
    started = time.perf_counter()
    records, counts = extract(args.pdf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts["pdf_sha256"] = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    counts["corpus_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    counts["seconds"] = round(time.perf_counter() - started, 3)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    counts["peak_rss_mib"] = round(peak / (1024 ** 2 if sys.platform == "darwin" else 1024), 2)
    counts["bytes"] = args.output.stat().st_size
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
