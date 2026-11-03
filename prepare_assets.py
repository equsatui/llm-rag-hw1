import argparse
import hashlib
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parent
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PDF_URL = "https://new.mipt.ru/upload/mathematics/study/uchebniki/%D0%9B_%D0%A2%D0%A4%D0%9A%D0%9F_%D0%93%D0%BE%D1%80%D1%8F%D0%B9%D0%BD%D0%BE%D0%B2_%D0%9F%D0%BE%D0%BB%D0%BE%D0%B2%D0%B8%D0%BD%D0%BA%D0%B8%D0%BD_2025.pdf"
HASHES = {
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
}


def download(url, path, expected):
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Изменился исходный файл: " + path.name)
    path.write_bytes(data)
    print(path.name, len(data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=ROOT / "data" / "tfkp_2025.pdf")
    args = parser.parse_args()
    download(PDF_URL, args.pdf, "629f4b668ff43abf2a6e621aefd0f50e165ce8e88c52c05769149e1267d033ec")
    for name, sha in HASHES.items():
        download(f"https://huggingface.co/Qwen/Qwen3.5-4B/resolve/{REVISION}/{name}", ROOT / "assets" / "tokenizer" / name, sha)


if __name__ == "__main__":
    main()
