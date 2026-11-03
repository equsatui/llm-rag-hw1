import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).absolute().parent
REPO = 'mlx-community/Qwen3.5-4B-MLX-4bit'
REVISION = '32f3e8ecf65426fc3306969496342d504bfa13f3'
UV_VERSION = '0.13.6'
UV_SHA256 = '60fbd5c8a1ed681845eea28d9d622fcde03c51e91a43465b64a1e17db1ed3472'
WEIGHTS_SHA256 = '5fb9acd0246866381cf8c5c354c6db1019f6498eec4ccb4f5edcc71ffeacb2db'
TOKENIZER_SHA256 = '87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4'
MODEL = ROOT / 'models' / 'Qwen3.5-4B-MLX-4bit'
FILES = ['config.json', 'tokenizer_config.json', 'chat_template.jinja',
         'model.safetensors.index.json', 'tokenizer.json', 'vocab.json',
         'model.safetensors']


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(url, path, expected=None):
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError(f'Symbolic link is not allowed: {path}')
    if path.exists() and (expected is None or sha256(path) == expected):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.part')
    print(f'Downloading {path.name}', flush=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=120) as response, partial.open('wb') as target:
        downloaded = 0
        reported = 0
        for chunk in iter(lambda: response.read(8 * 1024 * 1024), b''):
            target.write(chunk)
            downloaded += len(chunk)
            if downloaded - reported >= 256 * 1024 * 1024:
                print(f'  {downloaded / 1e9:.2f} GB', flush=True)
                reported = downloaded
    if expected and sha256(partial) != expected:
        raise ValueError(f'Checksum mismatch: {partial}')
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise SystemExit('The local MLX environment requires Apple Silicon macOS.')
    for path in (ROOT, ROOT / '.runtime', ROOT / '.venv', ROOT / 'models', MODEL):
        if path.is_symlink():
            raise SystemExit(f'Symbolic link is not allowed: {path}')
    runtime = ROOT / '.runtime'
    runtime.mkdir(exist_ok=True)
    if not args.download_only:
        archive = runtime / 'uv-aarch64-apple-darwin.tar.gz'
        download(f'https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{archive.name}', archive, UV_SHA256)
        uv = runtime / 'uv-aarch64-apple-darwin' / 'uv'
        if not uv.exists():
            subprocess.run(['/usr/bin/tar', '-xzf', str(archive), '-C', str(runtime)], check=True)
        env = os.environ.copy()
        env.update(UV_NO_CONFIG='1', UV_CACHE_DIR=str(runtime / 'uv-cache'),
                   UV_PYTHON_INSTALL_DIR=str(runtime / 'python'),
                   UV_PYTHON_BIN_DIR=str(runtime / 'bin'))
        subprocess.run([str(uv), 'python', 'install', '3.11.18', '--no-bin'], env=env, check=True)
        python = ROOT / '.venv' / 'bin' / 'python'
        if not python.exists():
            subprocess.run([str(uv), 'venv', '--python', '3.11.18', str(ROOT / '.venv')], env=env, check=True)
        subprocess.run([str(uv), 'pip', 'install', '--python', str(python), '-r', str(ROOT / 'requirements-local.txt')], env=env, check=True)
    hashes = {}
    for filename in FILES:
        expected = {'model.safetensors': WEIGHTS_SHA256,
                    'tokenizer.json': TOKENIZER_SHA256}.get(filename)
        target = MODEL / filename
        download(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{filename}', target, expected)
        hashes[filename] = sha256(target)
    config = json.loads((MODEL / 'config.json').read_text())
    if config.get('model_type') != 'qwen3_5' or config.get('quantization', {}).get('bits') != 4:
        raise ValueError('Unexpected model architecture or quantization.')
    manifest = {'repo': REPO, 'revision': REVISION, 'sha256': hashes,
                'architecture': config['architectures'], 'quantization': config['quantization']}
    (MODEL / 'revision.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(f'Ready: {MODEL}', flush=True)


if __name__ == '__main__':
    main()
