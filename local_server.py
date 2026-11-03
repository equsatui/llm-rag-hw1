import argparse
import json
import logging
import math
import os
from pathlib import Path
import resource
import time
from http.server import HTTPServer

ROOT = Path(__file__).absolute().parent
os.environ.update(HF_HOME=str(ROOT / '.runtime' / 'hf'), HF_HUB_OFFLINE='1',
                  HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1',
                  TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
os.environ.pop('HF_TOKEN', None)
os.environ.pop('HUGGING_FACE_HUB_TOKEN', None)

import mlx.core as mx
from mlx_lm.models.cache import LRUPromptCache
from mlx_lm.server import APIHandler, ModelProvider, ResponseGenerator, _run_http_server

ALIAS = 'Qwen/Qwen3.5-4B'
REVISION = '32f3e8ecf65426fc3306969496342d504bfa13f3'
ALLOWED = {'model', 'messages', 'max_tokens', 'max_completion_tokens', 'temperature',
           'top_p', 'top_k', 'min_p', 'presence_penalty', 'repetition_penalty',
           'presence_context_size', 'repetition_context_size', 'seed',
           'chat_template_kwargs', 'stream', 'n'}


class LocalHandler(APIHandler):
    def do_POST(self):
        self.started = time.perf_counter()
        if self.path != '/v1/chat/completions':
            self.send_error(404)
            return
        super().do_POST()

    def validate_model_parameters(self):
        unknown = set(self.body) - ALLOWED
        if unknown:
            raise ValueError(f'Unsupported parameters: {sorted(unknown)}')
        for header in ('X-Question-ID', 'X-Submission-ID'):
            if not self.headers.get(header, '').strip():
                raise ValueError(f'{header} is required')
        if self.stream or self.body.get('n', 1) != 1:
            raise ValueError('Only stream=false and n=1 are supported')
        if self.requested_model not in (ALIAS, 'mlx-community/Qwen3.5-4B-MLX-4bit', 'default_model'):
            raise ValueError(f'This endpoint only serves {ALIAS}')
        self.requested_model = 'default_model'
        self.seed = self.body.get('seed', 42)
        self.presence_penalty = self.body.get('presence_penalty', 1.5)
        self.repetition_penalty = self.body.get('repetition_penalty', 1.0)
        kwargs = self.body.get('chat_template_kwargs', {'enable_thinking': False})
        if not isinstance(kwargs, dict) or set(kwargs) - {'enable_thinking'}:
            raise ValueError('Only chat_template_kwargs.enable_thinking is supported')
        if type(kwargs.get('enable_thinking', False)) is not bool:
            raise ValueError('enable_thinking must be boolean')
        kwargs = {'enable_thinking': False, **kwargs}
        self.chat_template_kwargs = kwargs
        super().validate_model_parameters()
        for key in ('temperature', 'top_p', 'min_p', 'presence_penalty', 'repetition_penalty'):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f'{key} must be finite')
        if 'max_tokens' in self.body and 'max_completion_tokens' in self.body:
            raise ValueError('Use only one output token limit')
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 4096:
            raise ValueError('max_tokens must be between 1 and 4096')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32)')
        messages = self.body.get('messages')
        if not isinstance(messages, list) or not messages:
            raise ValueError('messages must be a nonempty list')
        for message in messages:
            if not isinstance(message, dict) or set(message) != {'role', 'content'}:
                raise ValueError('Each message must contain only role and content')
            if message['role'] not in ('system', 'user', 'assistant') or not isinstance(message['content'], str):
                raise ValueError('Only text system/user/assistant messages are supported')
        tokenizer = self.response_generator.model_provider.tokenizer
        if tokenizer is None:
            raise ValueError('The model is still loading; wait for /health')
        prompt = tokenizer.apply_chat_template(messages, tools=None, tokenize=True,
                                               add_generation_prompt=True, **kwargs)
        if len(prompt) + self.max_tokens > 8192:
            raise ValueError('Full prompt plus max_tokens exceeds 8192 tokens')
        self.prompt_tokens = len(prompt)

    def generate_response(self, *args, **kwargs):
        response = super().generate_response(*args, **kwargs)
        response['model'] = ALIAS
        usage = response['usage']
        if usage['prompt_tokens'] != self.prompt_tokens:
            raise ValueError('Prompt token count differs from preflight count')
        if usage.get('prompt_tokens_details', {}).get('cached_tokens', 0) != 0:
            raise ValueError('Prompt cache must be disabled')
        record = {'id': self.headers['X-Question-ID'],
                  'submission_id': self.headers['X-Submission-ID'],
                  'model': ALIAS, 'revision': REVISION, 'llm_calls': 1,
                  **usage, 'elapsed_ms': (time.perf_counter() - self.started) * 1000,
                  'peak_mlx_memory_bytes': mx.get_peak_memory(),
                  'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'finish_reason': response['choices'][0]['finish_reason']}
        with self.response_generator.cli_args.audit.open('a', encoding='utf-8') as target:
            target.write(json.dumps(record, ensure_ascii=False) + '\n')
        return response

    def handle_models_request(self):
        self._set_completion_headers(200)
        self.end_headers()
        self.wfile.write(json.dumps({'object': 'list', 'data': [{'id': ALIAS, 'object': 'model'}]}).encode())

    def handle_health_check(self):
        ready = (self.response_generator.is_healthy and
                 self.response_generator.model_provider.tokenizer is not None)
        self._set_completion_headers(200 if ready else 503)
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'ok' if ready else 'loading'}).encode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=ROOT / 'models' / 'Qwen3.5-4B-MLX-4bit')
    parser.add_argument('--port', type=int, choices=[8765], default=8765)
    args = parser.parse_args()
    for path in (ROOT, ROOT / 'models', args.model):
        if path.is_symlink():
            raise SystemExit(f'Symbolic link is not allowed: {path}')
    if args.model.absolute() != ROOT / 'models' / 'Qwen3.5-4B-MLX-4bit':
        raise SystemExit('Use the fixed model directory created by setup_local.py')
    manifest = json.loads((args.model / 'revision.json').read_text())
    if manifest['revision'] != REVISION:
        raise SystemExit('Unexpected model revision')
    logs = ROOT / 'logs'
    if logs.is_symlink():
        raise SystemExit('logs must not be a symbolic link')
    logs.mkdir(exist_ok=True)
    args.audit = logs / 'local-server-usage.jsonl'
    if args.audit.is_symlink():
        raise SystemExit('The audit log must not be a symbolic link')
    args.model = str(args.model.absolute())
    vars(args).update(adapter_path=None, draft_model=None, num_draft_tokens=0,
                      trust_remote_code=False, pipeline=False, chat_template='',
                      use_default_chat_template=False, chat_template_args={'enable_thinking': False},
                      allowed_origins=[], temp=0.7, top_p=0.8, top_k=20, min_p=0.0,
                      max_tokens=1536, decode_concurrency=1, prompt_concurrency=1,
                      prefill_step_size=1024, prompt_cache_size=0, prompt_cache_bytes=0,
                      kv_bits=None, kv_group_size=64, quantized_kv_start=0)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    pid_file = ROOT / '.runtime' / 'local-server.pid'
    if pid_file.is_symlink():
        raise SystemExit('The PID file must not be a symbolic link')
    pid_file.write_text(str(os.getpid()) + '\n')
    logging.info('Local server PID=%s', os.getpid())
    provider = ModelProvider(args)
    generator = ResponseGenerator(provider, LRUPromptCache(max_size=0, max_bytes=0))
    _run_http_server('127.0.0.1', args.port, generator,
                     server_class=HTTPServer, handler_class=LocalHandler)


if __name__ == '__main__':
    main()
