"""Chat-completions transport."""
import json
import os
from urllib.request import Request, urlopen

from .core import digest


class ChatLLM:
    """Supply a model-specific token counter including chat-template overhead.

    Over-budget inputs raise; diagnose() trims retrieved passages beforehand.
    Pass provider settings in options;
    configure the evaluation judge with reasoning_effort="low"."""
    def __init__(self, endpoint, model, count_tokens, *, key_env="LLM_API_KEY",
                 options=None, output_limit=8192, revision="unspecified", timeout=300):
        self.endpoint, self.model, self.count_tokens = endpoint, model, count_tokens
        self.key_env, self.options = key_env, dict(options or {})
        if {"messages", "model", "stream", "max_tokens", "max_completion_tokens"} & self.options.keys():
            raise ValueError("Transport-owned fields must not be overridden in options")
        self.output_limit, self.timeout = output_limit, timeout
        self.fingerprint = digest([endpoint, model, revision, self.options, output_limit])
        self.usage = []

    @staticmethod
    def messages(prompt, payload=None):
        messages = [{"role": "system", "content": prompt}]
        if payload is not None:
            messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})
        return messages

    def input_tokens(self, prompt, payload=None):
        return self.count_tokens(self.messages(prompt, payload))

    def text(self, prompt, payload=None, input_limit=46000):
        messages = self.messages(prompt, payload)
        if self.count_tokens(messages) > input_limit:
            raise ValueError("Input token budget exceeded")
        body = dict(model=self.model, messages=messages, stream=False,
                    max_completion_tokens=self.output_limit, **self.options)
        request = Request(self.endpoint, json.dumps(body).encode(), headers={
            "Authorization": "Bearer " + os.environ[self.key_env],
            "Content-Type": "application/json"})
        with urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        self.usage.append(result.get("usage", {}))
        choice = result["choices"][0]
        content = choice["message"].get("content")
        if choice.get("finish_reason") == "length" or not content:
            raise ValueError("No complete visible answer; retain raw output externally for audit")
        return content

    def json(self, prompt, payload, input_limit=46000):
        # Require strict JSON.
        result = json.loads(self.text(prompt, payload, input_limit))
        if not isinstance(result, dict):
            raise ValueError("Expected JSON object")
        return result
