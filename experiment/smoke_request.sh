#!/bin/bash
# (Git Bash) One short and one medium request against the tracing server.
curl -s http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"Write a Python function that checks if a string is a palindrome."}],"max_tokens":120,"chat_template_kwargs":{"enable_thinking":false}}' | head -c 400; echo
