"""OpenAI Python SDK integration by monkey-patching ``client.chat.completions.create``.

TODO: patch create to wrap calls in spans and record token usage where available.
"""

# TODO: define patch_openai_client / unpatch helpers
