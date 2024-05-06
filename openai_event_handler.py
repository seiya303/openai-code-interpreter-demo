# -*- coding: utf-8 -*-
from typing import Any

from openai import AssistantEventHandler
from openai.types.beta.threads.text import Text
from typing_extensions import override


class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self, text: Text) -> None:
        print("\nassistant > ", end="", flush=True)

    @override
    def on_text_delta(self, delta: Any, snapshot: Any) -> None:
        print(delta.value, end="", flush=True)

    def on_tool_call_created(self, tool_call: Any) -> None:
        print(f"\nassistant > {tool_call.type}\n", flush=True)

    def on_tool_call_delta(self, delta: Any, snapshot: Any) -> None:
        if delta.type == "code_interpreter":
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print("\n\noutput >", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)
