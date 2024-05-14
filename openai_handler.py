# -*- coding: utf-8 -*-
import time
from os.path import dirname, join
from typing import (
    Any,
    Iterable,
    Literal,
    Optional,
    Tuple,
    TypedDict,
)

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from openai.lib.streaming._assistants import AssistantStreamManager
from openai.pagination import SyncCursorPage
from openai.types.beta.thread import Thread
from openai.types.beta.thread_create_params import (
    Message as CreateMessage,
)
from openai.types.beta.thread_create_params import (
    MessageAttachment,
)
from openai.types.beta.threads import (
    Message,
    Run,
    TextContentBlock,
)
from streamlit.runtime.uploaded_file_manager import (
    UploadedFile,
)

from openai_event_handler import EventHandler

dotenv_path = join(dirname(__file__), ".env.local")
load_dotenv(dotenv_path)

client = OpenAI()
# IF: https://platform.openai.com/docs/assistants/how-it-works/creating-assistants
assistant = client.beta.assistants.create(
    name="汎用アシスタント",
    instructions="あなたは汎用的なアシスタントです。質問には簡潔かつ正確に答えてください。",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4o",
)
ASSISTANT_ID = assistant.id

global_messages: list[Any] = []


class CustomMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str
    attachments: Optional[Iterable[MessageAttachment]]
    metadata: Optional[Any]


# If no file is uploaded, the 'files' variable is assigned a list containing a single 'None' value.
def submit_message(
    user_message: str,
    files: Optional[list[Optional[UploadedFile]]] = None,
    assistant_id: str = ASSISTANT_ID,
) -> Tuple[Thread, AssistantStreamManager[EventHandler]]:
    print("assistant_id:", assistant_id)
    print("user_message:", user_message)
    print("files:", files)

    with st.chat_message("user"):
        st.write(user_message)

    if files is None:
        files = [None]

    file_ids = submit_file(files) if files[0] is not None else []

    messages: list[CustomMessage] = [
        {"role": "user", "content": user_message, "attachments": None, "metadata": None}
    ]
    if len(file_ids) > 0:
        messages[0]["attachments"] = [
            {
                "file_id": file_ids[0],
                "tools": [{"type": "code_interpreter"}],
            }
        ]

    # IFは修正される可能性があるため、下のURLを確認する
    # https://platform.openai.com/docs/assistants/how-it-works/managing-threads-and-messages
    _messages: Iterable[CreateMessage] = [CreateMessage(**msg) for msg in messages]
    thread = client.beta.threads.create(messages=_messages)
    print("thread_id:", thread.id)

    stream = client.beta.threads.runs.stream(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="ユーザーのメッセージと同じ言語で回答してください。回答を生成する際はユーザーへの確認は不要です。",
        event_handler=EventHandler(),
    )
    return thread, stream


def submit_file(files: list[Optional[UploadedFile]]) -> list[str]:
    if files:
        ids = []
        for file in files:
            if file is not None:
                # IF: https://platform.openai.com/docs/assistants/how-it-works/creating-assistants
                _file = client.files.create(
                    file=file.read(),
                    purpose="assistants",
                )
                ids.append(_file.id)
        return ids
    else:
        return []


def get_response(thread: Thread) -> SyncCursorPage[Message]:
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")


def pretty_print(messages: SyncCursorPage[Message]) -> None:
    for m in messages:
        print("role:", m.role)
        print("content:", m.content)
        if m.role == "assistant":
            for content in m.content:
                image_file_id = ""
                cont_dict = content.model_dump()
                if (_image_file := cont_dict.get("image_file")) is not None and isinstance(
                    _image_file, dict
                ):
                    print("image_file:", _image_file)
                    if (_image_file_id := _image_file.get("file_id")) is not None and isinstance(
                        _image_file_id, str
                    ):
                        image_file_id = _image_file_id
                        st.image(get_file(image_file_id))

                if cont_dict.get("text") is not None and isinstance(content, TextContentBlock):
                    message_content = content.text
                    annotations = message_content.annotations
                    files = []
                    for (
                        index,
                        annotation,
                    ) in enumerate(annotations):
                        message_content.value = message_content.value.replace(
                            annotation.text,
                            f" [{index}]",
                        )
                        if file_path := getattr(
                            annotation,
                            "file_path",
                            None,
                        ):
                            files.append(
                                (
                                    file_path.file_id,
                                    annotation.text.split("/")[-1],
                                )
                            )
                    for file in files:
                        st.download_button(
                            f"{file[1]} : ダウンロード",
                            get_file(file[0]),
                            file_name=file[1],
                        )


def wait_on_run(run: Run, thread: Thread) -> Run:
    while run.status == "queued" or run.status == "in_progress":
        print("wait_on_run", run.id, thread.id)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        print("run.status:", run.status)
        time.sleep(0.5)
    return run


def wait_on_stream(stream: AssistantStreamManager[EventHandler], thread: Thread) -> None:
    with st.chat_message("assistant"):
        with stream as s:
            st.write_stream(s.text_deltas)
            s.until_done()
        if "thread" in st.session_state:
            thread = st.session_state["thread"]
            pretty_print(get_response(thread))


def get_file(file_id: str) -> bytes:
    retrieve_file = client.files.with_raw_response.content(file_id)
    content: bytes = retrieve_file.content
    return content
