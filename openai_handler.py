# -*- coding: utf-8 -*-
import time
from os.path import dirname, join
from typing import (
    Any,
    Iterable,
    Literal,
    Optional,
    TypedDict,
)

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
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

dotenv_path = join(dirname(__file__), ".env.local")
load_dotenv(dotenv_path)

client = OpenAI()
# IF: https://platform.openai.com/docs/assistants/how-it-works/creating-assistants
assistant = client.beta.assistants.create(
    name="汎用アシスタント",
    instructions="あなたは汎用的なアシスタントです。質問には簡潔かつ正確に答えてください。",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-turbo",
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
) -> tuple[Thread, Run]:
    print("assistant_id:", assistant_id)
    print("user_message:", user_message)
    print("files:", files)

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

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )
    print("run_id:", run.id)
    return thread, run


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
        print("m:", m)
        with st.chat_message(m.role):
            print("content:", m.content)
            for content in m.content:
                image_file_id = ""
                cont_dict = content.model_dump()
                if (_image_file := cont_dict.get("image_file")) is not None and isinstance(
                    _image_file, dict
                ):
                    print(_image_file)
                    if (_image_file_id := _image_file.get("file_id")) is not None and isinstance(
                        _image_file_id, str
                    ):
                        image_file_id = _image_file_id
                        st.image(get_file(image_file_id))

                if cont_dict.get("text") is not None and isinstance(content, TextContentBlock):
                    message_content = content.text
                    annotations = message_content.annotations
                    citations = []
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
                            cited_file = client.files.retrieve(file_path.file_id)
                            citation = (
                                f"[{index}] Click <here> to download {cited_file.filename}, "
                                f"file_id: {file_path.file_id}"
                            )
                            citations.append(citation)
                            files.append(
                                (
                                    file_path.file_id,
                                    annotation.text.split("/")[-1],
                                )
                            )
                    message_content.value += "\n" + "\n".join(citations)
                    st.write(message_content.value)
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


def get_file(file_id: str) -> bytes:
    retrieve_file = client.files.with_raw_response.content(file_id)
    content: bytes = retrieve_file.content
    return content
