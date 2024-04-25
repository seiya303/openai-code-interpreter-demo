from typing import Optional
from openai import OpenAI
from openai.pagination import SyncCursorPage
from openai.types.beta.threads import Message, Run
from openai.types.beta.thread import Thread
import time
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from os.path import join, dirname

from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), ".env.local")
load_dotenv(dotenv_path)

client = OpenAI()
# IF: https://platform.openai.com/docs/assistants/how-it-works/creating-assistants
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Answer questions briefly, in a sentence or less.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-turbo",
)
ASSISTANT_ID = assistant.id

global_messages = []


def submit_message(
    user_message: str,
    files: list[Optional[UploadedFile]] = [None],
    assistant_id: str = ASSISTANT_ID,
) -> tuple[Thread, Run]:
    print("assistant_id:", assistant_id)
    print("user_message:", user_message)
    print("files:", files)

    file_ids = submit_file(files) if files[0] is not None else []

    messages = [
        {
            "role": "user",
            "content": user_message,
        }
    ]
    if len(file_ids) > 0:
        messages[0]["attachments"] = (
            {"file_id": file_ids[0], "tools": [{"type": "code_interpreter"}]},
        )

    # IFは修正される可能性があるため、下のURLを確認する
    # https://platform.openai.com/docs/assistants/how-it-works/managing-threads-and-messages
    thread = client.beta.threads.create(messages=messages)
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
                file = client.files.create(file=file.read(), purpose="assistants")
                ids.append(file.id)
        return ids
    else:
        return []


def get_response(thread) -> SyncCursorPage[Message]:
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")


def pretty_print(messages: SyncCursorPage[Message]) -> None:
    for m in messages:
        print("m:", m)
        with st.chat_message(m.role):
            print("content:", m.content)
            for content in m.content:

                image_file_id = ""
                cont_dict = content.model_dump()
                if cont_dict.get("image_file") is not None:
                    print(cont_dict.get("image_file"))
                    image_file_id = cont_dict.get("image_file").get("file_id")
                    st.image(get_file(image_file_id))

                if cont_dict.get("text") is not None:
                    message_content = content.text
                    annotations = message_content.annotations
                    citations = []
                    files = []
                    for index, annotation in enumerate(annotations):
                        message_content.value = message_content.value.replace(
                            annotation.text, f" [{index}]"
                        )
                        if file_path := getattr(annotation, "file_path", None):
                            cited_file = client.files.retrieve(file_path.file_id)
                            citations.append(
                                f"[{index}] Click <here> to download {cited_file.filename}, file_id: {file_path.file_id}"
                            )
                            files.append(
                                (file_path.file_id, annotation.text.split("/")[-1])
                            )
                    message_content.value += "\n" + "\n".join(citations)
                    st.write(message_content.value)
                    for file in files:
                        st.download_button(
                            f"{file[1]} : ダウンロード",
                            get_file(file[0]),
                            file_name=file[1],
                        )


def wait_on_run(run: Run, thread) -> Run:
    while run.status == "queued" or run.status == "in_progress":
        print("wait_on_run", run.id, thread.id)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        print("run.status:", run.status)
        time.sleep(0.5)
    return run


def get_file(file_id) -> bytes:
    retrieve_file = client.files.with_raw_response.content(file_id)
    content = retrieve_file.content
    return content
