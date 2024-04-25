import streamlit as st
from openai import OpenAI

import openai_handler

st.title("Assistant API Interpreter")

client = OpenAI()

with st.form("form", clear_on_submit=False):
    user_question = st.text_area("文章を入力")
    file = [
        st.file_uploader("ファイルをアップロード", accept_multiple_files=False)
    ] or None
    submitted = st.form_submit_button("送信")

if submitted:
    st.session_state["thread"], st.session_state["run"] = openai_handler.submit_message(
        user_question, file
    )
    st.session_state["run"] = openai_handler.wait_on_run(
        st.session_state["run"], st.session_state["thread"]
    )

if st.session_state.get("thread"):
    openai_handler.pretty_print(openai_handler.get_response(st.session_state["thread"]))
