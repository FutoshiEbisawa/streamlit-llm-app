import streamlit as st
from dotenv import load_dotenv
import os

# LangChain 最新構成の import
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# --- 環境変数読み込み (.env 内の APIキーを使用) ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- LLM（OpenAI）初期化 ---
llm = ChatOpenAI(
    model="gpt-4o-mini",      # 提出課題では gpt-4o-mini でも OK
    temperature=0.7,
    api_key=OPENAI_API_KEY
)

# --- LLMに質問するための関数 ---
def ask_llm(user_text: str, mode: str) -> str:
    """
    引数:
        user_text: 入力フォームのテキスト
        mode: ラジオボタンで選択した専門家モード
    戻り値:
        LLM の回答（文字列）
    """

    # 専門家ごとのシステムメッセージ
    if mode == "ダイエットの専門家":
        system_prompt = (
            "あなたは世界最高のダイエット専門家です。"
            "専門的かつ優しく実践的なアドバイスを返してください。"
        )
    else:
        system_prompt = (
            "あなたはプロのキャリアコーチです。"
            "相談者が前向きに行動できるアドバイスを返してください。"
        )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text)
    ]

    # ★ ここがエラー原因だったので invoke() に変更
    response = llm.invoke(messages)

    return response.content


# --- Streamlit UI ---
st.title("🧠 LLMアプリ（提出課題）")
st.write("このアプリは、専門家モードを使い分けてLLMから回答を得るデモアプリです。")
st.write("以下の手順で使ってください：")
st.write("1. 専門家のタイプを選択する")
st.write("2. 質問内容を入力する")
st.write("3. 『送信』ボタンを押す")

# --- 専門家モード選択 ---
mode = st.radio(
    "専門家モードを選択してください：",
    ["ダイエットの専門家", "キャリアコーチ"]
)

# --- 入力フォーム ---
user_input = st.text_area("質問したい内容を入力してください")

# --- 実行ボタン ---
if st.button("送信"):
    if user_input.strip() == "":
        st.error("テキストを入力してください。")
    else:
        st.write("### 回答：")
        answer = ask_llm(user_input, mode)
        st.write(answer)
