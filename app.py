import os
import streamlit as st
import requests
import re
import random
import json
import hashlib  # 画像ハッシュ用
import time    # 遅延用
from io import BytesIO
from PIL import Image
import asyncio
from concurrent.futures import ThreadPoolExecutor

# ▼ 画像解析用（ViTモデル）
import torch
from transformers import AutoFeatureExtractor, ViTForImageClassification
# ▲

from streamlit_chat import message  # streamlit-chat のメッセージ表示用関数

# ------------------------------------------------------------------
# st.set_page_config() は最初に呼び出す
# ------------------------------------------------------------------
st.set_page_config(page_title="ぼくのともだち", layout="wide")
st.title("ぼくのともだち V3.0 + 画像解析＆検索")

# ------------------------------------------------------------------
# config.toml の読み込み（テーマ設定）
# ------------------------------------------------------------------
try:
    try:
        import tomllib  # Python 3.11以降の場合
    except ImportError:
        import toml as tomllib
    with open("config.toml", "rb") as f:
        config_data = tomllib.load(f)
    theme_config = config_data.get("theme", {})
    primaryColor = theme_config.get("primaryColor", "#729075")
    backgroundColor = theme_config.get("backgroundColor", "#f1ece3")
    secondaryBackgroundColor = theme_config.get("secondaryBackgroundColor", "#fff8ef")
    textColor = theme_config.get("textColor", "#5e796a")
    font = theme_config.get("font", "monospace")
except Exception:
    primaryColor = "#729075"
    backgroundColor = "#f1ece3"
    secondaryBackgroundColor = "#fff8ef"
    textColor = "#5e796a"
    font = "monospace"

# ------------------------------------------------------------------
# 背景・共通スタイルの設定（テーマ設定を反映）
# ------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    body {{
        background-color: {backgroundColor};
        font-family: {font}, sans-serif;
        color: {textColor};
    }}
    .chat-container {{
        max-height: 600px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-bottom: 20px;
        background-color: {secondaryBackgroundColor};
    }}
    /* バブルチャット用のスタイル */
    .chat-bubble {{
        background-color: #d4f7dc;
        border-radius: 10px;
        padding: 8px;
        display: inline-block;
        max-width: 80%;
        word-wrap: break-word;
        white-space: pre-wrap;
        margin: 4px 0;
    }}
    .chat-header {{
        font-weight: bold;
        margin-bottom: 4px;
        color: {primaryColor};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------------
# ユーザーの名前入力＆AIの年齢入力（上部）
# ------------------------------------------------------------------
user_name = st.text_input("あなたの名前を入力してください", value="ユーザー", key="user_name")
ai_age = st.number_input("AIの年齢を指定してください", min_value=1, value=30, step=1, key="ai_age")

# ------------------------------------------------------------------
# サイドバー：カスタム新キャラクター設定、クイズ、画像アップロード、検索利用
# ------------------------------------------------------------------
st.sidebar.header("カスタム新キャラクター設定")
custom_new_char_name = st.sidebar.text_input("新キャラクターの名前（未入力ならランダム）", value="", key="custom_new_char_name")
custom_new_char_personality = st.sidebar.text_area("新キャラクターの性格・特徴（未入力ならランダム）", value="", key="custom_new_char_personality")

st.sidebar.header("ミニゲーム／クイズ")
if st.sidebar.button("クイズを開始する", key="quiz_start_button"):
    quiz_list = [
        {"question": "日本の首都は？", "answer": "東京"},
        {"question": "富士山の標高は何メートル？", "answer": "3776"},
        {"question": "寿司の主な具材は何？", "answer": "酢飯"},
        {"question": "桜の花言葉は？", "answer": "美しさ"}
    ]
    quiz = random.choice(quiz_list)
    st.session_state.quiz_active = True
    st.session_state.quiz_question = quiz["question"]
    st.session_state.quiz_answer = quiz["answer"]
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": "クイズ", "content": "クイズ: " + quiz["question"]})

st.sidebar.header("画像解析")
# ファイルアップローダーは1つのみ
uploaded_image = st.sidebar.file_uploader("画像をアップロードしてください", type=["png", "jpg", "jpeg"], key="file_uploader_key")

# インターネット検索利用のON/OFF（チェックボックスにユニークなキーを指定）
use_internet = st.sidebar.checkbox("インターネット検索を使用する", value=True, key="internet_search_checkbox_1")
st.sidebar.info("※スマホの場合は、画面左上のハンバーガーメニューからサイドバーにアクセスできます。")

# ------------------------------------------------------------------
# キャラクター定義（固定メンバー）
# ------------------------------------------------------------------
# ※キャラクター名は定数として再定義
USER_NAME = "user"
ASSISTANT_NAME = "assistant"
YUKARI_NAME = "ゆかり"
SHINYA_NAME = "しんや"
MINORU_NAME = "みのる"
NEW_CHAR_NAME = "新キャラクター"
NAMES = [YUKARI_NAME, SHINYA_NAME, MINORU_NAME]

# 新キャラクターはセッションに一度だけ生成（固定化）
if "new_char" not in st.session_state:
    def generate_new_character():
        """サイドバーで入力があればそれを使い、なければランダム"""
        if custom_new_char_name.strip() and custom_new_char_personality.strip():
            return custom_new_char_name.strip(), custom_new_char_personality.strip()
        candidates = [
            ("たけし", "冷静沈着で皮肉屋、どこか孤高な存在"),
            ("さとる", "率直かつ辛辣で、常に現実を鋭く指摘する"),
            ("りさ", "自由奔放で斬新なアイデアを持つ、ユニークな感性の持ち主"),
            ("けんじ", "クールで合理的、論理に基づいた意見を率直に述べる"),
            ("なおみ", "独創的で個性的、常識にとらわれず新たな視点を提供する")
        ]
        return random.choice(candidates)
    st.session_state.new_char = generate_new_character()
new_name, new_personality = st.session_state.new_char

# ------------------------------------------------------------------
# APIキー、モデル設定（Gemini API）
# ------------------------------------------------------------------
API_KEY = st.secrets["general"]["api_key"]
MODEL_NAME = "gemini-2.0-flash-001"

# ------------------------------------------------------------------
# セッション初期化：チャット履歴、画像解析キャッシュ、最後の画像ハッシュ、検索結果キャッシュ、APIステータス
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "analyzed_images" not in st.session_state:
    st.session_state.analyzed_images = {}
if "last_uploaded_hash" not in st.session_state:
    st.session_state.last_uploaded_hash = None
if "search_cache" not in st.session_state:
    st.session_state.search_cache = {}
if "gemini_status" not in st.session_state:
    st.session_state.gemini_status = ""
if "tavily_status" not in st.session_state:
    st.session_state.tavily_status = ""

# ------------------------------------------------------------------
# アイコン画像の読み込み（同じディレクトリの avatars フォルダを参照）
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
avatar_dir = os.path.join(BASE_DIR, "avatars")
try:
    img_user = Image.open(os.path.join(avatar_dir, "user.png"))
    img_yukari = Image.open(os.path.join(avatar_dir, "yukari.png"))
    img_shinya = Image.open(os.path.join(avatar_dir, "shinya.png"))
    img_minoru = Image.open(os.path.join(avatar_dir, "minoru.png"))
    img_newchar = Image.open(os.path.join(avatar_dir, "new_character.png"))
except Exception as e:
    st.error(f"画像読み込みエラー: {e}")
    img_user, img_yukari, img_shinya, img_minoru, img_newchar = "👤", "🌸", "🌊", "🍀", "⭐"

avatar_img_dict = {
    USER_NAME: img_user,
    YUKARI_NAME: img_yukari,
    SHINYA_NAME: img_shinya,
    MINORU_NAME: img_minoru,
    NEW_CHAR_NAME: img_newchar,
    ASSISTANT_NAME: "🤖",
    "クイズ": "❓",
    "画像解析": "🖼️",
}

# ------------------------------------------------------------------
# クラス定義：各エージェント（キャラクター）ごとに応答生成を行う
# ------------------------------------------------------------------
class ChatAgent:
    def __init__(self, name, style, detail):
        self.name = name
        self.style = style
        self.detail = detail

    def generate_response(self, question: str, ai_age: int, search_info: str = "") -> str:
        current_user = st.session_state.get("user_name", "ユーザー")
        prompt = f"【{current_user}さんの質問】\n{question}\n\n"
        if search_info:
            prompt += f"最新の情報によると、{search_info}という報告があります。\n"
        prompt += f"このAIは{ai_age}歳として振る舞います。\n"
        prompt += f"{self.name}は【{self.style}な視点】で、{self.detail}。\n"
        prompt += "あなたの回答のみを出力してください。"
        response = call_gemini_api(prompt)
        return response

# ------------------------------------------------------------------
# 並列実行用：エージェントごとの応答生成（並列化で高速化）
# ------------------------------------------------------------------
def generate_discussion_parallel(question: str, persona_params: dict, ai_age: int, search_info: str = "") -> str:
    # 各エージェントのインスタンス生成
    agents = []
    for name, params in persona_params.items():
        agents.append(ChatAgent(name, params["style"], params["detail"]))
    # 新キャラクター
    new_agent = ChatAgent(new_name, new_personality, "")
    agents.append(new_agent)
    # 並列に各エージェントの応答を生成
    responses = {}
    with ThreadPoolExecutor(max_workers=len(agents)) as executor:
        future_to_agent = {executor.submit(agent.generate_response, question, ai_age, search_info): agent for agent in agents}
        for future in future_to_agent:
            agent = future_to_agent[future]
            responses[agent.name] = future.result()
    # 各エージェントの応答を整形して結合
    conversation = "\n".join([f"{agent.name}: {responses[agent.name]}" for agent in agents])
    return conversation

def continue_discussion_parallel(additional_input: str, history: str, ai_age: int, search_info: str = "") -> str:
    # ここも同様に並列に各エージェントの応答を生成
    # まずは既存の会話履歴をまとめた上で、各エージェントが独自の応答を返す
    persona_params = adjust_parameters(additional_input, ai_age)
    agents = []
    for name, params in persona_params.items():
        agents.append(ChatAgent(name, params["style"], params["detail"]))
    new_agent = ChatAgent(new_name, new_personality, "")
    agents.append(new_agent)
    responses = {}
    with ThreadPoolExecutor(max_workers=len(agents)) as executor:
        future_to_agent = {executor.submit(agent.generate_response, additional_input, ai_age, search_info): agent for agent in agents}
        for future in future_to_agent:
            agent = future_to_agent[future]
            responses[agent.name] = future.result()
    conversation = "\n".join([f"{agent.name}: {responses[agent.name]}" for agent in agents])
    return conversation

# ------------------------------------------------------------------
# Gemini API 呼び出し関数はそのまま利用（上記クラス内で呼ばれる）
# → remove_json_artifacts, call_gemini_api は既存の実装

# ------------------------------------------------------------------
# ViTモデルを用いた画像解析モデルのロード（キャッシュ）はそのまま利用
@st.cache_resource
def load_image_classification_model():
    model_name = "google/vit-base-patch16-224"
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    model.eval()
    return extractor, model

extractor, vit_model = load_image_classification_model()

def analyze_image_with_vit(pil_image: Image.Image) -> str:
    """ViTで画像分類を行い、上位3クラスを文字列化（RGB変換済み）"""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    inputs = extractor(pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = vit_model(**inputs)
    logits = outputs.logits
    topk = logits.topk(3)
    top_indices = topk.indices[0].tolist()
    probs = torch.nn.functional.softmax(logits, dim=1)[0]
    labels = vit_model.config.id2label
    result_str = []
    for idx in top_indices:
        label_name = labels[idx]
        confidence = probs[idx].item()
        result_str.append(f"{label_name} ({confidence*100:.1f}%)")
    return ", ".join(result_str)

# ------------------------------------------------------------------
# インターネット検索実行（tavily API利用＋キャッシュ＆非同期処理）
# ------------------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor

@st.cache_data(show_spinner=False)
def cached_get_search_info(query: str) -> str:
    url = "https://api.tavily.com/search"
    token = st.secrets["tavily"]["token"]
    headers = {
         "Authorization": f"Bearer {token}",
         "Content-Type": "application/json"
    }
    payload = {
         "query": query,
         "topic": "general",
         "search_depth": "basic",
         "max_results": 1,
         "time_range": None,
         "days": 3,
         "include_answer": True,
         "include_raw_content": False,
         "include_images": False,
         "include_image_descriptions": False,
         "include_domains": [],
         "exclude_domains": []
    }
    try:
         response = requests.post(url, headers=headers, json=payload)
         if response.status_code == 200:
             st.session_state.tavily_status = "tavily API: OK"
         else:
             st.session_state.tavily_status = f"tavily API Error {response.status_code}: {response.text}"
         data = response.json()
         result = data.get("answer", "")
         return result
    except Exception as e:
         st.session_state.tavily_status = f"tavily API Exception: {str(e)}"
         return ""

executor = ThreadPoolExecutor(max_workers=1)

def async_get_search_info(query: str) -> str:
    with st.spinner("最新情報を検索中…"):
        future = executor.submit(cached_get_search_info, query)
        return future.result()

# ------------------------------------------------------------------
# 既存のチャットメッセージを表示（st.chat_input 形式）
# ------------------------------------------------------------------
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    display_name = user_name if role == "user" else role
    if role == "user":
        with st.chat_message(role, avatar=avatar_img_dict.get(USER_NAME)):
            st.markdown(
                f'<div style="text-align: right;"><div class="chat-bubble"><div class="chat-header">{display_name}</div>{content}</div></div>',
                unsafe_allow_html=True,
            )
    else:
        with st.chat_message(role, avatar=avatar_img_dict.get(role, "🤖")):
            st.markdown(
                f'<div style="text-align: left;"><div class="chat-bubble"><div class="chat-header">{display_name}</div>{content}</div></div>',
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------------
# ユーザー入力の取得（st.chat_input）
# ------------------------------------------------------------------
user_input = st.chat_input("何か質問や話したいことがありますか？")
if user_input:
    # インターネット検索利用（tavily API） ※チェックボックスにユニークキーを指定
    search_info = async_get_search_info(user_input) if st.sidebar.checkbox("インターネット検索を使用する", value=True, key="internet_search_checkbox_1") else ""
    
    if st.session_state.get("quiz_active", False):
        if user_input.strip().lower() == st.session_state.quiz_answer.strip().lower():
            quiz_result = "正解です！おめでとうございます！"
        else:
            quiz_result = f"残念、不正解です。正解は {st.session_state.quiz_answer} です。"
        st.session_state.messages.append({"role": "クイズ", "content": quiz_result})
        with st.chat_message("クイズ", avatar=avatar_img_dict["クイズ"]):
            st.markdown(
                f'<div style="text-align: left;"><div class="chat-bubble"><div class="chat-header">クイズ</div>{quiz_result}</div></div>',
                unsafe_allow_html=True,
            )
        st.session_state.quiz_active = False
    else:
        with st.chat_message("user", avatar=avatar_img_dict.get(USER_NAME)):
            st.markdown(
                f'<div style="text-align: right;"><div class="chat-bubble"><div class="chat-header">{user_name}</div>{user_input}</div></div>',
                unsafe_allow_html=True,
            )
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # AI応答生成：エージェントごとの並列処理を利用
        if len(st.session_state.messages) == 1:
            persona_params = adjust_parameters(user_input, ai_age)
            discussion = generate_discussion_parallel(user_input, persona_params, ai_age, search_info=search_info)
        else:
            history = "\n".join(
                f'{msg["role"]}: {msg["content"]}'
                for msg in st.session_state.messages
                if msg["role"] in NAMES or msg["role"] == NEW_CHAR_NAME
            )
            discussion = continue_discussion_parallel(user_input, history, ai_age, search_info=search_info)
        
        # 各行ごとに応答を解析してチャット履歴に追加＆表示
        for line in discussion.split("\n"):
            line = line.strip()
            if line:
                parts = line.split(":", 1)
                role = parts[0]
                content = parts[1].strip() if len(parts) > 1 else ""
                st.session_state.messages.append({"role": role, "content": content})
                display_name = user_name if role == "user" else role
                if role == "user":
                    with st.chat_message(role, avatar=avatar_img_dict.get(USER_NAME)):
                        st.markdown(
                            f'<div style="text-align: right;"><div class="chat-bubble"><div class="chat-header">{display_name}</div>{content}</div></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    with st.chat_message(role, avatar=avatar_img_dict.get(role, "🤖")):
                        st.markdown(
                            f'<div style="text-align: left;"><div class="chat-bubble"><div class="chat-header">{display_name}</div>{content}</div></div>',
                            unsafe_allow_html=True,
                        )
                time.sleep(random.uniform(3, 10))  # ランダムな遅延（3～10秒）

# ------------------------------------------------------------------
# 画像アップロードがあれば、かつ新しい画像の場合のみ解析し会話開始
# ------------------------------------------------------------------
if not st.session_state.get("quiz_active", False) and uploaded_image is not None:
    image_bytes = uploaded_image.getvalue()
    image_hash = hashlib.md5(image_bytes).hexdigest()
    if st.session_state.last_uploaded_hash != image_hash:
        st.session_state.last_uploaded_hash = image_hash
        if image_hash in st.session_state.analyzed_images:
            analysis_text = st.session_state.analyzed_images[image_hash]
        else:
            pil_img = Image.open(BytesIO(image_bytes))
            label_text = analyze_image_with_vit(pil_img)  # ViTで解析
            analysis_text = f"{label_text}"
            st.session_state.analyzed_images[image_hash] = analysis_text

        st.session_state.messages.append({"role": "画像解析", "content": analysis_text})
        with st.chat_message("画像解析", avatar=avatar_img_dict.get("画像解析", "🖼️")):
            st.markdown(
                f'<div style="text-align: left;"><div class="chat-bubble"><div class="chat-header">画像解析</div>{analysis_text}</div></div>',
                unsafe_allow_html=True,
            )

        persona_params = adjust_parameters("image analysis", ai_age)
        discussion_about_image = discuss_image_analysis(analysis_text, persona_params, ai_age)
        for line in discussion_about_image.split("\n"):
            line = line.strip()
            if line:
                parts = line.split(":", 1)
                role = parts[0]
                content = parts[1].strip() if len(parts) > 1 else ""
                st.session_state.messages.append({"role": role, "content": content})
                display_name = user_name if role == "user" else role
                if role == "user":
                    with st.chat_message(role, avatar=avatar_img_dict.get(USER_NAME)):
                        st.markdown(
                            f'<div style="text-align: right;"><div class="chat-bubble"><div class="chat-header">{display_name}</div>{content}</div></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    with st.chat_message(role, avatar=avatar_img_dict.get(role, "🤖")):
                        st.markdown(
                            f'<div style="text-align: left;"><div class="chat-bubble"><div class="chat-header">{display_name}</div>{content}</div></div>',
                            unsafe_allow_html=True,
                        )
                time.sleep(random.uniform(3, 10))  # ランダムな遅延（3～10秒）

# ------------------------------------------------------------------
# チャット履歴の表示
# ------------------------------------------------------------------
st.header("会話履歴")
if st.session_state.messages:
    for msg in reversed(st.session_state.messages):
        display_name = user_name if msg["role"] == "user" else msg["role"]
        if msg["role"] == "user":
            with st.chat_message("user", avatar=avatar_img_dict.get(USER_NAME)):
                st.markdown(
                    f'<div style="text-align: right;"><div class="chat-bubble"><div class="chat-header">{display_name}</div>{msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            with st.chat_message(msg["role"], avatar=avatar_img_dict.get(msg["role"], "🤖")):
                st.markdown(
                    f'<div style="text-align: left;"><div class="chat-bubble"><div class="chat-header">{display_name}</div>{msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )
else:
    st.markdown("<p style='color: gray;'>ここに会話が表示されます。</p>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# APIステータスの表示（サイドバー）
# ------------------------------------------------------------------
st.sidebar.header("APIステータス")
st.sidebar.write("【Gemini API】", st.session_state.gemini_status)
st.sidebar.write("【tavily API】", st.session_state.tavily_status)
