# Streamlitアプリのエントリーポイント
import os
import sys
import sys
import os

# Add the current working directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import datetime
import difflib

from processing.openai_api import correct_text
from processing.csv_parser import parse_csv, parse_csv_text
from processing.text_viewer import show_diff, highlight_corrections
from processing.token_manager import count_tokens, split_text, estimate_cost, MODEL_PRICING, get_max_tokens_for_model
from processing.api_key_manager import get_api_key, get_total_cost, reset_cost

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
# processingディレクトリの存在を確認
processing_dir = os.path.join(parent_dir, "processing")


# セッション状態の初期化
if "original_text" not in st.session_state:
    st.session_state["original_text"] = ""

# セッション状態から original_text を取得
original_text = st.session_state.get("original_text", "")

# その他のセッション状態の初期化
if "corrected_text" not in st.session_state:
    st.session_state["corrected_text"] = ""
if "correction_words" not in st.session_state:
    st.session_state["correction_words"] = []
if "csv_text" not in st.session_state:
    st.session_state["csv_text"] = "誤,正"
if "run_correction" not in st.session_state:
    st.session_state["run_correction"] = False
if "download" not in st.session_state:
    st.session_state["download"] = False
if "display_mode" not in st.session_state:
    st.session_state["display_mode"] = "normal"  # normal, diff, highlight
if "show_api_settings" not in st.session_state:
    st.session_state["show_api_settings"] = False
if "current_cost" not in st.session_state:
    st.session_state["current_cost"] = 0.0
if "show_word_history" not in st.session_state:
    st.session_state["show_word_history"] = False
if "word_list_name" not in st.session_state:
    st.session_state["word_list_name"] = ""
if "new_wrong" not in st.session_state:
    st.session_state["new_wrong"] = ""
if "new_correct" not in st.session_state:
    st.session_state["new_correct"] = ""

st.set_page_config(layout="wide", page_title="Teams Transcript Cleaner")

st.title("Teams Transcript Cleaner")

# APIキー設定画面の表示切替
def toggle_api_settings():
    st.session_state["show_api_settings"] = not st.session_state["show_api_settings"]

# 単語集履歴表示切替
def toggle_word_history():
    st.session_state["show_word_history"] = not st.session_state["show_word_history"]

# サイドバー（左3割）。
with st.sidebar:
    st.header("設定")
    
    # APIキー設定
    api_key = get_api_key()
    if not api_key:
        st.warning("OpenAI APIキーが設定されていません。Windows環境変数に「OPENAI_API_KEY」を設定してください。")
    
    # コスト表示
    total_cost = get_total_cost()
    st.subheader("コスト情報")
    st.info(f"累計使用コスト: ${total_cost:.4f} USD")
    
    if st.button("コスト履歴をリセット"):
        reset_cost()
        st.success("コスト履歴をリセットしました。")
        st.experimental_rerun()
    
    # 現在のセッションのコスト
    st.info(f"現在のセッションコスト: ${st.session_state['current_cost']:.4f} USD")
    
    # 最新のモデルリストを使用
    model_options = ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
    model = st.selectbox("OpenAIモデル選択", model_options)
    
    # モデル料金表示
    price_per_1k = MODEL_PRICING.get(model, 0.01)
    st.caption(f"料金: ${price_per_1k} USD / 1000トークン")
    
    custom_prompt = st.text_area("カスタムプロンプト", value="漏字脱字訂正優先、日本語文法訂正は次、意思齟齬は禁止")

    # ファイルアップロード処理
    uploaded_txt = st.file_uploader("訂正前議事録（TXT）アップロード", type=["txt"])
    if uploaded_txt is not None:
        content = uploaded_txt.read().decode("utf-8")
        
        # 新しい議事録がアップロードされた場合、過去履歴を削除
        if content != st.session_state["original_text"]:
            st.session_state["corrected_text"] = ""
            
            # 小さなインジケーターを表示（控えめな通知）
            st.caption("✓ 新しい議事録を読み込みました。")
        
        st.session_state["original_text"] = content
        
        # トークン数とコスト概算を表示
        tokens = count_tokens(content)
        cost_info = estimate_cost(content, model)
        st.info(f"テキスト長: {len(content)}文字, 概算トークン数: {tokens}, 概算コスト: ${cost_info['cost_usd']:.4f} USD")
    
    uploaded_csv = st.file_uploader("誤字脱字一覧 (CSV) アップロード", type=["csv"])
    if uploaded_csv is not None:
        # ファイルの内容を読み込んでからparse_csv_textに渡す
        csv_content = uploaded_csv.read().decode('utf-8')
        st.session_state["correction_words"] = parse_csv_text(csv_content)
        # CSVテキストを更新
        st.session_state["csv_text"] = csv_content
    
    # CSV編集エリア
    st.subheader("誤字脱字一覧編集")
    csv_text = st.text_area("誤字脱字一覧（CSV形式表示・編集可）", value=st.session_state["csv_text"], height=150)
    

    # 訂正実行ボタンの処理
    if st.button("訂正実行"):
        if not api_key:
            st.error("OpenAI APIキーが設定されていません。先にAPIキーを設定してください。")
        elif st.session_state["original_text"]:
            with st.spinner("訂正処理中..."):
                # CSVテキストエリアから直接誤字脱字リストを読み込む
                correction_words = parse_csv_text(csv_text)
                
                original_text = st.session_state["original_text"]
                tokens = count_tokens(original_text)
                max_tokens = get_max_tokens_for_model(model)
                max_input_tokens = max_tokens // 2
                total_cost = 0.0
                if tokens > max_input_tokens:
                    st.warning(f"テキストが長いです。{len(split_text(original_text, max_input_tokens))} つのチャンクに分割して処理します。")
                    chunks = split_text(original_text, max_input_tokens)
                    corrected_chunks = []
                    progress_bar = st.progress(0)
                    for i, chunk in enumerate(chunks):
                        corrected_chunk, chunk_cost = correct_text(
                            custom_prompt, 
                            chunk,
                            correction_words, # ここで直接読み込んだリストを使用
                            model
                        )
                        corrected_chunks.append(corrected_chunk)
                        total_cost += chunk_cost
                        progress_bar.progress((i + 1) / len(chunks))
                    corrected = "\n\n".join(corrected_chunks)
                else:
                    corrected, total_cost = correct_text(
                        custom_prompt, 
                        original_text,
                        correction_words, # ここで直接読み込んだリストを使用
                        model
                    )
                st.session_state["current_cost"] += total_cost
                st.session_state["corrected_text"] = corrected
                st.session_state["corrected_text_edited"] = corrected
                st.success(f"訂正が完了しました（コスト: ${total_cost:.4f} USD）")
        else:
            st.error("訂正前議事録をアップロードしてください")

# ダウンロード機能
st.subheader("ダウンロード")
if st.session_state["corrected_text"]:
    # 現在の日時を取得してファイル名に使用
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # テキストファイルとしてダウンロード
    st.download_button(
        label="訂正結果をテキストファイルでダウンロード",
        data=st.session_state["corrected_text"],
        file_name=f"corrected_transcript_{now}.txt",
        mime="text/plain"
    )

# 差分表示のチェックボックス
show_diff = st.checkbox("差分表示")

# --- 双方向編集・リアルタイムプレビュー対応 ---
# 編集用の一時変数、再定義
corrected_text = st.session_state.get("corrected_text", "")
corrected_text_edited = st.session_state.get("corrected_text_edited", corrected_text)

# UI: 訂正前・訂正後を左右均等に並べる
col1, col2 = st.columns(2)
with col1:
    st.subheader("訂正前のファイル内容（編集不可）")
    st.text_area(
        "原文（編集不可）",
        value=st.session_state["original_text"],  # 元のテキストを表示
        height=400,
        key="original_fixed",
        disabled=True
    )
with col2:
    st.subheader("訂正後のファイル内容（手修正可）")
    new_corrected = st.text_area(
        "訂正後テキスト（手修正可・下に差分がリアルタイム表示されます）",
        value=corrected_text_edited,
        height=400,
        key="corrected_editable"
    )

# 訂正後を手修正した場合、セッションに反映
if new_corrected != st.session_state.get("corrected_text_edited", corrected_text):
    st.session_state["corrected_text_edited"] = new_corrected
    st.session_state["corrected_text"] = new_corrected

# 差分表示（show_diffがTrueのときのみ差分ビューアを表示）
if show_diff:
    # 差分を生成
    diff = difflib.ndiff(original_text.splitlines(), new_corrected.splitlines())
    diff_html = []
    for line in diff:
        if line.startswith('- '):
            diff_html.append(f'<span style="background-color: #ffaaaa;">{line[2:]}</span>')
        elif line.startswith('+ '):
            diff_html.append(f'<span style="background-color: #aaffaa;">{line[2:]}</span>')
        elif line.startswith('? '):
            for i, char in enumerate(line[2:]):
                if char != ' ':
                    diff_html[-1] = diff_html[-1][:i] + f'<span style="background-color: #ffff00;">{char}</span>' + diff_html[-1][i+1:]
        else:
            diff_html.append(line[2:])
    
    # HTML差分ビューアを表示
    st.markdown('<br>'.join(diff_html), unsafe_allow_html=True)

# 訂正後→訂正前コピー機能
if st.button("訂正後テキストを訂正前にコピー"):
    if st.session_state["corrected_text"]:
        # コピー実行
        st.session_state["original_text"] = st.session_state["corrected_text"]
        
        # 控えめな通知
        st.caption("✓ コピーが完了しました。")

# --- CSS追加 ---
st.markdown(
    """
    <style>
    .highlight-group {
        border: 2.5px solid #ff3333 !important;
        border-radius: 8px;
        margin: 12px 0;
        padding: 8px 12px;
        background: #fff8f8;
        box-shadow: 0 2px 8px #ffcccc44;
    }
    .stButton > button {
        margin-right: 8px;
        min-width: 100px;
    }
    .button-row {
        display: flex;
        gap: 12px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
