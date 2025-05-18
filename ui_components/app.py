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
from processing.csv_parser import parse_csv_text
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
    disable_correction = not st.session_state["original_text"] or not st.session_state["correction_words"]
    if st.button("訂正実行", disabled=disable_correction):
        if not api_key:
            st.error("OpenAI APIキーが設定されていません。先にAPIキーを設定してください。")
        elif st.session_state["original_text"]:
            # --- CSVバリデーション ---
            import csv
            import io
            csv_file = io.StringIO(csv_text)
            reader = list(csv.reader(csv_file))
            # ヘッダー検証
            if not reader or len(reader[0]) < 2 or reader[0][0] != "誤" or reader[0][1] != "正":
                st.error("CSVのヘッダーは「誤,正」で始まる必要があります。")
            else:
                # 各行検証
                csv_error = False
                for i, row in enumerate(reader[1:], start=2):
                    if len(row) < 2 or not row[0].strip() or not row[1].strip():
                        st.error(f"CSV {i}行目に不正なデータがあります（空欄や2列未満）。")
                        csv_error = True
                        break
                if not csv_error:
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

# 差分表示のチェックボックス
show_diff = st.checkbox("差分表示")

# 差分表示（show_diffがTrueのときのみ差分ビューアを表示）
if show_diff:
    # 差分比較前に正規化
    def normalize_text(text):
        # 改行コード統一、前後空白除去、全角スペース→半角スペース
        return text.replace('\r\n', '\n').replace('\r', '\n').replace('　', ' ').strip()
    norm_original = normalize_text(original_text)
    norm_corrected = normalize_text(new_corrected)

    # 1文字単位で差分を比較してHTML生成
    def diff_chars_html(a, b):
        import html
        sm = difflib.SequenceMatcher(None, a, b)
        html_out = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                html_out.append(html.escape(a[i1:i2]))
            elif tag == 'replace':
                html_out.append('<span style="background-color: #ffaaaa;">' + html.escape(a[i1:i2]) + '</span>')
                html_out.append('<span style="background-color: #aaffaa;">' + html.escape(b[j1:j2]) + '</span>')
            elif tag == 'delete':
                html_out.append('<span style="background-color: #ffaaaa;">' + html.escape(a[i1:i2]) + '</span>')
            elif tag == 'insert':
                html_out.append('<span style="background-color: #aaffaa;">' + html.escape(b[j1:j2]) + '</span>')
        return ''.join(html_out)

    # 行数を揃える
    orig_lines = norm_original.splitlines()
    corr_lines = norm_corrected.splitlines()
    max_lines = max(len(orig_lines), len(corr_lines))
    diff_html = []
    for i in range(max_lines):
        a = orig_lines[i] if i < len(orig_lines) else ""
        b = corr_lines[i] if i < len(corr_lines) else ""
        diff_html.append(diff_chars_html(a, b))
    st.markdown('<br>'.join(diff_html), unsafe_allow_html=True)

# 訂正後→訂正前コピー機能＋ダウンロードボタン（横並び）
# 訂正後→訂正前コピー機能＋ダウンロードボタン（縦並び）
disable_copy = not st.session_state["original_text"] or not st.session_state["correction_words"]
if st.button("訂正後テキストを訂正前にコピー", disabled=disable_copy):
    if st.session_state["corrected_text"]:
        # コピー実行（訂正後テキストは絶対に消さない）
        st.session_state["original_text"] = st.session_state["corrected_text"]
        st.caption("✓ コピーが完了しました。")
        st.rerun()
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
st.download_button(
    label="最終確定ダウンロード",
    data=st.session_state["corrected_text"] if st.session_state["corrected_text"] else "",
    file_name=f"final_transcript_{now}.txt",
    mime="text/plain",
    disabled=not bool(st.session_state["corrected_text"])
)

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
