# Streamlitアプリのエントリーポイント
import os
import sys
import sys
import os

# Add the current working directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import datetime
import difflib

from processing.openai_api import correct_text
from processing.csv_parser import parse_csv_text
from processing.token_manager import (
    count_tokens,
    split_text,
    estimate_cost,
    MODEL_PRICING,
    get_max_tokens_for_model,
)
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
if "corrected_text_edited" not in st.session_state:  # 初期化を追加
    st.session_state["corrected_text_edited"] = ""
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
if "processing_mode" not in st.session_state:
    st.session_state["processing_mode"] = "misspelling"  # デフォルトは誤字脱字修正
if "last_uploaded_content" not in st.session_state:
    st.session_state["last_uploaded_content"] = None


st.set_page_config(layout="wide", page_title="Teams Transcript Cleaner")

st.title("Teams Transcript Cleaner")


# 単語集履歴表示切替
def toggle_word_history():
    st.session_state["show_word_history"] = not st.session_state["show_word_history"]


# サイドバー（左3割）。
with st.sidebar:
    st.header("設定")

    # APIキー設定
    api_key = get_api_key()
    if not api_key:
        st.warning(
            "OpenAI APIキーが設定されていません。Windows環境変数に「OPENAI_API_KEY」を設定してください。"
        )

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

    # 処理モード選択
    processing_mode = st.radio(
        "処理モード選択",
        ("misspelling", "grammar", "summarize"),
        format_func=lambda x: {
            "misspelling": "誤字脱字修正",
            "grammar": "文法訂正",
            "summarize": "要約",
        }[x],
        key="processing_mode",
    )
    custom_prompt = st.text_area(
        "追加の指示（モードに応じて入力）",
        value="",
        placeholder="例: 誤字脱字モード「特に人名を重点的に確認」、要約モード「300字以内で結論を先に」",
    )

    # ファイルアップロード処理
    uploaded_txt = st.file_uploader("訂正前議事録（TXT）アップロード", type=["txt"])
    if uploaded_txt is not None:
        newly_uploaded_content = uploaded_txt.read().decode("utf-8")

        if newly_uploaded_content != st.session_state.get(
            "last_uploaded_content", None
        ):
            st.session_state["original_text"] = newly_uploaded_content
            st.session_state["last_uploaded_content"] = (
                newly_uploaded_content  # 最後にアップロードされた内容を記憶
            )
            st.session_state["corrected_text"] = ""
            st.session_state["corrected_text_edited"] = ""  # 編集後もクリア
            st.caption("✓ 新しい議事録を読み込みました。")

        # トークン数とコスト概算を表示
        tokens = count_tokens(st.session_state["original_text"])
        cost_info = estimate_cost(st.session_state["original_text"], model)
        st.info(
            f"テキスト長: {len(st.session_state['original_text'])}文字, 概算トークン数: {tokens}, 概算コスト: ${cost_info['cost_usd']:.4f} USD"
        )

    # 誤字脱字修正モードの場合のみCSV関連を表示
    if st.session_state["processing_mode"] == "misspelling":
        uploaded_csv = st.file_uploader("誤字脱字一覧 (CSV) アップロード", type=["csv"])
        if uploaded_csv is not None:
            # ファイルの内容を読み込んでからparse_csv_textに渡す
            csv_content = uploaded_csv.read().decode("utf-8")
            st.session_state["correction_words"] = parse_csv_text(csv_content)
            # CSVテキストを更新
            st.session_state["csv_text"] = csv_content

        # CSV編集エリア
        st.subheader("誤字脱字一覧編集")
        # モード切り替え時の再レンダリングでキーが重複しないように、モード特有のキーにする
        csv_text_input = st.text_area(
            "誤字脱字一覧（CSV形式表示・編集可）",
            value=st.session_state["csv_text"],
            height=150,
            key="csv_text_area_misspelling",
        )
        if csv_text_input != st.session_state["csv_text"]:  # 変更があった場合
            st.session_state["csv_text"] = csv_text_input
            st.session_state["correction_words"] = parse_csv_text(
                csv_text_input
            )  # 即時パースして反映
    # 他のモードではCSV関連UIは表示しない。st.session_state["csv_text"]等は保持される。

    # 訂正実行ボタンの処理
    disable_correction = not st.session_state["original_text"]
    if st.session_state["processing_mode"] == "misspelling":
        # 誤字脱字モードでは、CSVテキストが存在し、かつ有効なデータ行が1行以上あることを確認
        current_csv_text_for_button = st.session_state.get("csv_text", "").strip()
        if not current_csv_text_for_button:
            disable_correction = True
        else:
            parsed_words_for_button = parse_csv_text(current_csv_text_for_button)
            if not parsed_words_for_button:  # ヘッダーのみ、または不正なCSVの場合
                disable_correction = True
    # 文法訂正・要約モードでは original_text があれば基本的に有効

    if st.button("訂正実行", disabled=disable_correction):
        if not api_key:
            st.error(
                "OpenAI APIキーが設定されていません。先にAPIキーを設定してください。"
            )
        elif st.session_state["original_text"]:
            action_should_proceed = True
            correction_words_for_api = []

            if st.session_state["processing_mode"] == "misspelling":
                # --- CSVバリデーション (誤字脱字修正モードのみ) ---
                import csv
                import io

                current_csv_text = st.session_state.get("csv_text", "").strip()
                if not current_csv_text:
                    st.error(
                        "誤字脱字一覧CSVが空です。アップロードまたは編集してください。"
                    )
                    action_should_proceed = False
                else:
                    csv_file = io.StringIO(current_csv_text)
                    reader_list = list(csv.reader(csv_file))
                    if (
                        not reader_list
                        or len(reader_list[0]) < 2
                        or reader_list[0][0] != "誤"
                        or reader_list[0][1] != "正"
                    ):
                        st.error("CSVのヘッダーは「誤,正」で始まる必要があります。")
                        action_should_proceed = False
                    else:
                        # ヘッダーを除いた実データ行があるか確認
                        parsed_words = parse_csv_text(current_csv_text)
                        if (
                            not parsed_words
                        ):  # parse_csv_textが空リストを返すのはデータ行がない場合
                            st.error(
                                "誤字脱字一覧に有効なデータ行がありません。CSVを確認してください。"
                            )
                            action_should_proceed = False
                        else:  # データ行がある場合、個別の行チェック（parse_csv_text内でもある程度行われるが念のため）
                            for i, row_data in enumerate(
                                reader_list[1:], start=2
                            ):  # ヘッダーの次から
                                if (
                                    len(row_data) < 2
                                    or not row_data[0].strip()
                                    or not row_data[1].strip()
                                ):
                                    st.error(
                                        f"CSV {i}行目に不正なデータがあります（空欄や2列未満）。"
                                    )
                                    action_should_proceed = False
                                    break
                        if action_should_proceed:
                            correction_words_for_api = parsed_words

            elif st.session_state["processing_mode"] == "grammar":
                # 文法訂正モードでは、UIからのCSV編集はできないが、
                # 過去に誤字脱字モードで設定したCSVがあればそれを利用する
                if st.session_state.get("csv_text", "").strip():
                    # ここでは厳密なバリデーションエラーは出さず、内容があれば使う程度
                    parsed_grammar_csv = parse_csv_text(st.session_state["csv_text"])
                    if parsed_grammar_csv:
                        correction_words_for_api = parsed_grammar_csv
            # 要約モードでは correction_words_for_api は [] のまま

            if action_should_proceed:
                with st.spinner(
                    f"{dict(misspelling='訂正', grammar='校正', summarize='要約')[st.session_state['processing_mode']]}処理中..."
                ):
                    # API呼び出し部分は変更なし (ユーザー提供のコードのままとする)
                    original_text_for_api_call = st.session_state["original_text"]
                    tokens = count_tokens(original_text_for_api_call)
                    max_tokens = get_max_tokens_for_model(model)
                    max_input_tokens = (
                        max_tokens // 2
                    )  # 出力とオーバーヘッドのために半分を予約
                    total_cost = 0.0
                    if tokens > max_input_tokens:
                        st.warning(
                            f"テキストが長いです。{len(split_text(original_text_for_api_call, max_input_tokens))} つのチャンクに分割して処理します。"
                        )
                        chunks = split_text(
                            original_text_for_api_call, max_input_tokens
                        )
                        corrected_chunks = []
                        progress_bar = st.progress(0)
                        for i, chunk in enumerate(chunks):
                            corrected_chunk, chunk_cost = correct_text(
                                st.session_state["processing_mode"],
                                custom_prompt,
                                chunk,
                                correction_words_for_api,
                                model,
                            )
                            corrected_chunks.append(corrected_chunk)
                            total_cost += chunk_cost
                            progress_bar.progress((i + 1) / len(chunks))
                        corrected = "\n\n".join(corrected_chunks)
                    else:
                        corrected, total_cost = correct_text(
                            st.session_state["processing_mode"],
                            custom_prompt,
                            original_text_for_api_call,
                            correction_words_for_api,
                            model,
                        )
                    st.session_state["current_cost"] += total_cost
                    st.session_state["corrected_text"] = corrected
                    st.session_state["corrected_text_edited"] = (
                        corrected  # 手修正用エリアにも反映
                    )
                    # 訂正実行後は差分表示をリセット
                    st.session_state["show_diff_checkbox_state"] = False
                    st.success(
                        f"{dict(misspelling='訂正', grammar='校正', summarize='要約')[st.session_state['processing_mode']]}が完了しました（コスト: ${total_cost:.4f} USD）"
                    )
        else:
            st.error("訂正前議事録をアップロードしてください")


# --- 双方向編集・リアルタイムプレビュー対応 ---
# 編集用の一時変数、再定義
corrected_text = st.session_state.get("corrected_text", "")
corrected_text_edited = st.session_state.get("corrected_text_edited", corrected_text)

if "show_diff_checkbox_state" not in st.session_state:
    st.session_state["show_diff_checkbox_state"] = False

# UI: 訂正前・訂正後を左右均等に並べる
col1, col2 = st.columns(2)

with col1:
    st.subheader("訂正前のファイル内容（編集不可）")

    st.text_area(
        "原文（編集不可）",
        value=st.session_state["original_text"],  # 常に現在の原文を表示
        height=400,
        key="original_fixed",
        disabled=True,
    )
with col2:
    st.subheader("訂正後のファイル内容（手修正可）")
    new_corrected = st.text_area(
        "訂正後テキスト（手修正可・下に差分がリアルタイム表示されます）",
        value=corrected_text_edited,
        height=400,
        key="corrected_editable",
    )

# 訂正後を手修正した場合、セッションに反映
if new_corrected != st.session_state.get("corrected_text_edited", corrected_text):
    st.session_state["corrected_text_edited"] = new_corrected
    st.session_state["corrected_text"] = new_corrected

# 差分表示のチェックボックス
st.session_state["show_diff_checkbox_state"] = st.checkbox(
    "差分表示", value=st.session_state["show_diff_checkbox_state"]
)

# 差分表示（show_diffがTrueのときのみ差分ビューアを表示）
if st.session_state["show_diff_checkbox_state"]:
    # 差分比較前に正規化
    def normalize_text(text):
        # 改行コード統一、前後空白除去、全角スペース→半角スペース
        return text.replace("\r\n", "\n").replace("\r", "\n").replace("　", " ").strip()

    norm_original = normalize_text(original_text)
    norm_corrected = normalize_text(new_corrected)

    # 1文字単位で差分を比較してHTML生成
    def diff_chars_html(a, b):
        import html

        sm = difflib.SequenceMatcher(None, a, b)
        html_out = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                html_out.append(html.escape(a[i1:i2]))
            elif tag == "replace":
                html_out.append(
                    '<span style="background-color: #ffaaaa;">'
                    + html.escape(a[i1:i2])
                    + "</span>"
                )
                html_out.append(
                    '<span style="background-color: #aaffaa;">'
                    + html.escape(b[j1:j2])
                    + "</span>"
                )
            elif tag == "delete":
                html_out.append(
                    '<span style="background-color: #ffaaaa;">'
                    + html.escape(a[i1:i2])
                    + "</span>"
                )
            elif tag == "insert":
                html_out.append(
                    '<span style="background-color: #aaffaa;">'
                    + html.escape(b[j1:j2])
                    + "</span>"
                )
        return "".join(html_out)

    # 行数を揃える
    orig_lines = norm_original.splitlines()
    corr_lines = norm_corrected.splitlines()
    max_lines = max(len(orig_lines), len(corr_lines))
    diff_html = []
    for i in range(max_lines):
        a = orig_lines[i] if i < len(orig_lines) else ""
        b = corr_lines[i] if i < len(corr_lines) else ""
        diff_html.append(diff_chars_html(a, b))
    st.markdown("<br>".join(diff_html), unsafe_allow_html=True)

# 訂正後→訂正前コピー機能＋ダウンロードボタン（縦並び）
disable_copy_button = not st.session_state.get(
    "corrected_text_edited", ""
)  # 編集後テキストがなければ無効

if st.session_state["processing_mode"] == "misspelling":
    current_csv_text_for_copy = st.session_state.get("csv_text", "").strip()
    if not current_csv_text_for_copy:
        disable_copy_button = True
    else:
        parsed_words_for_copy = parse_csv_text(current_csv_text_for_copy)
        if not parsed_words_for_copy:
            disable_copy_button = True

# ボタンを横並びにするために列を使用
button_col1, button_col2 = st.columns(2)
with button_col1:
    if st.button(
        "訂正後テキストを訂正前にコピー",
        disabled=disable_copy_button,
        use_container_width=True,
    ):
        if st.session_state.get("corrected_text_edited"):  # 編集後テキストが存在すれば
            st.session_state["original_text"] = st.session_state[
                "corrected_text_edited"
            ]  # 編集後のテキストをコピー
            st.session_state["show_diff_checkbox_state"] = False
            st.rerun()
with button_col2:
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label="最終確定ダウンロード",
        data=(
            st.session_state["corrected_text_edited"]
            if st.session_state["corrected_text_edited"]
            else ""
        ),  # 編集後のテキストをダウンロード
        file_name=f"final_transcript_{now}.txt",
        mime="text/plain",
        disabled=not bool(st.session_state.get("corrected_text_edited", "")),
        use_container_width=True,
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
    unsafe_allow_html=True,
)
