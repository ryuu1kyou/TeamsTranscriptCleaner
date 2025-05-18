# OpenAI API通信インタフェース
import os
from openai import OpenAI
from typing import List, Dict, Tuple
from processing.token_manager import MODEL_PRICING, count_tokens
from processing.api_key_manager import get_api_key, update_cost


# APIクライアントの初期化
def get_client():
    """APIクライアントを取得する"""
    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "OpenAI APIキーが設定されていません。Windows環境変数「OPENAI_API_KEY」を設定してください。"
        )

    return OpenAI(api_key=api_key)


def correct_text(
    processing_mode: str,
    user_custom_prompt: str,
    input_text: str,
    correction_words: List[Dict[str, str]],
    model: str = "gpt-4o",
) -> Tuple[str, float]:
    """
    OpenAI APIを使用してテキストを訂正する

    Args:
        processing_mode: 処理モード ("misspelling", "grammar", "summarize")
        user_custom_prompt: ユーザーが入力した追加の指示
        input_text: 訂正対象のテキスト
        correction_words: 訂正単語リスト [{"誤": "誤字", "正": "正字"}, ...]
        model: 使用するモデル名

    Returns:
        (処理されたテキスト, 使用コスト)
    """
    client = get_client()
    messages = []
    system_content = ""

    if processing_mode == "misspelling":
        strict_system_prompt = (
            "あなたは誤字脱字訂正専用のAIです。絶対に要約、追加、削除、言い換え、書式変更、文体変更、内容の修正、段落の統合・分割、語順の変更、その他の編集は行わず、"
            "誤字脱字リストに基づく訂正のみを行ってください。誤字脱字リストに該当しない箇所は一切変更せず、元のテキストをそのまま残してください。"
            "訂正リストにない箇所は絶対に修正しないでください。人間による最終確認・修正が必須です。"
        )
        correction_instruction = ""
        if correction_words:
            correction_instruction = (
                "以下の誤字脱字リストを優先的に適用してください：\n"
            )
            for word in correction_words:
                correction_instruction += f"「{word['誤']}」→「{word['正']}」\n"

        system_content = f"{strict_system_prompt}\n\n{correction_instruction}".strip()
        if user_custom_prompt:
            system_content += f"\n\nユーザー補足: {user_custom_prompt}"

    elif processing_mode == "grammar":
        grammar_system_prompt = (
            "あなたは日本語の文章を自然で文法的に正しく校正するAIです。\n"
            "以下の指示に従って、提供されたテキストを修正してください。\n"
            "1. 文法的な誤りを修正します。\n"
            "2. 不自然な表現をより自然な日本語に修正します。\n"
            "3. 誤字脱字があれば修正します。"
        )
        if correction_words:
            grammar_system_prompt += (
                "特に、以下のリストにある単語は優先的に修正してください：\n"
            )
            for word in correction_words:
                grammar_system_prompt += f"「{word['誤']}」→「{word['正']}」\n"

        grammar_system_prompt += (
            "\n4. 元の文章の意味や主要な情報を保持し、勝手に内容を追加したり削除したりしないでください。\n"
            "5. 文体は元のテキストに合わせてください。"
        )
        system_content = grammar_system_prompt.strip()
        if user_custom_prompt:
            system_content += f"\n\nユーザー補足: {user_custom_prompt}"

    elif processing_mode == "summarize":
        summarize_system_prompt = (
            "あなたは提供された日本語のテキストを要約するAIです。\n"
            "以下の指示に従って、テキストの要点をまとめてください。\n"
            "1. テキスト全体の主要なトピックと結論を把握します。\n"
            "2. 重要な情報を抽出し、冗長な部分や詳細は省略します。\n"
            "3. 元のテキストの意図を正確に反映した要約を作成します。"
        )
        system_content = summarize_system_prompt.strip()
        if user_custom_prompt:
            system_content += f"\n\nユーザーからの具体的な指示: {user_custom_prompt}"
        # 要約モードでは誤字脱字リストは使用しない

    else:  # デフォルトフォールバック (誤字脱字修正)
        # このケースは通常発生しないはずだが、念のため
        system_content = "あなたはテキスト処理AIです。ユーザーの指示に従ってください。"
        if user_custom_prompt:
            system_content += f"\n\nユーザー指示: {user_custom_prompt}"

    messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": input_text})

    # モデル名が有効かチェック
    if model not in MODEL_PRICING:
        model = "gpt-4o"  # デフォルトモデル

    # リクエスト前のトークン数を概算
    # prompt_tokens_estimate = count_tokens(system_content) # API応答のusage.prompt_tokens を使うので不要
    # input_tokens_estimate = count_tokens(input_text) # API応答のusage.prompt_tokens を使うので不要

    # APIリクエスト
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        top_p=1,
        presence_penalty=0,
        frequency_penalty=0,
    )

    corrected_text = response.choices[0].message.content

    # 実際の使用トークン数を取得
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens

    # コスト計算
    price_per_1k = MODEL_PRICING.get(model, 0.01)
    cost = (total_tokens / 1000) * price_per_1k

    # コスト履歴を更新
    update_cost(cost)

    return corrected_text, cost
