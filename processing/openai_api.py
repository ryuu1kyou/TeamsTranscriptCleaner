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
        raise ValueError("OpenAI APIキーが設定されていません。設定画面からAPIキーを設定してください。")
    
    return OpenAI(api_key=api_key)

def correct_text(prompt: str, input_text: str, correction_words: List[Dict[str, str]], model: str = "gpt-4o") -> Tuple[str, float]:
    """
    OpenAI APIを使用してテキストを訂正する
    
    Args:
        prompt: システムプロンプト
        input_text: 訂正対象のテキスト
        correction_words: 訂正単語リスト [{"誤": "誤字", "正": "正字"}, ...]
        model: 使用するモデル名
    
    Returns:
        (訂正されたテキスト, 使用コスト)
    """
    client = get_client()
    
    # 厳格なシステムプロンプトを必ず先頭に追加
    strict_system_prompt = (
        "あなたは誤字脱字訂正専用のAIです。絶対に要約、追加、削除、言い換え、書式変更、文体変更、内容の修正、段落の統合・分割、語順の変更、その他の編集は行わず、"
        "誤字脱字リストに基づく訂正のみを行ってください。誤字脱字リストに該当しない箇所は一切変更せず、元のテキストをそのまま残してください。"
        "訂正リストにない箇所は絶対に修正しないでください。人間による最終確認・修正が必須です。"
    )
    # 訂正単語リストをプロンプトに追加
    correction_instruction = "以下の誤字脱字リストを優先的に適用してください：\n"
    for word in correction_words:
        correction_instruction += f"「{word['誤']}」→「{word['正']}」\n"
    # ユーザープロンプトも参考情報として追加
    full_prompt = f"{strict_system_prompt}\n\n{correction_instruction}\n\nユーザー補足: {prompt}"
    
    # モデル名が有効かチェック
    if model not in MODEL_PRICING:
        model = "gpt-4o"  # デフォルトモデル
    
    # リクエスト前のトークン数を概算
    prompt_tokens = count_tokens(full_prompt)
    input_tokens = count_tokens(input_text)
    total_input_tokens = prompt_tokens + input_tokens
    
    # APIリクエスト
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": input_text}
        ],
        temperature=0,
        top_p=1,
        presence_penalty=0,
        frequency_penalty=0
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




