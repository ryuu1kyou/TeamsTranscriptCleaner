# トークン数管理モジュール
from typing import List, Dict, Tuple
import re

# モデル別の料金（1000トークンあたりのUSD）
MODEL_PRICING = {
    "gpt-4-1106-preview": 0.01,  # GPT-4 Turbo
    "gpt-4-0125-preview": 0.01,  # GPT-4 Turbo
    "gpt-4-1106-vision-preview": 0.01,  # GPT-4 Vision
    "gpt-4-turbo": 0.01,  # GPT-4 Turbo
    "gpt-4o": 0.01,  # GPT-4o
    "gpt-4o-mini": 0.005,  # GPT-4o Mini
    "gpt-4": 0.03,  # GPT-4
    "gpt-3.5-turbo": 0.0015  # GPT-3.5 Turbo (レガシー)
}

# モデル別の最大トークン数
MODEL_MAX_TOKENS = {
    "gpt-4-1106-preview": 128000,
    "gpt-4-0125-preview": 128000,
    "gpt-4-1106-vision-preview": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 4096
}

def count_tokens(text: str) -> int:
    """
    テキストのトークン数を概算する（簡易版）
    
    Args:
        text: 対象テキスト
    
    Returns:
        概算トークン数
    """
    # 日本語は文字数の約0.5倍、英語は単語数の約1.3倍がトークン数の目安
    # 簡易的な計算として、文字数÷4をトークン数とする
    return len(text) // 4

def split_text(text: str, max_tokens: int = 4000) -> List[str]:
    """
    テキストをトークン数制限に基づいて分割する
    
    Args:
        text: 分割対象のテキスト
        max_tokens: 1チャンクあたりの最大トークン数
    
    Returns:
        分割されたテキストのリスト
    """
    # 段落ごとに分割
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_token_count = 0
    
    for paragraph in paragraphs:
        paragraph_token_count = count_tokens(paragraph)
        
        # 1つの段落が最大トークン数を超える場合は、文単位で分割
        if paragraph_token_count > max_tokens:
            sentences = re.split(r'(?<=[。．！？])', paragraph)
            for sentence in sentences:
                if not sentence:  # 空の文はスキップ
                    continue
                    
                sentence_token_count = count_tokens(sentence)
                
                # 現在のチャンクにこの文を追加するとトークン数を超える場合
                if current_token_count + sentence_token_count > max_tokens:
                    # 現在のチャンクを確定
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                        current_chunk = []
                        current_token_count = 0
                
                # この文自体が最大トークン数を超える場合（まれなケース）
                if sentence_token_count > max_tokens:
                    # 文字数で強制的に分割
                    chars_per_token = 4  # 簡易的に4文字≒1トークンとする
                    max_chars = max_tokens * chars_per_token
                    for i in range(0, len(sentence), max_chars):
                        chunk = sentence[i:i+max_chars]
                        chunks.append(chunk)
                else:
                    current_chunk.append(sentence)
                    current_token_count += sentence_token_count
        else:
            # 現在のチャンクにこの段落を追加するとトークン数を超える場合
            if current_token_count + paragraph_token_count > max_tokens:
                # 現在のチャンクを確定
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_token_count = 0
            
            # 段落を追加
            current_chunk.append(paragraph)
            current_token_count += paragraph_token_count
    
    # 最後のチャンクを追加
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def estimate_cost(text: str, model: str) -> Dict[str, float]:
    """
    APIリクエストの概算コストを計算する
    
    Args:
        text: 対象テキスト
        model: 使用するモデル名
    
    Returns:
        コスト情報の辞書 {"tokens": トークン数, "cost_usd": 概算コスト(USD)}
    """
    tokens = count_tokens(text)
    
    # デフォルト料金
    price_per_1k = MODEL_PRICING.get(model, 0.01)
    
    # コスト計算
    cost_usd = (tokens / 1000) * price_per_1k
    
    return {
        "tokens": tokens,
        "cost_usd": cost_usd
    }

def get_max_tokens_for_model(model: str) -> int:
    """
    モデルの最大トークン数を取得する
    
    Args:
        model: モデル名
    
    Returns:
        最大トークン数
    """
    return MODEL_MAX_TOKENS.get(model, 4000)

