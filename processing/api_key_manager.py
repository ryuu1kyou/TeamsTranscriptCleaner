# APIキー管理モジュール
import os
import json
from pathlib import Path
from typing import Dict, List, Any
import datetime

# APIキー保存用のディレクトリ（コスト記録用）
CONFIG_DIR = Path.home() / ".teams_transcript_cleaner"
CONFIG_FILE = CONFIG_DIR / "config.json"
WORD_HISTORY_FILE = CONFIG_DIR / "word_history.json"


def initialize_config_dir():
    """設定ディレクトリを初期化する"""
    if not CONFIG_DIR.exists():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if not CONFIG_FILE.exists():
        with open(CONFIG_FILE, "w") as f:
            json.dump({"total_cost": 0.0}, f)

    if not WORD_HISTORY_FILE.exists():
        with open(WORD_HISTORY_FILE, "w") as f:
            json.dump({"word_lists": []}, f)


def get_api_key() -> str:
    """環境変数からAPIキーを取得する"""
    # 環境変数から取得
    api_key = os.getenv("OPENAI_API_KEY", "")
    return api_key


def load_config() -> Dict:
    """設定ファイルを読み込む"""
    initialize_config_dir()

    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # ファイルが存在しないか、JSONとして不正な場合は新しい設定を返す
        return {"total_cost": 0.0}


def update_cost(cost: float):
    """使用コストを更新する"""
    config = load_config()
    config["total_cost"] = config.get("total_cost", 0.0) + cost

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)

    return config["total_cost"]


def get_total_cost() -> float:
    """累計コストを取得する"""
    config = load_config()
    return config.get("total_cost", 0.0)


def reset_cost():
    """コスト履歴をリセットする"""
    config = load_config()
    config["total_cost"] = 0.0

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)


# 単語集履歴管理機能
def save_word_list(name: str, words: List[Dict[str, str]]):
    """単語集を履歴に保存する"""
    initialize_config_dir()

    try:
        with open(WORD_HISTORY_FILE, "r") as f:
            history = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        history = {"word_lists": []}

    # 同名の単語集があれば上書き、なければ追加
    found = False
    for i, word_list in enumerate(history["word_lists"]):
        if word_list["name"] == name:
            history["word_lists"][i] = {
                "name": name,
                "words": words,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            found = True
            break

    if not found:
        history["word_lists"].append(
            {
                "name": name,
                "words": words,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    # 最大10件まで保存
    if len(history["word_lists"]) > 10:
        history["word_lists"] = history["word_lists"][-10:]

    with open(WORD_HISTORY_FILE, "w") as f:
        json.dump(history, f)


def get_word_lists() -> List[Dict[str, Any]]:
    """保存された単語集リストを取得する"""
    initialize_config_dir()

    try:
        with open(WORD_HISTORY_FILE, "r") as f:
            history = json.load(f)
        return history.get("word_lists", [])
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def delete_word_list(name: str):
    """単語集を削除する"""
    initialize_config_dir()

    try:
        with open(WORD_HISTORY_FILE, "r") as f:
            history = json.load(f)

        history["word_lists"] = [
            wl for wl in history["word_lists"] if wl["name"] != name
        ]

        with open(WORD_HISTORY_FILE, "w") as f:
            json.dump(history, f)

        return True
    except (json.JSONDecodeError, FileNotFoundError):
        return False
