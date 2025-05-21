# システムアーキテクチャ

- GUIアプリケーションを提供する。
- クリーニング処理は、Pythonスクリプトとして実装する。
- GUIは、`ui_components/app.py` に実装する。

## 主要な技術的決定

- Pythonを主要なプログラミング言語として使用する。
- GUIフレームワークとして、streamlitを使用する。

## 構造設計

- モジュール分割：
  1. `ui_components/`：画面入出力層
     - `ui_components/app.py`：Streamlitアプリケーションのエントリーポイント。UIの構築、ユーザーからの入力処理、結果の表示を行う。
        - ファイルアップロードハンドラ：議事録テキストファイルと誤字脱字一覧CSVファイルのアップロードを処理する。
        - モデル選択 UI：OpenAIモデルを選択するためのUIを提供する。
        - プロンプトカスタマイザ：OpenAI APIへの指示をカスタマイズするためのUIを提供する。
        - 結果表示リーダー：訂正前後のテキストを表示し、差分表示機能を提供する。
  2. `processing/`：業務処理層
     - `processing/csv_parser.py`：CSVファイルを読み込み、誤字脱字リストを解析する。
        - `parse_csv_text()`：CSV形式のテキストから誤字脱字リストを解析し、辞書のリストとして返す。
     - `processing/openai_api.py`：OpenAI APIとの通信を処理し、テキストの訂正を行う。
        - `correct_text()`：OpenAI APIを使用してテキストを訂正し、訂正後のテキストと使用コストを返す。
     - `processing/token_manager.py`：OpenAI APIのトークン数を管理する。
        - `count_tokens()`：テキストのトークン数を概算する（簡易的な文字数ベースの推定）。
        - `split_text()`：テキストをトークン数制限に基づいて分割する（段落や文単位での分割を試みる）。
        - `estimate_cost()`：APIリクエストの概算コストを計算する。
        - `get_max_tokens_for_model()`：モデルの最大トークン数を取得する。
     - `processing/api_key_manager.py`：OpenAI APIキーを管理し、APIの使用コストを追跡する。
        - `get_api_key()`：環境変数からAPIキーを取得する。
        - `load_config()`：設定ファイルを読み込む。
        - `update_cost()`：使用コストを更新する。
        - `get_total_cost()`：累計コストを取得する。
        - `reset_cost()`：コスト履歴をリセットする。
        - `save_word_list(name, words)`：指定された名前で単語リストを保存します（最大10件まで保持し、古いものから置き換えられる可能性があります）。
        - `get_word_lists()`：保存されている単語リストの一覧を取得します。
        - `delete_word_list(name)`：指定された名前の単語リストを履歴から削除します。
        - これらの関数により、ユーザーは頻繁に使用する修正用CSVデータのコレクションを管理できますが、現在の `app.py` のUIでは、これらの保存/読み込み/削除機能はまだ直接提供されていないようです。
