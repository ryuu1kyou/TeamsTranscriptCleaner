# CSVパーサモジュール
import csv
import io
from typing import List, Dict

def parse_csv_text(csv_text: str) -> List[Dict[str, str]]:
    """
    CSV形式のテキストから誤字脱字リストを解析する
    
    Args:
        csv_text: CSV形式のテキスト
    
    Returns:
        誤字脱字リスト [{"誤": "誤字", "正": "正字"}, ...]
    """
    result = []
    
    try:
        # 文字列をCSVとして処理
        csv_file = io.StringIO(csv_text)
        reader = csv.reader(csv_file)
        
        # ヘッダー行をスキップ
        header = next(reader, None)
        
        # 各行を処理
        for row in reader:
            if len(row) >= 2:
                result.append({"誤": row[0], "正": row[1]})
    except Exception as e:
        print(f"CSVテキストパース中にエラーが発生しました: {e}")
        # エラーが発生した場合は空のリストを返す
        return []
    
    return result
