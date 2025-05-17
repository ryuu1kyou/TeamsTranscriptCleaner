# 訂正済テキストビューア
from typing import List
import difflib
import re

def show_diff(original: str, corrected: str) -> str:
    """
    原文と訂正文の差分をHTML形式で表示する
    
    Args:
        original: 原文テキスト
        corrected: 訂正済みテキスト
    
    Returns:
        差分表示のHTML
    """
    # difflib を使用して差分を取得
    d = difflib.HtmlDiff()
    diff_html = d.make_file(original.splitlines(), corrected.splitlines(), 
                           fromdesc="訂正前", todesc="訂正後")
    
    # スタイルを追加
    styled_html = diff_html.replace(
        '<style type="text/css">',
        '<style type="text/css">\n'
        'body { font-family: sans-serif; font-size: 14px; }\n'
        'table.diff { border-collapse: collapse; width: 100%; }\n'
        'td { padding: 5px; vertical-align: top; white-space: pre-wrap; }\n'
        '.diff_add { background-color: #aaffaa; }\n'
        '.diff_chg { background-color: #ffff77; }\n'
        '.diff_sub { background-color: #ffaaaa; }\n'
    )
    
    return styled_html

def highlight_corrections(original: str, corrected: str) -> str:
    """
    訂正箇所の連続した変更行を一つの枠でまとめてハイライトするHTMLを生成する
    
    Args:
        original: 原文テキスト
        corrected: 訂正済みテキスト
    
    Returns:
        ハイライト表示のHTML
    """
    import difflib
    import html
    
    original_lines = original.splitlines(keepends=True)
    corrected_lines = corrected.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, original_lines, corrected_lines)
    result = []
    in_group = False
    group_lines = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            if in_group:
                # 直前まで変更グループだった場合、まとめて枠で出力
                result.append('<div class="highlight-group">' + ''.join(group_lines) + '</div>')
                group_lines = []
                in_group = False
            result.extend([html.escape(line) for line in corrected_lines[j1:j2]])
        elif tag == 'replace' or tag == 'insert':
            in_group = True
            for line in corrected_lines[j1:j2]:
                group_lines.append(f'<span class="highlight">{html.escape(line)}</span>')
        # 削除部分は表示しない
    # 最後が変更グループで終わる場合
    if in_group and group_lines:
        result.append('<div class="highlight-group">' + ''.join(group_lines) + '</div>')
    html_result = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset=\"UTF-8\">
        <style>
            body {{ font-family: sans-serif; font-size: 14px; line-height: 1.6; }}
            .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
            .highlight {{ color: red; font-weight: bold; background: #fff0f0; }}
            .highlight-group {{ border: 2px solid #ff8888; border-radius: 6px; margin: 8px 0; padding: 4px 8px; background: #fff8f8; }}
        </style>
    </head>
    <body>
        <div class=\"container\">
            {''.join(result)}
        </div>
    </body>
    </html>
    """
    return html_result
