import json
from bs4 import BeautifulSoup
import re


def load_notes(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_styles(soup):
    """确保所有必需的样式都被正确定义"""
    required_styles = {
        'img.cover': '''
img.cover {
    width: 100%;
    border-radius: 8px;
    margin: 1rem 0;
}''',
        '.tag': '''
.tag {
    background: #007acc;
    color: white;
    padding: 0.2rem 0.5rem;
    margin-right: 0.3rem;
    border-radius: 4px;
    font-size: 0.9em;
    display: inline-block;
    margin-bottom: 0.3rem;
    transition: background-color 0.3s ease;
}
.tag:hover {
    background: #005999;
}''',
        '.tags': '''
.tags {
    margin-bottom: 1rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
}''',
        '.note-card': '''
.note-card {
    background: white;
    border-radius: 8px;
    padding: 1.5rem;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
    display: flex;
    flex-direction: column;
}
.note-card:hover {
    transform: translateY(-5px);
}
.note-card h3 {
    margin: 0.5rem 0;
    color: #2c3e50;
}
.note-card .date {
    color: #666;
    font-size: 0.9em;
}
.note-card p {
    margin: 0.5rem 0;
    color: #555;
}
.note-card a {
    color: #007acc;
    text-decoration: none;
    margin-top: auto;
    padding-top: 1rem;
}
.note-card a:hover {
    color: #005999;
}
.tag-line {
    color: #888;        /* 灰色 */
    font-size: 0.9em;   /* 稍小一点的字体 */
}
'''
    }
    # 找到或创建style标签
    style_tag = soup.find('style')
    if not style_tag:
        style_tag = soup.new_tag('style')
        if soup.head:
            soup.head.append(style_tag)
        else:
            head = soup.new_tag('head')
            head.append(style_tag)
            soup.html.insert(0, head)

    # 获取现有的样式内容
    current_styles = style_tag.string or ''

    # 对于每个必需的样式
    for selector, style_definition in required_styles.items():
        # 创建用于查找的模式
        pattern = re.compile(f"{selector}\s*{{\s*[^}}]*}}", re.MULTILINE | re.DOTALL)

        # 检查是否存在该选择器的样式
        if pattern.search(current_styles):
            # 如果存在，替换它
            current_styles = pattern.sub(style_definition.strip(), current_styles)
        else:
            # 如果不存在，添加它
            current_styles += style_definition

    # 更新style标签的内容
    style_tag.string = current_styles

def clean_body_top(body):
    """移除 body 中顶部连续的封面图、标题、日期、tags 信息"""
    removable_tags = []

    for tag in body.find_all(recursive=False):
        if tag.name == 'img' and 'cover' in tag.get('class', []):
            removable_tags.append(tag)
        elif tag.name == 'h1':
            removable_tags.append(tag)
        elif tag.name == 'p' and '日期' in tag.get_text():
            removable_tags.append(tag)
        elif tag.name == 'div' and 'tags' in tag.get('class', []):
            removable_tags.append(tag)
        else:
            break

    for tag in removable_tags:
        tag.decompose()


def update_note_html(note):
    # 原有的update_note_html函数保持不变
    file_path = note.get('file')
    if not file_path:
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    ensure_styles(soup)
    body = soup.body
    if not body:
        return

    clean_body_top(body)

    if note.get('image'):
        image_path = note['image'].replace('./', '../', 1)
        img = soup.new_tag('img', src=image_path, alt="封面图", **{'class': 'cover'})
        body.insert(0, img)

    h1 = soup.new_tag('h1')
    h1.string = note.get('title', '无标题')
    body.insert(1, h1)

    if note.get('date'):
        p_date = soup.new_tag('p')
        strong = soup.new_tag('strong')
        strong.string = "日期："
        p_date.append(strong)
        p_date.append(note['date'])
        body.insert(2, p_date)

    if 'key_words' in note and isinstance(note['key_words'], list):
        tags_div = soup.new_tag('div', **{'class': 'tags'})
        for word in note['key_words']:
            span = soup.new_tag('span', **{'class': 'tag'})
            span.string = word
            tags_div.append(span)
        body.insert(3, tags_div)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(str(soup.prettify()))

    print(f"✅ 已更新：{file_path}")


def generate_index_html(notes, template_path='index_old.html'):
    """从模板生成新的index.html"""
    # 读取模板
    with open(template_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # 确保主页面也有tag相关的样式
    ensure_styles(soup)

    # 找到content-grid div并清空现有的note-card
    content_grid = soup.find('div', class_='content-grid')
    if content_grid:
        content_grid.clear()

    # 按日期排序笔记（降序）
    sorted_notes = sorted(notes, key=lambda x: x.get('date', ''), reverse=True)

    # 为每个笔记创建card
    for note in sorted_notes:
        article = soup.new_tag('article', **{'class': 'note-card'})

        # 添加日期
        date_div = soup.new_tag('div', **{'class': 'date'})
        date_div.string = note.get('date', '')
        article.append(date_div)

        # 添加标题
        h3 = soup.new_tag('h3')
        h3.string = note.get('title', '')
        article.append(h3)

        # 添加图片
        if note.get('image'):
            img = soup.new_tag('img', src=note['image'], style="max-width:100%; border-radius:8px; margin-bottom:1rem;")
            article.append(img)

        # 添加描述
        p = soup.new_tag('p')
        p.string = note.get('description', '')
        article.append(p)

        # 添加关键词标签
        if note.get('key_words'):
            # 外层div用于统一设置样式
            tags_div = soup.new_tag('div', **{'class': 'tags'})

            # 创建内容字符串，例如 "关键词: xxx yyy zzz"
            keywords_text = '关键词: ' + ' '.join(note['key_words'])

            # 创建span用于包裹整行关键词文字
            tag_line = soup.new_tag('span', **{'class': 'tag-line'})
            tag_line.string = keywords_text

            tags_div.append(tag_line)
            article.append(tags_div)

        # 添加链接
        a = soup.new_tag('a', href=note.get('file', ''))
        a.string = '阅读更多 →'
        article.append(a)

        content_grid.append(article)

    # 写入新的index.html
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(str(soup.prettify()))

    print("✅ 已更新：index.html")

if __name__ == "__main__":
    notes = load_notes('notes.json')
    # 先更新所有笔记
    for note in notes:
        update_note_html(note)
    # 然后重新生成index.html
    generate_index_html(notes)