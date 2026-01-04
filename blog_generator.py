#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的博客生成器 - 支持多语言切换和多个特殊窗口
(新增) 支持 LaTeX/PDF 格式的组会汇报记录 (meetings)
"""

import os
import json
import re
import markdown
import yaml
import random
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class BlogGenerator:
    def __init__(self, config_path: str = "blog_config.json"):
        self.config_path = config_path
        self.config = self.load_config()

        # 为所有内容类型初始化列表
        self.articles = []
        self.papers = []
        self.reviews = []
        self.open_sources = []
        self.products = []
        self.meetings = []  # (新增) 组会汇报记录

        # Pass 1: 存储所有文章的完整信息 (用于 Pass 2)
        self.all_processed_info = []
        # Pass 2: 存储翻译地图
        self.translation_map = {}

        self.available_images = self.load_available_images()

    def load_available_images(self) -> List[str]:
        images_dir = Path(self.config['images_dir'])
        if not images_dir.exists():
            print(f"Warning: Images directory '{images_dir}' not found. Creating directory...")
            images_dir.mkdir(exist_ok=True)
            return []
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.webp', '*.bmp', '*.svg']
        images = []
        for ext in image_extensions:
            images.extend(glob.glob(str(images_dir / ext)))
            images.extend(glob.glob(str(images_dir / ext.upper())))
        relative_images = []
        for img_path in images:
            relative_path = os.path.relpath(img_path).replace('\\', '/')
            relative_images.append(relative_path)
        print(f"Found {len(relative_images)} images in {images_dir}")
        return relative_images

    def get_random_image(self) -> str:
        if self.available_images:
            return random.choice(self.available_images)
        else:
            return self.config['default_cover']

    def load_config(self) -> Dict:
        default_config = {
            "blog_title": "Cialtion's Tech Blog",
            "blog_subtitle": "Simple Love",
            "author": "Cialtion",
            "description": "Sharing insights on LLM, LLM_Memory, Omni, Agent, AI-Human Interaction, AI Companion",
            "avatar_text": "C",
            "contact": {
                "university": "Graduated from SJTU (chemistry, 2021~2025), pursuing PhD degree in SJTU and Sii(LLM, 2025~)",
                "fields": "LLM | LLM_Memory | Omni | AI Companion | AI Desktop Robot",
                "email_school": "cialtion737410@sjtu.edu.cn",
                "email_personal": "cialtion@outlook.com"
            },
            "social_links": {
                "github": "https://github.com/HaxxorCialtion",
                "bilibili": "https://space.bili.com/Cialtion",
                "zhihu": "https://www.zhihu.com/people/Cialtion",
                "twitter": "#"
            },
            "markdown_dir": "markdown_posts",
            "output_dir": "posts",
            "papers_dir": "papers_notes",
            "papers_output_dir": "papers",
            "reviews_dir": "reviews_notes",
            "reviews_output_dir": "reviews",
            "opensource_dir": "opensource_notes",
            "opensource_output_dir": "opensource",
            "products_dir": "products_notes",
            "products_output_dir": "products",
            # (新增) meetings 配置
            "meetings_dir": "meetings_notes",
            "meetings_output_dir": "meetings",
            "meetings_pdf_dir": "meetings_pdf",  # 存放编译好的 PDF
            "images_dir": "images",
            "default_cover": "images/default_cover.jpg"
        }
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        else:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
        return default_config

    def parse_frontmatter(self, content: str) -> tuple:
        if content.startswith('---'):
            try:
                _, frontmatter, markdown_content = content.split('---', 2)
                metadata = yaml.safe_load(frontmatter.strip())
                return metadata or {}, markdown_content.strip()
            except:
                return {}, content
        return {}, content

    def extract_summary(self, content: str, max_length: int = 150) -> str:
        content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        content = re.sub(r'#+\s+.*?\n', '', content)
        content = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', content)
        content = re.sub(r'!\[([^\]]*)\]\([^\)]*\)', '', content)
        content = re.sub(r'[*_`]', '', content)
        content = ' '.join(content.split())
        return content[:max_length] + ('...' if len(content) > max_length else '')

    def extract_title_from_content(self, content: str, filename: str) -> str:
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        return filename.replace('_', ' ').replace('-', ' ').title()

    def convert_markdown_to_html(self, markdown_file: str) -> Optional[Dict]:
        """
        转换Markdown为HTML
        """
        try:
            with open(markdown_file, 'r', encoding='utf-8') as f:
                content = f.read()

            metadata, markdown_content = self.parse_frontmatter(content)

            md = markdown.Markdown(extensions=[
                'codehilite', 'tables', 'fenced_code', 'toc',
                'attr_list', 'def_list', 'footnotes'
            ], extension_configs={
                'codehilite': {'css_class': 'highlight'},
                'toc': {'title': 'Table of Contents'}
            })

            article_html = md.convert(markdown_content)
            filename = Path(markdown_file).stem

            title = metadata.get('title') or self.extract_title_from_content(markdown_content, filename)
            description = metadata.get('description') or self.extract_summary(markdown_content)
            tags = metadata.get('tags', [])
            cover_image = metadata.get('cover') or self.get_random_image()

            result = {
                'title': title,
                'description': description,
                'date': metadata.get('date', datetime.now().strftime('%Y-%m-%d')),
                'tags': tags,
                'cover_image': cover_image,
                'filename': filename,
                'html_content': article_html,
                'toc': getattr(md, 'toc', ''),
                'lang': metadata.get('lang', 'zh'),
                'translation_id': metadata.get('translation_id'),
                # (新增) 支持 PDF 路径
                'pdf_path': metadata.get('pdf'),
                'content_type': metadata.get('type', 'markdown'),  # 'markdown' 或 'pdf'
            }

            return result
        except Exception as e:
            print(f"Error converting {markdown_file}: {e}")
            return None

    def generate_article_html(self, article_info: Dict, translations: List[Dict] = None) -> str:
        """
        生成文章HTML
        """
        try:
            with open('templates/article_template.html', 'r', encoding='utf-8') as f:
                template = f.read()
        except FileNotFoundError:
            print("⚠ Template file not found: templates/article_template.html")
            print("Please run: python3 template_generator.py")
            return ""

        translations_html = self._generate_translations_html(
            translations,
            article_info.get('lang')
        )

        replacements = {
            '{{TITLE}}': article_info['title'],
            '{{BLOG_TITLE}}': self.config['blog_title'],
            '{{DESCRIPTION}}': article_info['description'],
            '{{DATE}}': article_info['date'],
            '{{AUTHOR}}': self.config['author'],
            '{{COVER_IMAGE}}': f"../{article_info['cover_image']}",
            '{{TAGS}}': self._generate_tags_html(article_info['tags']),
            '{{TOC}}': self._generate_toc_html(article_info['toc']),
            '{{CONTENT}}': article_info['html_content'],
            '{{GITHUB_URL}}': self.config['social_links']['github'],
            '{{EMAIL}}': self.config['contact']['email_school'],
            '{{TRANSLATIONS}}': translations_html,
        }

        for placeholder, value in replacements.items():
            template = template.replace(placeholder, str(value))

        return template

    def generate_meeting_html(self, meeting_info: Dict) -> str:
        """
        (新增) 生成组会记录的 HTML 页面 - 学术论文风格，嵌入 PDF
        """
        try:
            with open('templates/meeting_article_template.html', 'r', encoding='utf-8') as f:
                template = f.read()
        except FileNotFoundError:
            print("⚠ Template file not found: templates/meeting_article_template.html")
            print("Please run: python3 template_generator.py")
            return ""

        pdf_path = meeting_info.get('pdf_path', '')
        # 确保 PDF 路径正确（相对于输出目录）
        if pdf_path and not pdf_path.startswith('..'):
            pdf_path = f"../{pdf_path}"

        replacements = {
            '{{TITLE}}': meeting_info['title'],
            '{{BLOG_TITLE}}': self.config['blog_title'],
            '{{DESCRIPTION}}': meeting_info['description'],
            '{{DATE}}': meeting_info['date'],
            '{{AUTHOR}}': self.config['author'],
            '{{TAGS}}': self._generate_tags_html(meeting_info['tags']),
            '{{PDF_PATH}}': pdf_path,
            '{{CONTENT}}': meeting_info.get('html_content', ''),
            '{{GITHUB_URL}}': self.config['social_links']['github'],
            '{{EMAIL}}': self.config['contact']['email_school'],
        }

        for placeholder, value in replacements.items():
            template = template.replace(placeholder, str(value))

        return template

    def _generate_translations_html(self, translations: Optional[List[Dict]], current_lang: str) -> str:
        if not translations or len(translations) < 2:
            return ""

        lang_map = {
            'en': 'English',
            'zh': '中文',
        }

        links_html = ""
        sorted_translations = sorted(translations, key=lambda x: x.get('lang', ''))

        for trans in sorted_translations:
            lang_code = trans.get('lang')
            lang_name = lang_map.get(lang_code, lang_code)
            href = trans.get('link', '#')
            fixed_href = f"../{href}"

            if lang_code == current_lang:
                links_html += f'<a href="{fixed_href}" class="active">{lang_name}</a>'
            else:
                links_html += f'<a href="{fixed_href}">{lang_name}</a>'

        return f'''
        <div class="translation-links">
            <span>Read in: </span>
            {links_html}
        </div>'''

    def _generate_tags_html(self, tags: List[str]) -> str:
        if not tags:
            return ""
        tags_html = "".join([f'<span class="tag">{tag}</span>' for tag in tags])
        return f'<div class="article-tags">{tags_html}</div>'

    def _generate_toc_html(self, toc_content: str) -> str:
        if not toc_content:
            return ""
        return f'''<div class="toc-container">
            <div class="toc-title">
                <i class="fas fa-list"></i>
                Table of Contents
            </div>
            <div class="toc">{toc_content}</div>
        </div>'''

    def process_markdown_files(self):
        """
        Pass 1: 处理所有Markdown文件，仅收集数据。
        """
        content_types = [
            {'type': 'article', 'list': self.articles, 'src_key': 'markdown_dir', 'out_key': 'output_dir',
             'default_src': 'markdown_posts', 'default_out': 'posts'},
            {'type': 'paper', 'list': self.papers, 'src_key': 'papers_dir', 'out_key': 'papers_output_dir',
             'default_src': 'papers_notes', 'default_out': 'papers'},
            {'type': 'review', 'list': self.reviews, 'src_key': 'reviews_dir', 'out_key': 'reviews_output_dir',
             'default_src': 'reviews_notes', 'default_out': 'reviews'},
            {'type': 'opensource', 'list': self.open_sources, 'src_key': 'opensource_dir',
             'out_key': 'opensource_output_dir', 'default_src': 'opensource_notes', 'default_out': 'opensource'},
            {'type': 'product', 'list': self.products, 'src_key': 'products_dir', 'out_key': 'products_output_dir',
             'default_src': 'products_notes', 'default_out': 'products'},
            # (新增) meetings
            {'type': 'meeting', 'list': self.meetings, 'src_key': 'meetings_dir', 'out_key': 'meetings_output_dir',
             'default_src': 'meetings_notes', 'default_out': 'meetings'},
        ]

        for ct in content_types:
            src_dir = self.config.get(ct['src_key'], ct['default_src'])
            out_dir = self.config.get(ct['out_key'], ct['default_out'])
            self._process_files_pass1(src_dir, out_dir, ct['list'], ct['type'])

        print(f"--- Pass 1 Complete: Collected {len(self.all_processed_info)} total items ---")

    def _process_files_pass1(self, source_dir: str, output_dir: str, target_list: List[Dict], content_type: str):
        """
        仅执行数据收集
        """
        source_path = Path(source_dir)
        output_path = Path(output_dir)

        output_path.mkdir(exist_ok=True)

        if not source_path.exists():
            print(f"Directory not found: {source_path}. Creating...")
            source_path.mkdir(exist_ok=True)
            return

        markdown_files = list(source_path.glob("*.md"))
        if not markdown_files:
            print(f"No Markdown files found in {source_path}")
            return

        print(f"--- Processing {content_type} files from {source_path} ---")
        for md_file in markdown_files:
            print(f"Processing: {md_file}")
            article_info = self.convert_markdown_to_html(str(md_file))

            if article_info:
                article_info['output_path'] = output_path / f"{article_info['filename']}.html"
                article_info['content_type_category'] = content_type  # 标记类型

                self.all_processed_info.append(article_info)

                item_data = {
                    'title': article_info['title'],
                    'description': article_info['description'],
                    'date': article_info['date'],
                    'tags': article_info['tags'],
                    'image': article_info['cover_image'],
                    'link': f"{output_dir}/{article_info['filename']}.html",
                    'lang': article_info['lang'],
                    'translation_id': article_info['translation_id'],
                    # (新增) PDF 相关
                    'pdf_path': article_info.get('pdf_path'),
                }

                target_list.append(item_data)

    def _build_translation_map(self):
        print("--- Pass 1.5: Building translation map ---")
        all_items = self.articles + self.papers + self.reviews + self.open_sources + self.products + self.meetings

        for item in all_items:
            trans_id = item.get('translation_id')
            if trans_id:
                if trans_id not in self.translation_map:
                    self.translation_map[trans_id] = []

                self.translation_map[trans_id].append({
                    'lang': item.get('lang'),
                    'link': item.get('link')
                })
        print(f"Found {len(self.translation_map)} translation groups.")

    def _generate_all_html_pass2(self):
        """Pass 2: 使用翻译地图生成所有 HTML 文件"""
        print(f"--- Pass 2: Generating {len(self.all_processed_info)} HTML files ---")

        for article_info in self.all_processed_info:
            translations = []
            trans_id = article_info.get('translation_id')

            if trans_id and trans_id in self.translation_map:
                translations = self.translation_map[trans_id]

            # (新增) 根据类型选择不同的生成器
            if article_info.get('content_type_category') == 'meeting':
                html_content = self.generate_meeting_html(article_info)
            else:
                html_content = self.generate_article_html(article_info, translations)

            if html_content:
                html_file = article_info['output_path']
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                print(f"✅ Generated: {html_file}")

    def update_main_blog(self):
        """
        更新主页
        """
        try:
            with open('templates/index_template.html', 'r', encoding='utf-8') as f:
                template = f.read()
        except FileNotFoundError:
            print("⚠ Template file not found: templates/index_template.html")
            print("Please run: python3 template_generator.py")
            return

        default_lang = 'zh'

        display_articles = [
            article for article in self.articles
            if article.get('lang', default_lang) == default_lang
        ]

        sorted_articles = sorted(display_articles, key=lambda x: x['date'], reverse=True)

        articles_html = ""

        # --- 特殊卡片区 ---
        papers_card = f'''
        <div class="article-card special-card">
        <div class="special-icon"><i class="fas fa-graduation-cap"></i></div>
        <div class="article-content">
        <div class="article-date"><i class="fas fa-star"></i> Special Section</div>
        <h3 class="article-title">Paper Reviews & Analysis</h3>
        <p class="article-description">Academic paper reviews, research notes and insights on LLM, AI and cutting-edge technologies</p>
        <p class="disclaimer"><strong>Disclaimer:</strong> These notes represent personal viewpoints, and most interpretive content is generated with AI assistance (e.g., Claude 4). There may be understanding biases or factual errors. For reference only.</p>
        <div class="article-tags"><span class="tag">Research</span><span class="tag">Papers</span></div>
        <a href="papers.html" class="read-more">Explore Papers <i class="fas fa-arrow-right"></i></a>
        </div>
        </div>'''

        reviews_card = f'''
        <div class="article-card special-card">
        <div class="special-icon"><i class="fas fa-search-plus"></i></div>
        <div class="article-content">
        <div class="article-date"><i class="fas fa-star"></i> Special Section</div>
        <h3 class="article-title">In-Depth Reviews</h3>
        <p class="article-description">Detailed analysis, benchmarks, and hands-on reviews of the latest AI models, software, and technologies.</p>
        <div class="article-tags"><span class="tag">Review</span><span class="tag">Benchmark</span><span class="tag">Analysis</span></div>
        <a href="reviews.html" class="read-more">Explore Reviews <i class="fas fa-arrow-right"></i></a>
        </div>
        </div>'''

        opensource_card = f'''
        <div class="article-card special-card">
        <div class="special-icon"><i class="fas fa-code-branch"></i></div>
        <div class="article-content">
        <div class="article-date"><i class="fas fa-star"></i> Special Section</div>
        <h3 class="article-title">Open-Source Projects</h3>
        <p class="article-description">My personal open-source contributions, libraries, and tools. Feel free to use, fork, and contribute!</p>
        <div class="article-tags"><span class="tag">Open-Source</span><span class="tag">Code</span><span class="tag">GitHub</span></div>
        <a href="opensource.html" class="read-more">Explore Projects <i class="fas fa-arrow-right"></i></a>
        </div>
        </div>'''

        products_card = f'''
        <div class="article-card special-card">
        <div class="special-icon"><i class="fas fa-box-open"></i></div>
        <div class="article-content">
        <div class="article-date"><i class="fas fa-star"></i> Special Section</div>
        <h3 class="article-title">Product Showcases</h3>
        <p class="article-description">Showcasing fun and practical AI-powered applications and products I've built or am working on.</p>
        <div class="article-tags"><span class="tag">Product</span><span class="tag">Demo</span><span class="tag">Showcase</span></div>
        <a href="products.html" class="read-more">Explore Products <i class="fas fa-arrow-right"></i></a>
        </div>
        </div>'''

        # (新增) 组会记录卡片
        meetings_card = f'''
        <div class="article-card special-card meetings-card">
        <div class="special-icon"><i class="fas fa-chalkboard-teacher"></i></div>
        <div class="article-content">
        <div class="article-date"><i class="fas fa-star"></i> Special Section</div>
        <h3 class="article-title">Group Meeting Reports</h3>
        <p class="article-description">组会汇报记录 - LaTeX 排版的学术汇报文档，包含研究进展、论文分享与讨论。</p>
        <div class="article-tags"><span class="tag">Meeting</span><span class="tag">LaTeX</span><span class="tag">Academic</span></div>
        <a href="meetings.html" class="read-more">View Reports <i class="fas fa-arrow-right"></i></a>
        </div>
        </div>'''

        articles_html += papers_card
        articles_html += reviews_card
        articles_html += opensource_card
        articles_html += products_card
        articles_html += meetings_card  # (新增)

        # --- 普通文章卡片 ---
        for article in sorted_articles:
            tags_html = " ".join([f'<span class="tag">{tag}</span>' for tag in article['tags']])
            articles_html += f'''
            <div class="article-card">
                <img src="{article['image']}" alt="{article['title']}" class="article-image" onerror="this.style.display='none'">
                <div class="article-content">
                    <div class="article-date">
                        <i class="fas fa-calendar-alt"></i>
                        {article['date']}
                    </div>
                    <h3 class="article-title">{article['title']}</h3>
                    <p class="article-description">{article['description']}</p>
                    <div class="article-tags">{tags_html}</div>
                    <a href="{article['link']}" class="read-more">
                        Read More <i class="fas fa-arrow-right"></i>
                    </a>
                </div>
            </div>'''

        header_image = self.get_random_image()

        replacements = {
            '{{BLOG_TITLE}}': self.config['blog_title'],
            '{{BLOG_SUBTITLE}}': self.config['blog_subtitle'],
            '{{AUTHOR}}': self.config['author'],
            '{{DESCRIPTION}}': self.config['description'],
            '{{AVATAR_TEXT}}': self.config['avatar_text'],
            '{{UNIVERSITY}}': self.config['contact']['university'],
            '{{FIELDS}}': self.config['contact']['fields'],
            '{{GITHUB_URL}}': self.config['social_links']['github'],
            '{{EMAIL_SCHOOL}}': self.config['contact']['email_school'],
            '{{EMAIL_PERSONAL}}': self.config['contact']['email_personal'],
            '{{BILIBILI_URL}}': self.config['social_links']['bilibili'],
            '{{ZHIHU_URL}}': self.config['social_links']['zhihu'],
            '{{TWITTER_URL}}': self.config['social_links']['twitter'],
            '{{HEADER_IMAGE}}': header_image,
            '{{ARTICLES_HTML}}': articles_html
        }

        for placeholder, value in replacements.items():
            template = template.replace(placeholder, str(value))

        with open('index.html', 'w', encoding='utf-8') as f:
            f.write(template)

        print("✅ Main blog page updated (filtered for default lang)")

    def _generate_special_page(self,
                               template_file: str,
                               output_file: str,
                               items: List[Dict],
                               page_title: str,
                               no_items_icon: str = "fas fa-book-open"):
        """
        生成特殊页面
        """
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                template = f.read()
        except FileNotFoundError:
            print(f"⚠ Template file not found: {template_file}")
            print("Please run: python3 template_generator.py")
            return

        default_lang = 'zh'
        display_items = [
            item for item in items
            if item.get('lang', default_lang) == default_lang
        ]

        sorted_items = sorted(display_items, key=lambda x: x['date'], reverse=True)

        header_image = self.get_random_image()

        items_html = ""
        if sorted_items:
            for item in sorted_items:
                tags_html = " ".join([f'<span class="tag">{tag}</span>' for tag in item['tags']])
                # (新增) 如果有 PDF，添加下载按钮
                pdf_button = ""
                if item.get('pdf_path'):
                    pdf_button = f'<a href="{item["pdf_path"]}" class="pdf-download" target="_blank"><i class="fas fa-file-pdf"></i> PDF</a>'

                items_html += f'''
                <div class="paper-card">
                    <img src="{item['image']}" alt="{item['title']}" class="paper-image" onerror="this.style.display='none'">
                    <div class="paper-content">
                        <div class="paper-meta">
                            <div class="meta-item">
                                <i class="fas fa-calendar-alt"></i>
                                {item['date']}
                            </div>
                        </div>
                        <h3 class="paper-title">{item['title']}</h3>
                        <p class="paper-description">{item['description']}</p>
                        <div class="paper-tags">{tags_html}</div>
                        <div class="paper-actions">
                            <a href="{item['link']}" class="read-more">
                                Read More <i class="fas fa-arrow-right"></i>
                            </a>
                            {pdf_button}
                        </div>
                    </div>
                </div>'''
        else:
            items_html = f'''
            <div class="no-papers">
                <i class="{no_items_icon}"></i>
                <h3>No {page_title} Yet</h3>
                <p>{page_title} 正在准备中，敬请期待...</p>
            </div>'''

        replacements = {
            '{{BLOG_TITLE}}': self.config['blog_title'],
            '{{AUTHOR}}': self.config['author'],
            '{{GITHUB_URL}}': self.config['social_links']['github'],
            '{{EMAIL}}': self.config['contact']['email_school'],
            '{{HEADER_IMAGE}}': header_image,
            '{{PAPERS_HTML}}': items_html
        }

        for placeholder, value in replacements.items():
            template = template.replace(placeholder, str(value))

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(template)

        print(f"✅ {page_title} page generated: {output_file} (filtered for default lang)")


def main():
    """
    主函数
    """
    print("🚀 Starting blog generator...")

    generator = BlogGenerator()

    # Pass 1: 处理所有Markdown文件 (仅收集数据)
    generator.process_markdown_files()

    # Pass 1.5: 构建翻译地图
    generator._build_translation_map()

    # Pass 2: 使用翻译地图生成所有文章的 HTML
    generator._generate_all_html_pass2()

    # --- 更新索引页 (使用已收集的数据) ---

    # 3. 更新主页
    generator.update_main_blog()

    # 4. 生成所有特殊页面
    generator._generate_special_page(
        template_file='templates/papers_template.html',
        output_file='papers.html',
        items=generator.papers,
        page_title='Papers',
        no_items_icon="fas fa-graduation-cap"
    )
    generator._generate_special_page(
        template_file='templates/reviews_template.html',
        output_file='reviews.html',
        items=generator.reviews,
        page_title='Reviews',
        no_items_icon="fas fa-search-plus"
    )
    generator._generate_special_page(
        template_file='templates/opensource_template.html',
        output_file='opensource.html',
        items=generator.open_sources,
        page_title='Open-Source',
        no_items_icon="fas fa-code-branch"
    )
    generator._generate_special_page(
        template_file='templates/products_template.html',
        output_file='products.html',
        items=generator.products,
        page_title='Products',
        no_items_icon="fas fa-box-open"
    )
    # (新增) 生成 meetings 页面
    generator._generate_special_page(
        template_file='templates/meetings_template.html',
        output_file='meetings.html',
        items=generator.meetings,
        page_title='Meetings',
        no_items_icon="fas fa-chalkboard-teacher"
    )

    print(f"✅ Successfully processed {len(generator.articles)} articles, "
          f"{len(generator.papers)} papers, "
          f"{len(generator.reviews)} reviews, "
          f"{len(generator.open_sources)} open-sources, "
          f"{len(generator.products)} products, "
          f"and {len(generator.meetings)} meetings.")  # (新增)

    print("🎉 Blog generation completed!")


if __name__ == "__main__":
    main()
