#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的博客生成器 - 支持最少前置数据的Papers解读，增加随机图片选择功能
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
        self.articles = []
        self.papers = []
        self.available_images = self.load_available_images()
        
    def load_available_images(self) -> List[str]:
        """加载images目录下的所有图片文件"""
        images_dir = Path(self.config['images_dir'])
        if not images_dir.exists():
            print(f"Warning: Images directory '{images_dir}' not found. Creating directory...")
            images_dir.mkdir(exist_ok=True)
            return []
        
        # 支持的图片格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.webp', '*.bmp', '*.svg']
        images = []
        
        for ext in image_extensions:
            images.extend(glob.glob(str(images_dir / ext)))
            images.extend(glob.glob(str(images_dir / ext.upper())))
        
        # 转换为相对路径
        relative_images = []
        for img_path in images:
            relative_path = os.path.relpath(img_path).replace('\\', '/')
            relative_images.append(relative_path)
        
        print(f"Found {len(relative_images)} images in {images_dir}")
        return relative_images
    
    def get_random_image(self) -> str:
        """随机获取一张图片，如果没有图片则返回默认封面"""
        if self.available_images:
            return random.choice(self.available_images)
        else:
            return self.config['default_cover']
        
    def load_config(self) -> Dict:
        """加载配置"""
        default_config = {
            "blog_title": "Cialtion's Tech Blog",
            "blog_subtitle": "Simple Love", 
            "author": "Cialtion",
            "description": "Sharing insights on LLM, LLM_Memory, Omni, AI Desktop Robot, AI Companion and Anime",
            "avatar_text": "C",
            "contact": {
                "university": "Graduated from SJTU (chemistry, 2021~2025), pursuing PhD degree in SJTU and Sii(LLM, 2025~)",
                "fields": "LLM | LLM_Memory | Omni | AI Companion | AI Desktop Robot",
                "email_school": "cialtion737410@sjtu.edu.cn",
                "email_personal": "cialtion@outlook.com"
            },
            "social_links": {
                "github": "https://github.com/HaxxorCialtion",
                "bilibili": "https://space.bilibili.com/Cialtion", 
                "zhihu": "https://www.zhihu.com/people/Cialtion",
                "twitter": "#"
            },
            "markdown_dir": "markdown_posts",
            "papers_dir": "papers_notes",
            "output_dir": "posts",
            "papers_output_dir": "papers",
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
        """解析前置元数据"""
        if content.startswith('---'):
            try:
                _, frontmatter, markdown_content = content.split('---', 2)
                metadata = yaml.safe_load(frontmatter.strip())
                return metadata or {}, markdown_content.strip()
            except:
                return {}, content
        return {}, content
    
    def extract_summary(self, content: str, max_length: int = 150) -> str:
        """提取摘要"""
        content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        content = re.sub(r'#+\s+.*?\n', '', content)
        content = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', content)
        content = re.sub(r'!\[([^\]]*)\]\([^\)]*\)', '', content)
        content = re.sub(r'[*_`]', '', content)
        content = ' '.join(content.split())
        return content[:max_length] + ('...' if len(content) > max_length else '')
    
    def smart_extract_tags(self, content: str, is_paper: bool = False) -> List[str]:
        """智能提取标签"""
        tags = []
        
        # 从内容中提取常见的技术关键词
        keywords = {
            'LLM': ['llm', 'large language model', 'gpt', 'bert', 'transformer'],
            'NLP': ['nlp', 'natural language', 'text processing', '自然语言'],
            'AI': ['artificial intelligence', '人工智能', 'ai'],
            'ML': ['machine learning', '机器学习', 'deep learning', '深度学习'],
            'Papers': ['论文', 'paper', 'research', '研究'],
            'Memory': ['memory', '记忆', 'retrieval', '检索'],
            'Multimodal': ['multimodal', '多模态', 'vision', 'image'],
            'Robot': ['robot', '机器人', 'robotics'],
            'Attention': ['attention', '注意力', 'self-attention']
        }
        
        content_lower = content.lower()
        for tag, related_words in keywords.items():
            if any(word in content_lower for word in related_words):
                tags.append(tag)
        
        # 论文默认添加Research标签
        if is_paper and 'Research' not in tags:
            tags.append('Research')
            
        return tags[:5]  # 最多5个标签
    
    def extract_title_from_content(self, content: str, filename: str) -> str:
        """从内容中提取标题"""
        # 寻找第一个一级标题
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        
        # 如果没有找到标题，使用文件名
        return filename.replace('_', ' ').replace('-', ' ').title()
    
    def convert_markdown_to_html(self, markdown_file: str, is_paper: bool = False) -> Optional[Dict]:
        """转换Markdown为HTML"""
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
            
            # 智能提取信息
            title = metadata.get('title') or self.extract_title_from_content(markdown_content, filename)
            description = metadata.get('description') or self.extract_summary(markdown_content)
            
            # 修改这里：只使用前置元数据中的tags，如果没有就设为空列表
            tags = metadata.get('tags', [])
            
            # 如果metadata中没有指定cover，则随机选择一张图片
            cover_image = metadata.get('cover') or self.get_random_image()
            
            result = {
                'title': title,
                'description': description,
                'date': metadata.get('date', datetime.now().strftime('%Y-%m-%d')),
                'tags': tags,  # 直接使用前置元数据中的tags
                'cover_image': cover_image,
                'filename': filename,
                'html_content': article_html,
                'toc': getattr(md, 'toc', '')
            }
            
            return result
        except Exception as e:
            print(f"Error converting {markdown_file}: {e}")
            return None
    
    def generate_article_html(self, article_info: Dict, is_paper: bool = False) -> str:
        """生成文章HTML"""
        try:
            with open('templates/article_template.html', 'r', encoding='utf-8') as f:
                template = f.read()
        except FileNotFoundError:
            print("⚠ Template file not found: templates/article_template.html")
            print("Please run: python3 template_generator.py")
            return ""
        
        # 替换模板变量
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
            '{{EMAIL}}': self.config['contact']['email_school']
        }
        
        for placeholder, value in replacements.items():
            template = template.replace(placeholder, str(value))
        
        return template
    
    def _generate_tags_html(self, tags: List[str]) -> str:
        """生成标签HTML"""
        if not tags:  # 如果没有tags就返回空字符串
            return ""
        tags_html = "".join([f'<span class="tag">{tag}</span>' for tag in tags])
        return f'<div class="article-tags">{tags_html}</div>'
        
    def _generate_toc_html(self, toc_content: str) -> str:
        """生成目录HTML"""
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
        """处理所有Markdown文件"""
        # 处理普通文章
        self._process_files(self.config['markdown_dir'], self.config['output_dir'], False)
        
        # 处理论文笔记
        self._process_files(self.config.get('papers_dir', 'papers_notes'), 
                          self.config.get('papers_output_dir', 'papers'), True)
    
    def _process_files(self, source_dir: str, output_dir: str, is_paper: bool):
        """处理指定目录的文件"""
        source_path = Path(source_dir)
        output_path = Path(output_dir)
        
        output_path.mkdir(exist_ok=True)
        
        if not source_path.exists():
            print(f"Directory not found: {source_path}")
            source_path.mkdir(exist_ok=True)
            return
        
        markdown_files = list(source_path.glob("*.md"))
        if not markdown_files:
            print(f"No Markdown files found in {source_path}")
            return
            
        for md_file in markdown_files:
            print(f"Processing: {md_file}")
            article_info = self.convert_markdown_to_html(str(md_file), is_paper)
            
            if article_info:
                html_content = self.generate_article_html(article_info, is_paper)
                if html_content:
                    html_file = output_path / f"{article_info['filename']}.html"
                    
                    with open(html_file, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    
                    # 添加到相应列表
                    item_data = {
                        'title': article_info['title'],
                        'description': article_info['description'],
                        'date': article_info['date'],
                        'tags': article_info['tags'],
                        'image': article_info['cover_image'],
                        'link': f"{output_dir}/{article_info['filename']}.html"
                    }
                    
                    if is_paper:
                        self.papers.append(item_data)
                    else:
                        self.articles.append(item_data)
                    
                    print(f"✅ Generated: {html_file}")
    
    def update_main_blog(self):
        """更新主页"""
        try:
            with open('templates/index_template.html', 'r', encoding='utf-8') as f:
                template = f.read()
        except FileNotFoundError:
            print("⚠ Template file not found: templates/index_template.html")
            print("Please run: python3 template_generator.py")
            return
        
        sorted_articles = sorted(self.articles, key=lambda x: x['date'], reverse=True)
        
        # 生成文章卡片HTML
        articles_html = ""
        
        # 首先添加论文解读特殊卡片
        papers_card = f'''
        <div class="article-card special-card">
        <div class="special-icon">
        <i class="fas fa-graduation-cap"></i>
        </div>
        <div class="article-content">
        <div class="article-date">
        <i class="fas fa-star"></i>
        Special Section
        </div>
        <h3 class="article-title">Paper Reviews & Analysis</h3>
        <p class="article-description">Academic paper reviews, research notes and insights on LLM, AI and cutting-edge technologies</p>
        <p class="disclaimer"><strong>Disclaimer:</strong> These notes represent personal viewpoints, and most interpretive content is generated with AI assistance (e.g., Claude 4). There may be understanding biases or factual errors. For reference only.</p>
        <div class="article-tags">
        <span class="tag">Research</span>
        <span class="tag">Academic</span>
        <span class="tag">Papers</span>
        </div>
        <a href="papers.html" class="read-more">
        Explore Papers <i class="fas fa-arrow-right"></i>
        </a>
        </div>
        </div>'''
        
        articles_html += papers_card
        
        # 然后添加普通文章卡片
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
        
        # 为主页头部选择一张随机图片
        header_image = self.get_random_image()
        
        # 替换模板变量
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
        
        print("✅ Main blog page updated")
    
    def generate_papers_page(self):
        """生成论文解读页面"""
        try:
            with open('templates/papers_template.html', 'r', encoding='utf-8') as f:
                template = f.read()
        except FileNotFoundError:
            print("⚠ Template file not found: templates/papers_template.html")
            print("Please run: python3 template_generator.py")
            return
        
        # 按日期排序论文
        sorted_papers = sorted(self.papers, key=lambda x: x['date'], reverse=True)
        
        # 为Papers页面头部选择一张随机图片
        papers_header_image = self.get_random_image()
        
        # 生成论文卡片HTML
        papers_html = ""
        if sorted_papers:
            for paper in sorted_papers:
                tags_html = " ".join([f'<span class="tag">{tag}</span>' for tag in paper['tags']])
                
                papers_html += f'''
                <div class="paper-card">
                    <img src="{paper['image']}" alt="{paper['title']}" class="paper-image" onerror="this.style.display='none'">
                    <div class="paper-content">
                        <div class="paper-meta">
                            <div class="meta-item">
                                <i class="fas fa-calendar-alt"></i>
                                {paper['date']}
                            </div>
                        </div>
                        <h3 class="paper-title">{paper['title']}</h3>
                        <p class="paper-description">{paper['description']}</p>
                        <div class="paper-tags">{tags_html}</div>
                        <a href="{paper['link']}" class="read-more">
                            Read Review <i class="fas fa-arrow-right"></i>
                        </a>
                    </div>
                </div>'''
        else:
            papers_html = '''
            <div class="no-papers">
                <i class="fas fa-book-open"></i>
                <h3>No Papers Yet</h3>
                <p>论文解读正在准备中，敬请期待...</p>
            </div>'''
        
        # 替换模板变量
        replacements = {
            '{{BLOG_TITLE}}': self.config['blog_title'],
            '{{AUTHOR}}': self.config['author'],
            '{{GITHUB_URL}}': self.config['social_links']['github'],
            '{{EMAIL}}': self.config['contact']['email_school'],
            '{{HEADER_IMAGE}}': papers_header_image,
            '{{PAPERS_HTML}}': papers_html
        }
        
        for placeholder, value in replacements.items():
            template = template.replace(placeholder, str(value))
        
        with open('papers.html', 'w', encoding='utf-8') as f:
            f.write(template)
        
        print("✅ Papers page generated: papers.html")

def main():
    """主函数"""
    print("🚀 Starting blog generator...")
    
    generator = BlogGenerator()
    
    # 处理Markdown文件
    generator.process_markdown_files()
    
    # 更新主页
    generator.update_main_blog()
    print(f"✅ Successfully processed {len(generator.articles)} articles and {len(generator.papers)} papers")
    
    # 生成论文解读页面
    generator.generate_papers_page()
    
    print("🎉 Blog generation completed!")

if __name__ == "__main__":
    main()