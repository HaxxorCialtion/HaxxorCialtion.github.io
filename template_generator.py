#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新的模板生成器 - 支持多个特殊页面和导航
"""

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# 主页模板 (index_template)
# 我们在 <nav class="nav-menu"> 中添加了新页面的链接
# ----------------------------------------------------------------------------
index_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{BLOG_TITLE}}</title>
    <meta name="description" content="{{DESCRIPTION}}">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --primary-color: #2c3e50; --secondary-color: #3498db; --accent-color: #e74c3c;
            --background-color: #f8fafc; --card-background: #ffffff; --text-primary: #2d3748;
            --text-secondary: #718096; --border-color: #e2e8f0; --shadow-light: 0 4px 6px rgba(0, 0, 0, 0.07);
            --shadow-medium: 0 10px 25px rgba(0, 0, 0, 0.1); --border-radius: 12px; --transition: all 0.3s ease;
        }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background-color: var(--background-color); color: var(--text-primary); line-height: 1.6; }
        .sidebar { position: fixed; left: 0; top: 0; width: 280px; height: 100vh; background: linear-gradient(135deg, var(--primary-color) 0%, #34495e 100%); color: white; overflow-y: auto; z-index: 1000; transition: var(--transition); }
        .sidebar-header { padding: 2rem 1.5rem; text-align: center; border-bottom: 1px solid rgba(255, 255, 255, 0.1); }
        .avatar { width: 80px; height: 80px; border-radius: 50%; background: linear-gradient(135deg, var(--secondary-color), var(--accent-color)); display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-size: 2rem; font-weight: bold; }
        .sidebar-header h1 { font-size: 1.5rem; margin-bottom: 0.5rem; font-weight: 600; }
        .sidebar-header .tagline { color: rgba(255, 255, 255, 0.8); font-size: 0.9rem; }
        .nav-menu { padding: 1rem 0; }
        .nav-item { display: block; padding: 0.75rem 1.5rem; color: rgba(255, 255, 255, 0.9); text-decoration: none; transition: var(--transition); border-left: 3px solid transparent; }
        .nav-item:hover, .nav-item.active { background-color: rgba(255, 255, 255, 0.1); border-left-color: var(--secondary-color); color: white; }
        .nav-item i { width: 20px; margin-right: 0.75rem; }
        .contact-info { padding: 1rem 1.5rem; background: rgba(0, 0, 0, 0.1); margin: 1rem; border-radius: 8px; }
        .contact-info h4 { font-size: 0.9rem; margin-bottom: 0.5rem; color: rgba(255, 255, 255, 0.9); }
        .contact-info p { font-size: 0.8rem; color: rgba(255, 255, 255, 0.7); line-height: 1.4; }
        .social-links { padding: 1.5rem; border-top: 1px solid rgba(255, 255, 255, 0.1); }
        .social-links h3 { font-size: 1rem; margin-bottom: 1rem; color: rgba(255, 255, 255, 0.9); }
        .social-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; }
        .social-link { display: flex; align-items: center; padding: 0.5rem; background: rgba(255, 255, 255, 0.1); border-radius: 8px; color: white; text-decoration: none; transition: var(--transition); font-size: 0.85rem; }
        .social-link:hover { background: rgba(255, 255, 255, 0.2); transform: translateY(-1px); }
        .social-link i { margin-right: 0.5rem; width: 16px; }
        .main-content { margin-left: 280px; min-height: 100vh; background-color: var(--background-color); }
        .header { background: var(--card-background); padding: 1.5rem 2rem; border-bottom: 1px solid var(--border-color); box-shadow: var(--shadow-light); position: sticky; top: 0; z-index: 100; }
        .header-content { display: flex; justify-content: space-between; align-items: center; max-width: 1200px; margin: 0 auto; }
        .search-box { position: relative; width: 300px; }
        .search-input { width: 100%; padding: 0.75rem 1rem 0.75rem 2.5rem; border: 2px solid var(--border-color); border-radius: 25px; background-color: var(--background-color); font-size: 0.9rem; transition: var(--transition); }
        .search-input:focus { outline: none; border-color: var(--secondary-color); box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1); }
        .search-icon { position: absolute; left: 1rem; top: 50%; transform: translateY(-50%); color: var(--text-secondary); }
        .content-area { padding: 2rem; max-width: 1200px; margin: 0 auto; }
        .hero-section { text-align: center; margin-bottom: 3rem; padding: 4rem 2rem; background: linear-gradient(135deg, rgba(52, 152, 219, 0.1), rgba(231, 76, 60, 0.1)); border-radius: var(--border-radius); position: relative; overflow: hidden; }
        .hero-bg-image { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; opacity: 0.1; z-index: 1; }
        .hero-content { position: relative; z-index: 2; }
        .hero-logo { font-size: 3.5rem; font-weight: 700; background: linear-gradient(135deg, var(--secondary-color), var(--accent-color)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 1rem; line-height: 1.2; }
        .page-title { font-size: 2.5rem; font-weight: 700; color: var(--primary-color); margin-bottom: 0.5rem; }
        .page-subtitle { color: var(--text-secondary); font-size: 1.1rem; margin-bottom: 2rem; }
        .articles-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 1.5rem; margin-top: 2rem; }
        .article-card { background: var(--card-background); border-radius: var(--border-radius); overflow: hidden; box-shadow: var(--shadow-light); transition: var(--transition); border: 1px solid var(--border-color); }
        .article-card:hover { transform: translateY(-4px); box-shadow: var(--shadow-medium); }
        .article-image { width: 100%; height: 200px; object-fit: cover; background: linear-gradient(135deg, var(--secondary-color), var(--accent-color)); transition: var(--transition); }
        .article-image:hover { transform: scale(1.05); }
        .article-content { padding: 1.5rem; }
        .article-date { color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.5rem; display: flex; align-items: center; }
        .article-date i { margin-right: 0.5rem; }
        .article-title { font-size: 1.3rem; font-weight: 600; color: var(--primary-color); margin-bottom: 0.75rem; line-height: 1.4; }
        .article-description { color: var(--text-secondary); font-size: 0.95rem; line-height: 1.6; margin-bottom: 1rem; }
        .article-tags { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
        .tag { display: inline-block; padding: 0.25rem 0.75rem; background: linear-gradient(135deg, var(--secondary-color), #2980b9); color: white; border-radius: 20px; font-size: 0.8rem; font-weight: 500; transition: var(--transition); }
        .tag:hover { transform: scale(1.05); }
        .read-more { display: inline-flex; align-items: center; color: var(--secondary-color); text-decoration: none; font-weight: 500; transition: var(--transition); }
        .read-more:hover { color: var(--accent-color); transform: translateX(3px); }
        .read-more i { margin-left: 0.5rem; transition: var(--transition); }
        .read-more:hover i { transform: translateX(3px); }
        .mobile-toggle { display: none; position: fixed; top: 1rem; left: 1rem; z-index: 1001; background: var(--primary-color); color: white; border: none; border-radius: 8px; padding: 0.75rem; cursor: pointer; box-shadow: var(--shadow-medium); }

        /* Special Paper Review Card */
        .special-card { background: linear-gradient(135deg, var(--primary-color), #34495e); color: white; border: none; position: relative; overflow: hidden; }
        .special-card::before { content: ''; position: absolute; top: -50%; right: -50%; width: 100%; height: 100%; background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%); transform: rotate(45deg); }
        .special-card .special-icon { position: absolute; top: 1rem; right: 1rem; font-size: 3rem; color: rgba(255, 255, 255, 0.2); z-index: 1; }
        .special-card .article-content { position: relative; z-index: 2; }
        .special-card .article-date { color: rgba(255, 255, 255, 0.8); }
        .special-card .article-title { color: white; font-size: 1.4rem; }
        .special-card .article-description { color: rgba(255, 255, 255, 0.9); }
        .special-card .disclaimer { color: rgba(255, 255, 255, 0.7); font-size: 0.8rem; margin: 0.5rem 0; padding: 0.5rem; background: rgba(0, 0, 0, 0.2); border-radius: 6px; }
        .special-card .tag { background: rgba(255, 255, 255, 0.2); color: white; border: 1px solid rgba(255, 255, 255, 0.3); }
        .special-card .read-more { color: white; font-weight: 600; background: rgba(255, 255, 255, 0.1); padding: 0.5rem 1rem; border-radius: 25px; border: 1px solid rgba(255, 255, 255, 0.3); transition: var(--transition); }
        .special-card .read-more:hover { background: rgba(255, 255, 255, 0.2); transform: translateX(5px); }

        @media (max-width: 768px) { 
            .mobile-toggle { display: block; } 
            .sidebar { transform: translateX(-100%); } 
            .sidebar.active { transform: translateX(0); } 
            .main-content { margin-left: 0; } 
            .content-area { padding: 1rem; margin-top: 4rem; } 
            .hero-logo { font-size: 2.5rem; } 
            .page-title { font-size: 2rem; } 
            .articles-grid { grid-template-columns: 1fr; gap: 1rem; } 
        }
        .sidebar::-webkit-scrollbar { width: 4px; }
        .sidebar::-webkit-scrollbar-track { background: rgba(255, 255, 255, 0.1); }
        .sidebar::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.3); border-radius: 2px; }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .article-card { animation: fadeInUp 0.6s ease-out; }
    </style>
</head>
<body>
    <button class="mobile-toggle" onclick="toggleSidebar()"><i class="fas fa-bars"></i></button>
    <aside class="sidebar" id="sidebar">
        <div class="sidebar-header">
            <div class="avatar">{{AVATAR_TEXT}}</div>
            <h1>{{AUTHOR}}</h1>
            <p class="tagline">{{BLOG_SUBTITLE}}</p>
        </div>
        <nav class="nav-menu">
            <a href="index.html" class="nav-item active"><i class="fas fa-home"></i>Home</a>
            <a href="papers.html" class="nav-item"><i class="fas fa-graduation-cap"></i>Papers</a>
            <a href="reviews.html" class="nav-item"><i class="fas fa-search-plus"></i>Reviews</a>
            <a href="opensource.html" class="nav-item"><i class="fas fa-code-branch"></i>Open-Source</a>
            <a href="products.html" class="nav-item"><i class="fas fa-box-open"></i>Products</a>
        </nav>
        <div class="contact-info">
            <h4><i class="fas fa-graduation-cap"></i> Academic Info</h4>
            <p>{{UNIVERSITY}}<br>{{FIELDS}}</p>
        </div>
        <div class="social-links">
            <h3>Contact</h3>
            <div class="social-grid">
                <a href="{{GITHUB_URL}}" class="social-link" target="_blank"><i class="fab fa-github"></i>GitHub</a>
                <a href="mailto:{{EMAIL_SCHOOL}}" class="social-link"><i class="fas fa-envelope"></i>School Email</a>
                <a href="mailto:{{EMAIL_PERSONAL}}" class="social-link"><i class="fas fa-envelope-open"></i>Personal Email</a>
                <a href="{{BILIBILI_URL}}" class="social-link" target="_blank"><i class="fab fa-bilibili"></i>Bilibili</a>
                <a href="{{ZHIHU_URL}}" class="social-link" target="_blank"><i class="fas fa-book"></i>Zhihu</a>
                <a href="{{TWITTER_URL}}" class="social-link" target="_blank"><i class="fab fa-twitter"></i>Twitter</a>
            </div>
        </div>
    </aside>
    <main class="main-content">
        <header class="header">
            <div class="header-content">
                <div><h2>{{BLOG_TITLE}}</h2></div>
                <div class="search-box">
                    <input type="text" class="search-input" placeholder="Search articles..." id="searchInput">
                    <i class="fas fa-search search-icon"></i>
                </div>
            </div>
        </header>
        <div class="content-area">
            <div class="hero-section">
                <img src="{{HEADER_IMAGE}}" alt="Hero Background" class="hero-bg-image">
                <div class="hero-content">
                    <div class="hero-logo">Love is Simple, Simple is Love</div>
                    <h1 class="page-title">Welcome to My Tech World</h1>
                    <p class="page-subtitle">{{DESCRIPTION}}</p>
                </div>
            </div>
            <div class="articles-grid" id="articlesContainer">{{ARTICLES_HTML}}</div>
        </div>
    </main>
    <script>
        document.getElementById('searchInput').addEventListener('input', function(e) {
            const query = e.target.value.toLowerCase();
            const articles = document.querySelectorAll('.article-card:not(.special-card)');
            articles.forEach(article => {
                const title = article.querySelector('.article-title').textContent.toLowerCase();
                const description = article.querySelector('.article-description').textContent.toLowerCase();
                const tags = Array.from(article.querySelectorAll('.tag')).map(tag => tag.textContent.toLowerCase());
                const matches = title.includes(query) || description.includes(query) || tags.some(tag => tag.includes(query));
                article.style.display = matches ? 'block' : 'none';
            });
        });
        function toggleSidebar() { document.getElementById('sidebar').classList.toggle('active'); }
        document.addEventListener('click', (e) => {
            if (window.innerWidth <= 768) {
                const sidebar = document.getElementById('sidebar');
                const toggle = document.querySelector('.mobile-toggle');
                if (!sidebar.contains(e.target) && !toggle.contains(e.target)) { sidebar.classList.remove('active'); }
            }
        });
    </script>
</body>
</html>'''

# ----------------------------------------------------------------------------
# 文章模板 (article_template) - 无需修改
# ----------------------------------------------------------------------------
article_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}} - {{BLOG_TITLE}}</title>
    <meta name="description" content="{{DESCRIPTION}}">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github.min.css" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --primary-color: #2c3e50; --secondary-color: #3498db; --accent-color: #e74c3c;
            --background-color: #f8fafc; --card-background: #ffffff; --text-primary: #2d3748;
            --text-secondary: #718096; --border-color: #e2e8f0; --shadow-light: 0 4px 6px rgba(0, 0, 0, 0.07);
            --shadow-medium: 0 10px 25px rgba(0, 0, 0, 0.1); --border-radius: 12px; --transition: all 0.3s ease;
        }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background-color: var(--background-color); color: var(--text-primary); line-height: 1.7; }
        .navbar { background: var(--card-background); box-shadow: var(--shadow-light); padding: 1rem 0; position: sticky; top: 0; z-index: 100; border-bottom: 1px solid var(--border-color); }
        .nav-container { max-width: 1400px; margin: 0 auto; padding: 0 2rem; display: flex; justify-content: space-between; align-items: center; }
        .nav-brand { display: flex; align-items: center; text-decoration: none; color: var(--primary-color); font-weight: 600; font-size: 1.2rem; }
        .nav-brand i { margin-right: 0.5rem; color: var(--secondary-color); }
        .nav-links { display: flex; gap: 1.5rem; }
        .nav-link { color: var(--text-secondary); text-decoration: none; transition: var(--transition); padding: 0.5rem 1rem; border-radius: 8px; }
        .nav-link:hover { color: var(--secondary-color); background-color: rgba(52, 152, 219, 0.1); }
        .main-container { display: flex; max-width: 1400px; margin: 0 auto; gap: 2rem; padding: 2rem; }
        .toc-sidebar { width: 250px; flex-shrink: 0; position: sticky; top: 6rem; height: fit-content; max-height: calc(100vh - 8rem); overflow-y: auto; }
        .toc-container { background: var(--card-background); border: 1px solid var(--border-color); border-radius: var(--border-radius); padding: 1.5rem; }
        .toc-title { font-size: 1.1rem; font-weight: 600; color: var(--primary-color); margin-bottom: 1rem; display: flex; align-items: center; }
        .toc-title i { margin-right: 0.5rem; color: var(--secondary-color); }
        .toc ul { list-style: none; padding-left: 0; }
        .toc ul ul { padding-left: 1rem; margin-top: 0.3rem; }
        .toc li { margin: 0.3rem 0; }
        .toc a { color: var(--text-secondary); text-decoration: none; transition: var(--transition); display: block; padding: 0.3rem 0.5rem; border-radius: 4px; font-size: 0.9rem; line-height: 1.4; }
        .toc a:hover, .toc a.active { color: var(--secondary-color); background-color: rgba(52, 152, 219, 0.1); }
        .article-main { flex: 1; min-width: 0; }
        .article-header { background: var(--card-background); border-radius: var(--border-radius); padding: 2rem; margin-bottom: 2rem; box-shadow: var(--shadow-light); border: 1px solid var(--border-color); position: relative; overflow: hidden; }
        .article-cover { width: 100%; height: 350px; object-fit: cover; border-radius: var(--border-radius); margin-bottom: 1.5rem; box-shadow: var(--shadow-medium); transition: var(--transition); }
        .article-cover:hover { transform: scale(1.02); }
        .article-header::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 6px; background: linear-gradient(90deg, var(--secondary-color), var(--accent-color)); }
        .article-title { font-size: 2.5rem; font-weight: 700; color: var(--primary-color); margin-bottom: 1rem; line-height: 1.3; }
        .article-meta { display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.5rem; color: var(--text-secondary); font-size: 0.95rem; }
        .meta-item { display: flex; align-items: center; background: var(--background-color); padding: 0.5rem 1rem; border-radius: 20px; }
        .meta-item i { margin-right: 0.5rem; color: var(--secondary-color); }
        .article-tags { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
        .tag { background: linear-gradient(135deg, var(--secondary-color), #2980b9); color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem; font-weight: 500; box-shadow: var(--shadow-light); }
        .article-description { color: var(--text-secondary); font-size: 1.1rem; line-height: 1.6; font-style: italic; background: var(--background-color); padding: 1rem; border-radius: 8px; border-left: 4px solid var(--secondary-color); }
        .article-content { background: var(--card-background); border-radius: var(--border-radius); padding: 2.5rem; box-shadow: var(--shadow-light); border: 1px solid var(--border-color); }
        .article-content h1, .article-content h2, .article-content h3, .article-content h4, .article-content h5, .article-content h6 { color: var(--primary-color); margin: 2rem 0 1rem 0; font-weight: 600; line-height: 1.4; }
        .article-content h1 { font-size: 2.2rem; border-bottom: 3px solid var(--secondary-color); padding-bottom: 0.5rem; }
        .article-content h2 { font-size: 1.8rem; border-bottom: 2px solid var(--border-color); padding-bottom: 0.3rem; }
        .article-content p { margin: 1rem 0; line-height: 1.8; }
        .article-content img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: var(--shadow-light); margin: 1.5rem 0; }
        .article-content blockquote { border-left: 4px solid var(--secondary-color); margin: 1.5rem 0; padding: 1rem 1.5rem; background: var(--background-color); border-radius: 0 8px 8px 0; font-style: italic; }
        .article-content pre { background: #f6f8fa; border: 1px solid var(--border-color); border-radius: 8px; padding: 1.5rem; overflow-x: auto; margin: 1.5rem 0; }
        .article-content code { background: #f6f8fa; border: 1px solid var(--border-color); border-radius: 4px; padding: 0.2rem 0.4rem; font-family: 'Monaco', 'Consolas', monospace; font-size: 0.9em; }
        .article-content pre code { background: none; border: none; padding: 0; }
        .article-content table { width: 100%; border-collapse: collapse; margin: 1.5rem 0; border-radius: 8px; overflow: hidden; box-shadow: var(--shadow-light); }
        .article-content th, .article-content td { padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border-color); }
        .article-content th { background: var(--background-color); font-weight: 600; color: var(--primary-color); }
        .contact-card { background: linear-gradient(135deg, var(--primary-color), #34495e); color: white; border-radius: var(--border-radius); padding: 2rem; margin: 2rem 0; text-align: center; }
        .contact-card h3 { margin-bottom: 1rem; color: white; }
        .contact-links { display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; }
        .contact-link { display: inline-flex; align-items: center; padding: 0.5rem 1rem; background: rgba(255, 255, 255, 0.1); border-radius: 25px; color: white; text-decoration: none; transition: var(--transition); }
        .contact-link:hover { background: rgba(255, 255, 255, 0.2); transform: translateY(-1px); }
        .contact-link i { margin-right: 0.5rem; }
        .back-button { position: fixed; bottom: 2rem; right: 2rem; background: var(--secondary-color); color: white; border: none; border-radius: 50%; width: 60px; height: 60px; display: flex; align-items: center; justify-content: center; cursor: pointer; box-shadow: var(--shadow-medium); transition: var(--transition); text-decoration: none; }
        .back-button:hover { background: #2980b9; transform: translateY(-2px); }
        .translation-links { 
            margin: 1.5rem 0 0.5rem 0; padding: 1rem; background: var(--background-color); 
            border-radius: 8px; border: 1px solid var(--border-color); font-size: 0.95rem; 
        }
        .translation-links span { color: var(--text-secondary); margin-right: 0.5rem; }
        .translation-links a { 
            color: var(--secondary-color); text-decoration: none; font-weight: 500; 
            padding: 0.3rem 0.6rem; border-radius: 6px; transition: var(--transition); 
            margin: 0 0.25rem;
        }
        .translation-links a:hover { background-color: rgba(52, 152, 219, 0.1); }
        .translation-links a.active { 
            background-color: var(--secondary-color); color: white; 
            pointer-events: none; 
        }
        @media (max-width: 1024px) { .main-container { flex-direction: column; padding: 1rem; } .toc-sidebar { width: 100%; position: static; order: 2; margin-top: 2rem; } .article-main { order: 1; } }
        @media (max-width: 768px) { .nav-container { padding: 0 1rem; flex-direction: column; gap: 1rem; } .article-header, .article-content { padding: 1.5rem; } .article-title { font-size: 2rem; } .back-button { bottom: 1rem; right: 1rem; width: 50px; height: 50px; } .article-cover { height: 250px; } }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="nav-container">
            <a href="../index.html" class="nav-brand">
                <i class="fas fa-home"></i>{{BLOG_TITLE}}
            </a>
            <div class="nav-links">
                <a href="../index.html" class="nav-link"><i class="fas fa-arrow-left"></i>Back to Home</a>
                <a href="mailto:{{EMAIL}}" class="nav-link"><i class="fas fa-envelope"></i>Contact Me</a>
            </div>
        </div>
    </nav>
    <div class="main-container">
        <aside class="toc-sidebar">{{TOC}}</aside>
        <main class="article-main">
            <header class="article-header">
                <img src="{{COVER_IMAGE}}" alt="{{TITLE}}" class="article-cover" onerror="this.style.display='none'">
                <h1 class="article-title">{{TITLE}}</h1>
                <div class="article-meta">
                    <div class="meta-item"><i class="fas fa-calendar-alt"></i>Published: {{DATE}}</div>
                    <div class="meta-item"><i class="fas fa-user"></i>Author: {{AUTHOR}}</div>
                </div>
                {{TAGS}}
                <div class="article-description">{{DESCRIPTION}}</div>
            </header>
            {{TRANSLATIONS}}
            <article class="article-content">{{CONTENT}}</article>
            <div class="contact-card">
                <h3><i class="fas fa-heart"></i> Thanks for Reading</h3>
                <p>If this article was helpful to you, feel free to connect with me!</p>
                <div class="contact-links">
                    <a href="{{GITHUB_URL}}" class="contact-link" target="_blank"><i class="fab fa-github"></i>GitHub</a>
                    <a href="mailto:{{EMAIL}}" class="contact-link"><i class="fas fa-envelope"></i>Email</a>
                </div>
            </div>
        </main>
    </div>
    <a href="#" class="back-button" onclick="window.scrollTo({top: 0, behavior: 'smooth'}); return false;">
        <i class="fas fa-arrow-up"></i>
    </a>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"></script>
    <script>
        hljs.highlightAll();
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) { target.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
            });
        });
        function highlightCurrentSection() {
            const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
            const tocLinks = document.querySelectorAll('.toc a');
            let current = '';
            headings.forEach(heading => {
                const rect = heading.getBoundingClientRect();
                if (rect.top <= 100) { current = heading.id; }
            });
            tocLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('href') === '#' + current) { link.classList.add('active'); }
            });
        }
        window.addEventListener('scroll', highlightCurrentSection);
        document.addEventListener('DOMContentLoaded', highlightCurrentSection);
    </script>
</body>
</html>'''

# ----------------------------------------------------------------------------
# 特殊页面模板 (papers_template)
# 我们将其通用化，使用 {{PAGE_ICON}} 和 {{PAGE_TITLE_PLURAL}} 作为占位符
# ----------------------------------------------------------------------------
special_page_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{PAGE_TITLE_PLURAL}} - {{BLOG_TITLE}}</title>
    <meta name="description" content="A collection of {{PAGE_TITLE_PLURAL}}">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root { 
            --primary-color: #2c3e50; --secondary-color: #3498db; --accent-color: #e74c3c; 
            --background-color: #f8fafc; --card-background: #ffffff; --text-primary: #2d3748; 
            --text-secondary: #718096; --border-color: #e2e8f0; --shadow-light: 0 4px 6px rgba(0, 0, 0, 0.07); 
            --shadow-medium: 0 10px 25px rgba(0, 0, 0, 0.1); --border-radius: 12px; --transition: all 0.3s ease; 
        }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
            background-color: var(--background-color); color: var(--text-primary); line-height: 1.6; 
        }
        .header { 
            background: linear-gradient(135deg, var(--primary-color), #34495e); 
            color: white; padding: 4rem 0; text-align: center; position: relative; overflow: hidden;
        }
        .header-bg-image { 
            position: absolute; top: 0; left: 0; width: 100%; height: 100%; 
            object-fit: cover; opacity: 0.2; z-index: 1; 
        }
        .header-content { 
            max-width: 1200px; margin: 0 auto; padding: 0 2rem; position: relative; z-index: 2; 
        }
        .header h1 { 
            font-size: 3rem; font-weight: 700; margin-bottom: 0.5rem; 
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .header .subtitle { 
            font-size: 1.2rem; opacity: 0.9; margin-bottom: 1rem; 
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        }
        .nav-breadcrumb { 
            display: flex; justify-content: center; align-items: center; 
            gap: 0.5rem; font-size: 0.9rem; 
        }
        .nav-breadcrumb a { 
            color: rgba(255, 255, 255, 0.8); text-decoration: none; 
            transition: var(--transition); padding: 0.3rem 0.8rem; 
            border-radius: 15px; background: rgba(255, 255, 255, 0.1);
        }
        .nav-breadcrumb a:hover { 
            color: white; background: rgba(255, 255, 255, 0.2); 
        }
        .main-content { max-width: 1200px; margin: 2rem auto; padding: 0 2rem; }

        /* Papers Filter */
        .papers-filter { 
            background: var(--card-background); border-radius: var(--border-radius); 
            padding: 1.5rem; margin-bottom: 2rem; box-shadow: var(--shadow-light); 
            border: 1px solid var(--border-color); 
        }
        .filter-controls { display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; }
        .search-box { position: relative; flex: 1; min-width: 250px; }
        .search-input { 
            width: 100%; padding: 0.75rem 1rem 0.75rem 2.5rem; 
            border: 2px solid var(--border-color); border-radius: 25px; 
            background-color: var(--background-color); font-size: 0.9rem; 
            transition: var(--transition); 
        }
        .search-input:focus { 
            outline: none; border-color: var(--secondary-color); 
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1); 
        }
        .search-icon { 
            position: absolute; left: 1rem; top: 50%; 
            transform: translateY(-50%); color: var(--text-secondary); 
        }
        .filter-button { 
            padding: 0.5rem 1rem; border: 2px solid var(--border-color); 
            background: white; border-radius: 20px; cursor: pointer; 
            transition: var(--transition); text-decoration: none; 
            color: var(--text-secondary); font-size: 0.9rem; 
        }
        .filter-button:hover, .filter-button.active { 
            border-color: var(--secondary-color); 
            background: var(--secondary-color); color: white; 
        }

        /* Papers Grid */
        .papers-grid { 
            display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); 
            gap: 1.5rem; 
        }
        .paper-card { 
            background: var(--card-background); border-radius: var(--border-radius); 
            overflow: hidden; box-shadow: var(--shadow-light); 
            transition: var(--transition); border: 1px solid var(--border-color); 
        }
        .paper-card:hover { transform: translateY(-4px); box-shadow: var(--shadow-medium); }
        .paper-image { 
            width: 100%; height: 200px; object-fit: cover; 
            background: linear-gradient(135deg, var(--secondary-color), var(--accent-color)); 
            transition: var(--transition);
        }
        .paper-image:hover { transform: scale(1.05); }
        .paper-content { padding: 1.5rem; }
        .paper-meta { 
            display: flex; flex-wrap: wrap; gap: 0.5rem; 
            margin-bottom: 1rem; font-size: 0.85rem; 
        }
        .meta-item { 
            display: flex; align-items: center; color: var(--text-secondary); 
            background: var(--background-color); padding: 0.25rem 0.5rem; 
            border-radius: 15px; 
        }
        .meta-item i { margin-right: 0.3rem; color: var(--secondary-color); }
        .paper-title { 
            font-size: 1.3rem; font-weight: 600; color: var(--primary-color); 
            margin-bottom: 0.75rem; line-height: 1.4; 
        }
        .paper-description { 
            color: var(--text-secondary); font-size: 0.95rem; 
            line-height: 1.6; margin-bottom: 1rem; 
        }
        .paper-tags { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
        .tag { 
            display: inline-block; padding: 0.25rem 0.75rem; 
            background: linear-gradient(135deg, var(--secondary-color), #2980b9); 
            color: white; border-radius: 20px; font-size: 0.8rem; font-weight: 500; 
            transition: var(--transition); 
        }
        .tag:hover { transform: scale(1.05); }
        .read-more { 
            display: inline-flex; align-items: center; color: var(--secondary-color); 
            text-decoration: none; font-weight: 500; transition: var(--transition); 
        }
        .read-more:hover { color: var(--accent-color); transform: translateX(3px); }
        .read-more i { margin-left: 0.5rem; transition: var(--transition); }
        .read-more:hover i { transform: translateX(3px); }

        /* No papers state */
        .no-papers { 
            text-align: center; padding: 4rem 2rem; 
            background: var(--card-background); border-radius: var(--border-radius); 
            box-shadow: var(--shadow-light); border: 1px solid var(--border-color); 
        }
        .no-papers i { 
            font-size: 4rem; color: var(--secondary-color); margin-bottom: 1rem; 
        }
        .no-papers h3 { 
            font-size: 1.5rem; color: var(--primary-color); margin-bottom: 0.5rem; 
        }
        .no-papers p { color: var(--text-secondary); font-size: 1.1rem; }

        /* Stats */
        .papers-stats { 
            display: flex; justify-content: center; gap: 2rem; 
            margin-bottom: 2rem; flex-wrap: wrap; 
        }
        .stat-item { 
            text-align: center; background: var(--card-background); 
            padding: 1rem 1.5rem; border-radius: var(--border-radius); 
            box-shadow: var(--shadow-light); border: 1px solid var(--border-color); 
        }
        .stat-number { 
            font-size: 2rem; font-weight: 700; 
            color: var(--secondary-color); display: block; 
        }
        .stat-label { color: var(--text-secondary); font-size: 0.9rem; }

        @media (max-width: 768px) { 
            .header h1 { font-size: 2rem; } 
            .main-content { padding: 0 1rem; } 
            .papers-grid { grid-template-columns: 1fr; gap: 1rem; } 
            .filter-controls { flex-direction: column; align-items: stretch; }
            .papers-stats { flex-direction: column; gap: 1rem; }
        }
    </style>
</head>
<body>
    <header class="header">
        <img src="{{HEADER_IMAGE}}" alt="{{PAGE_TITLE_PLURAL}} Header Background" class="header-bg-image">
        <div class="header-content">
            <h1><i class="{{PAGE_ICON}}"></i> {{PAGE_TITLE_PLURAL}}</h1>
            <p class="subtitle">A collection of {{PAGE_TITLE_PLURAL}}</p>
            <nav class="nav-breadcrumb">
                <a href="index.html"><i class="fas fa-home"></i> Home</a>
                <i class="fas fa-chevron-right"></i>
                <span>{{PAGE_TITLE_PLURAL}}</span>
            </nav>
        </div>
    </header>

    <main class="main-content">
        <div class="papers-filter">
            <div class="filter-controls">
                <div class="search-box">
                    <input type="text" class="search-input" placeholder="Search {{PAGE_TITLE_PLURAL}} by title or tags..." id="searchInput">
                    <i class="fas fa-search search-icon"></i>
                </div>
            </div>
        </div>

        <div class="papers-stats" id="papersStats"></div>

        <div class="papers-grid" id="papersContainer">
            {{PAPERS_HTML}}
        </div>
    </main>

    <script>
        // 搜索功能
        document.getElementById('searchInput').addEventListener('input', function(e) {
            const query = e.target.value.toLowerCase();
            const papers = document.querySelectorAll('.paper-card');

            papers.forEach(paper => {
                const title = paper.querySelector('.paper-title').textContent.toLowerCase();
                const description = paper.querySelector('.paper-description').textContent.toLowerCase();
                const tags = Array.from(paper.querySelectorAll('.tag')).map(tag => tag.textContent.toLowerCase());

                const matches = title.includes(query) || 
                               description.includes(query) || 
                               tags.some(tag => tag.includes(query));

                paper.style.display = matches ? 'block' : 'none';
            });

            updateStats();
        });

        // 简单的分类过滤 (可以扩展)
        document.querySelectorAll('.filter-button').forEach(button => {
            button.addEventListener('click', function(e) { e.preventDefault(); });
        });

        // 更新统计信息
        function updateStats() {
            const visiblePapers = document.querySelectorAll('.paper-card:not([style*="display: none"])');
            const statsContainer = document.getElementById('papersStats');

            const totalTags = new Set(Array.from(visiblePapers).flatMap(paper => 
                Array.from(paper.querySelectorAll('.tag')).map(tag => tag.textContent)
            )).size;

            if (visiblePapers.length > 0) {
                statsContainer.innerHTML = `
                    <div class="stat-item">
                        <span class="stat-number">${visiblePapers.length}</span>
                        <span class="stat-label">Items</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-number">${totalTags}</span>
                        <span class="stat-label">Topics</span>
                    </div>
                `;
            } else {
                statsContainer.innerHTML = '';
            }
        }

        // 初始化统计
        document.addEventListener('DOMContentLoaded', updateStats);
    </script>
</body>
</html>'''


# ----------------------------------------------------------------------------
# 模板创建函数
# ----------------------------------------------------------------------------
def create_templates():
    """创建模板文件"""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)

    # ------------------------------------
    # 保存文章模板
    # ------------------------------------
    with open(templates_dir / "article_template.html", 'w', encoding='utf-8') as f:
        f.write(article_template)
    print("✅ Created article_template.html")

    # ------------------------------------
    # 保存主页模板
    # ------------------------------------
    with open(templates_dir / "index_template.html", 'w', encoding='utf-8') as f:
        f.write(index_template)
    print("✅ Created index_template.html")

    # ------------------------------------
    # 通用函数创建特殊页面模板
    # ------------------------------------
    def create_special_page(name: str, title: str, icon: str):
        """
        使用通用模板创建特定的页面模板文件
        :param name: 文件名 (e.g., "papers")
        :param title: 页面标题 (e.g., "Paper Reviews")
        :param icon: FontAwesome 图标 (e.g., "fas fa-graduation-cap")
        """
        template_content = special_page_template.replace(
            '{{PAGE_TITLE_PLURAL}}', title
        ).replace(
            '{{PAGE_ICON}}', icon
        )
        # 移除原版中特定于 "papers" 的过滤器按钮 (简化)
        import re
        template_content = re.sub(
            r'<a href="#" class="filter-button".*?data-category.*?</a>',
            '',
            template_content
        )

        with open(templates_dir / f"{name}_template.html", 'w', encoding='utf-8') as f:
            f.write(template_content)
        print(f"✅ Created {name}_template.html")

    # ------------------------------------
    # 批量创建所有特殊页面
    # ------------------------------------
    special_pages = [
        {"name": "papers", "title": "Paper Reviews", "icon": "fas fa-graduation-cap"},
        {"name": "reviews", "title": "In-Depth Reviews", "icon": "fas fa-search-plus"},
        {"name": "opensource", "title": "Open-Source", "icon": "fas fa-code-branch"},
        {"name": "products", "title": "Products", "icon": "fas fa-box-open"},
    ]

    for page in special_pages:
        create_special_page(page["name"], page["title"], page["icon"])

    print(f"✅ All templates created in {templates_dir}/")


def main():
    """主函数"""
    print("🚀 Creating blog templates...")
    create_templates()
    print("🎉 Template creation completed!")
    print("\nNext steps:")
    print("1. Put some images in the ./images directory")
    print("2. (If you haven't) Create dirs: mkdir reviews_notes opensource_notes products_notes")
    print("3. Run: python3 blog_generator.py")
    print("4. Your blog with all new sections will be generated!")


if __name__ == "__main__":
    main()