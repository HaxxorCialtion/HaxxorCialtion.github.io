# v2 — 极简简历门户

## 文件结构

```
v2/
├── index.html              # 主页(中英双语)
├── assets/
│   ├── style.css           # 样式
│   ├── i18n.js             # 英文字典(中文写在 index.html 里)
│   ├── app.js              # 语言切换 / PDF 链接 / 二维码探测
│   ├── resume_zh.pdf       # ← 把 ../cv/resume_mihoyo.pdf 拷过来重命名
│   ├── resume_en.pdf       # ← 把 ../cv/cv_en/resume_mihoyo_en.pdf 拷过来重命名
│   └── wechat.png          # (可选)微信二维码,放了就自动显示
└── README.md
```

## 接入到 GitHub Pages 的步骤

把 `v2/` 下的文件覆盖到仓库根目录(原来那一堆 `blog_generator.py` / `posts/` 等先备份在 `legacy/` 之类的目录,门户先上极简版,博客后面再筛进来)。

```bash
# 在 HaxxorCialtion.github.io 仓库根目录执行
mkdir -p legacy
# 把所有旧文件搬进 legacy(留一份 v2 之外的 README.md 不动)
shopt -s extglob
mv !(legacy|v2|.git|.gitignore|README.md) legacy/ 2>/dev/null

# 把 v2 内容上提到根目录
mv v2/* v2/.* . 2>/dev/null
rmdir v2

# 拷贝两份 PDF
cp ../cv/resume_mihoyo.pdf       ./assets/resume_zh.pdf
cp ../cv/cv_en/resume_mihoyo_en.pdf ./assets/resume_en.pdf

# (可选)微信二维码
# cp /path/to/wechat.png ./assets/wechat.png

git add .
git commit -m "v2: minimal resume portal"
git push
```

## 本地预览

```bash
cd /Users/cialtionshi/Desktop/blog/github_io   # 或 v2 当前所在目录
python3 -m http.server 8000
# 浏览器打开 http://localhost:8000
```

## 改内容

- **中文文案**:直接改 `index.html` 里 `data-i18n="..."` 节点的内容
- **英文文案**:改 `assets/i18n.js` 里对应 key 的字符串
- 两边 key 必须一一对应,新增节点时记得两边都补
- 颜色主调在 `assets/style.css` 顶部 `--blue` 那几个 token 改一下就行

## 后面把博客筛进来时

加新章节(例如"博文 / 论文笔记")的方法:在 `index.html` 里加一个 `<section>`,沿用 `.section / .section-tag` 样式;如果是独立的子页面,新建 `posts/index.html` 再在导航里加一项 `<a class="nav-link" href="./posts/">博文</a>` 即可。
