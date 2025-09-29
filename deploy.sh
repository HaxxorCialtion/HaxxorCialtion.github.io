#!/bin/bash
# 博客自动化部署脚本

echo "🚀 开始部署博客..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装，请先安装Python3"
    exit 1
fi

# 安装依赖
echo "📦 安装Python依赖..."
pip3 install -r requirements.txt

# 生成博客
echo "🔨 生成博客页面..."
python3 blog_generator.py

# 检查是否生成成功
if [ $? -eq 0 ]; then
    echo "✅ 博客生成成功"
else
    echo "❌ 博客生成失败"
    exit 1
fi

# Git操作
if [ -d ".git" ]; then
    echo "📤 推送到Git仓库..."
    
    # 添加所有文件
    git add .
    
    # 提交更改
    commit_message="Auto update blog - $(date '+%Y-%m-%d %H:%M:%S')"
    git commit -m "$commit_message"
    
    # 推送到远程仓库
    git push origin main
    
    if [ $? -eq 0 ]; then
        echo "✅ 成功推送到GitHub"
        echo "🌐 你的博客将在几分钟后更新"
    else
        echo "❌ 推送失败，请检查Git配置"
    fi
else
    echo "⚠️  未检测到Git仓库，跳过Git操作"
fi

echo "🎉 部署完成！"
