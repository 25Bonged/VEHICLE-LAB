#!/bin/bash
# Git Repository Setup Script for Vehicle Lab Backend

cd "$(dirname "$0")"

echo "🔧 Vehicle Lab - Git Repository Setup"
echo "======================================"
echo ""

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed."
    echo "   Please install Xcode Command Line Tools:"
    echo "   xcode-select --install"
    exit 1
fi

echo "✅ Git version: $(git --version)"
echo ""

# Check if already a git repo
if [ -d ".git" ]; then
    echo "⚠️  Git repository already initialized"
    git status --short | head -5
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Initialize repository
echo "📦 Initializing git repository..."
git init

# Add remote (update if needed)
echo "🔗 Adding remote repository..."
git remote add origin https://github.com/25Bonged/VEHICLE-LAB.git 2>/dev/null || \
    git remote set-url origin https://github.com/25Bonged/VEHICLE-LAB.git

# Show remote
echo "📍 Remote repository:"
git remote -v

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Review files: git status"
echo "  2. Add files: git add ."
echo "  3. Commit: git commit -m 'Initial commit'"
echo "  4. Push: git push -u origin main"
