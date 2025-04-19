#!/bin/bash
# Interactive Git script for managing GitHub pushes for the ai-learning-connect4 project
# Run from the root of your Git repository

# Ensure we're in a Git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "Error: This directory is not a Git repository. Please navigate to your project root."
    exit 1
fi

# Check if we're in the correct repository
repo_url=$(git remote -v | grep fetch | awk '{print $2}')
if ! echo "$repo_url" | grep -q "ai-learning-connect4"; then
    echo "Warning: This script is intended for the ai-learning-connect4 repository."
    echo "Current repository: $repo_url"
    read -p "Continue anyway? [Y/n]: " continue_anyway
    continue_anyway=${continue_anyway:-Y}
    if [ "$continue_anyway" != "y" ] && [ "$continue_anyway" != "Y" ]; then
        echo "Exiting."
        exit 1
    fi
fi

# Ensure .gitignore includes models/*, stats/*, and __pycache__/
if ! grep -Fx "models/*" .gitignore >/dev/null 2>&1 || ! grep -Fx "stats/*" .gitignore >/dev/null 2>&1 || ! grep -Fx "__pycache__/" .gitignore >/dev/null 2>&1; then
    echo "Updating .gitignore to include models/*, stats/*, and __pycache__/"
    echo -e "models/*\nstats/*\n__pycache__/" >> .gitignore
    # Stage .gitignore if it was modified
    git add .gitignore
fi
# Ensure empty directories are tracked
mkdir -p models stats
touch models/.gitkeep stats/.gitkeep
git add models/.gitkeep stats/.gitkeep

echo "=== GitHub Interaction Script for AI Learning Connect4 ==="
echo "Current directory: $(pwd)"
echo "Repository: $repo_url"
echo "Current branch: $(git branch --show-current)"
echo ""

# Step 1: Show current Git status
echo "=== Checking Git Status ==="
git status
echo ""

# Step 2: Stage changes interactively
echo "=== Staging Changes ==="
read -p "Do you want to stage all changes? [Y/n]: " stage_all
stage_all=${stage_all:-Y}
if [ "$stage_all" = "y" ] || [ "$stage_all" = "Y" ]; then
    echo "Staging all changes..."
    git add .
    echo "Changes staged:"
    git status --short
else
    echo "Enter the files you want to stage (space-separated, e.g., 'file1.py file2.txt'), or press Enter to skip:"
    read -r files_to_stage
    if [ -n "$files_to_stage" ]; then
        echo "Staging: $files_to_stage"
        git add $files_to_stage
        echo "Changes staged:"
        git status --short
    else
        echo "No files staged."
    fi
fi
echo ""

# Step 3: Check if there's anything to commit
if git diff --cached --quiet; then
    echo "Nothing staged to commit. Exiting."
    exit 0
fi

# Step 4: Commit changes
echo "=== Committing Changes ==="
echo "Current staged changes:"
git diff --cached --name-only
echo ""
read -p "Enter your commit message: " commit_message
if [ -z "$commit_message" ]; then
    echo "Commit message cannot be empty. Using default message."
    commit_message="Update for AI Learning Connect4 project"
fi
echo "Committing with message: '$commit_message'"
git commit -m "$commit_message"
echo "Commit complete:"
git log -1 --oneline
echo ""

# Step 5: Push to GitHub
echo "=== Pushing to GitHub ==="
current_branch=$(git branch --show-current)

# Check if the current branch has an upstream branch
if ! git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
    echo "The current branch '$current_branch' has no upstream branch set."
    read -p "Set upstream to 'origin/$current_branch' and push? [Y/n]: " set_upstream
    set_upstream=${set_upstream:-Y}
    if [ "$set_upstream" = "y" ] || [ "$set_upstream" = "Y" ]; then
        echo "Setting upstream and pushing to 'origin/$current_branch'..."
        git push --set-upstream origin "$current_branch"
    else
        echo "Push skipped. Your changes are committed locally but not on GitHub."
        exit 0
    fi
else
    read -p "Push to 'origin/$current_branch'? [Y/n]: " push_confirm
    push_confirm=${push_confirm:-Y}
    if [ "$push_confirm" = "y" ] || [ "$push_confirm" = "Y" ]; then
        echo "Pushing to 'origin/$current_branch'..."
        git push origin "$current_branch"
    else
        echo "Push skipped. Your changes are committed locally but not on GitHub."
        exit 0
    fi
fi

if [ $? -eq 0 ]; then
    echo "Successfully pushed to GitHub!"
else
    echo "Push failed. Check your connection or credentials."
    exit 1
fi
echo ""

# Step 6: Final status check
echo "=== Final Check ==="
read -p "Would you like to see the current Git status? [Y/n]: " status_confirm
status_confirm=${status_confirm:-Y}
if [ "$status_confirm" = "y" ] || [ "$status_confirm" = "Y" ]; then
    echo "Running 'git status'..."
    git status
else
    echo "Skipping status check."
fi
echo ""
echo "=== Done! ==="