#!/usr/bin/env bash
# One-time: copy Claude Code project settings and slash commands into .claude/
set -e
cd "$(dirname "$0")/.."
mkdir -p .claude/commands
cp claude-setup/settings.json .claude/settings.json
cp claude-setup/commands/*.md .claude/commands/
echo "Installed .claude/settings.json and commands: /measure /task /tag-version"
