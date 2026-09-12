---
name: jules-runner
description: Delegate large or background coding tasks to Jules (Google's autonomous coding agent) via Jules CLI, and manage remote sessions.
---

# Jules Runner Skill

## Goal
Delegate code authoring, test generation, or multi-step tasks to Google Jules asynchronously and pull the changes into the workspace.

## Instructions
1. Check available sessions or repos if context is needed:
   `jules remote list --session`
2. Start a new Jules task:
   `jules remote new --session "<task_prompt>"`
   (If running multiple variations, use `--parallel <number>`)
3. When the session finishes, pull the diff into the local branch:
   `jules remote pull --session <session_id>`

## Constraints
- Ensure user has logged in (`jules login`) before executing session tasks.
- Always review pulled diffs before proceeding to git commit.