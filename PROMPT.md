@prd.md @activity.md

This file defines the canonical Ralph Wiggum workflow.
Do not customize this workflow for a specific repo, framework, or command set.

## Workflow

1. Read `activity.md` first to understand the current state.
2. Read `prd.md` and follow its agent instructions exactly.
3. Find the single highest-priority task where `"passes": false`.
4. Implement exactly one task per iteration.
5. Run the verification required by the PRD and the repo's existing command documentation.
6. Log progress in `activity.md`, including what changed, what commands ran, and the results.
7. Update only that task's `"passes"` field to `true` once the task is verified complete.
8. Create one commit for that task only.
9. When all tasks have `"passes": true`, output `<promise>COMPLETE</promise>`.

## Rules

- Keep this prompt workflow-oriented, not repo-specific.
- Do not add start commands, build commands, or project-specific verification sequences here.
- Do not implement more than one task per iteration.
- Do not rewrite task definitions while executing them.
- If the repo's commands are unclear, discover them from the repo itself without changing this workflow.
