# Git Commit Workflow for AI Assistants

## Goals

- Git history should read as a **story**: small steps, one clear intent per commit.
- **Default:** one file per commit (smallest practical unit).
- **Exception:** several files in **one** commit only when they share the **same logical entity / same story** (incomplete or misleading if split). This should be **unusual**—always prefer smaller commits when in doubt.

## When to combine files (exception)

Combine only when splitting would:

- leave the tree broken or the feature half-wired, or
- tell an incoherent story (e.g. rename without updating the only call site).

**Always** state **why** those paths belong together in one commit when proposing a multi-file commit.

## User initiation

```
❌ AI: "Should I commit these files?"
❌ AI: "Let me commit this..."

✅ Human: "Commit the changed files"
✅ Human: "Do the commits now"
```

**Rule:** Do not suggest or initiate commits. Only run this workflow after an explicit user command.

## Staging

```bash
git add path/to/file.jl
# or, only when justified:
git add path/a.jl path/b.jl
```

- Avoid `git add .` and `git add -A` unless the user explicitly wants everything (still read the full staged diff after).
- Never stage unrelated changes into the same commit.

## Optional git hooks (`.githooks/`)

After `git config core.hooksPath .githooks`:

- **pre-commit:** rejects commits with **more than two** staged files. This
  supports the common “impl + test” or “snippet + verifier” pair without
  allowing large batches. Split bigger changes across commits or adjust the
  hook deliberately.
- **commit-msg:** rejects the commit if **any** message line is longer than
  **80 characters** (including the subject). Wrap prose and bullets; merge
  commits skip this check while `.git/MERGE_HEAD` is present.

## Read the full staged diff

```bash
git diff --staged
```

**Critical:**

- Do not pipe through `head`, `tail`, `less`, or otherwise truncate.
- Read every line; the commit message must reflect the actual diff.

**Exception:** machine-generated or binary files—review only as far as is practical.

## Approval before `git commit`

**Before** running `git commit`, present to the user:

1. **File(s)** to be committed (exact paths).
2. **Full proposed commit message** (subject + body in the format below).
3. If **more than one file:** a **short justification** why one commit is appropriate.

**Wait** for explicit approval. Only then:

```bash
git commit -F- <<'EOF'
<paste full message>
EOF
```

(or equivalent multi-line `-m` usage). Do not commit silently.

## Commit message structure

**1. Subject line (first line)** — Conventional Commits:

`type(scope): specific description from the diff`

- Types include `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, etc.
- `scope` is optional but recommended when it clarifies the area.
- Imperative mood; specific; no vague "update" / "fix things".

**2. Blank line**

**3. Summary** — one short paragraph, **one to three sentences**: what changed and why, for this commit only.

**4. Body detail (scale with the diff)** — after the summary:

- **Small / localized change:** optional bullets only when they add real value; often the summary is enough.
- **Large file, large diff, or several concerns in one commit:** expand with **substantive bullets**. Prefer **grouped sections** (short headings or bullet groups) that mirror the patch: major types/functions added, behavior changes, wiring/exports, migrations, risks, or follow-ups. A few vague bullets are not sufficient when the diff is hundreds of lines or touches multiple subsystems—the message should let a reviewer reconstruct *why* the patch looks the way it does without re-reading every hunk.

Use `-` bullet lists; omit sections that do not apply.

### Line length (enforced when hooks are enabled)

Keep **every** line at **80 characters or fewer**, including the subject and
each bullet. This keeps `git log` readable in terminals and mail archives. If
the hook is enabled, overlong lines cause the commit to fail—rewrap before
retrying.

### Example (single file)

```
refactor(basis): fold lagrange includes behind generator entrypoints

The basis tree listed seven separate lagrange_*.jl includes; they now load
through lagrange_generator.jl and lagrange_generated.jl so registration lives
in one place.

- Drop redundant includes from JuliaFEM.jl
- Note the Basis suffix convention in the include comment
```

### Example (multi-file — requires justification to user)

When proposing:

> **Files:** `src/foo.jl`, `test/foo/test_foo.jl`  
> **Why one commit:** adds `compute_bar` and the regression test that locks its behavior; splitting would leave the feature untested in history.  
> **Message:** …

## Cycle per commit

1. Stage the smallest set (usually one file).
2. `git diff --staged` — read all of it.
3. Propose message + paths (+ multi-file justification if needed).
4. Wait for approval.
5. Commit.
6. Repeat for remaining work.

## Common mistakes

### Vague or mismatched messages

```bash
# ❌ WRONG
git commit -m "Update documentation"

# ✅ RIGHT — subject + summary (+ bullets if needed), from the real diff
```

### Bundling unrelated files

```bash
# ❌ WRONG — two different stories
git add docs/CONTRIBUTING.md src/unrelated_fix.jl

# ✅ RIGHT — two commits, or ask user to split worktrees if mixed in one working tree
```

### Committing without approval

Never run `git commit` until the user has approved the proposed paths and message.

### Truncating the diff

Never use `git diff --staged | head` (etc.). Read the full diff.

## Checklist before each commit

- [ ] Did the user explicitly ask to commit?
- [ ] Is the staged set as **small** as it reasonably can be?
- [ ] If multiple files: is there a **clear single story**, and will you **justify** it to the user?
- [ ] If `.githooks` is active: are there **at most two** staged paths?
- [ ] Did I read the **full** `git diff --staged`?
- [ ] Subject line: Conventional Commits, specific, matches diff?
- [ ] Is **every** message line (subject, summary, bullets) **≤ 80 characters**?
- [ ] Body: blank line, then **1–3 sentence** summary, then **bullets scaled to diff size** (rich, grouped bullets for large / multi-concern commits)?
- [ ] Did I **propose** the commit and wait for **approval** before `git commit`?

## Why this matters

1. **Story:** History is easy to read and bisect.
2. **Review:** Each commit is reviewable in isolation.
3. **Honesty:** Messages match what the diff actually does.
4. **Control:** The author approves each step.

---

Git history is long-lived. Prefer small, well-explained commits and explicit approval before each `git commit`.
