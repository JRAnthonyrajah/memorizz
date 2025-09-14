# memorizz: Fork + Upstream Workflow

A disciplined way to keep your own modifications **and** regularly pull new changes from the original author.

---

## Terminology / Mental Model

- **`upstream`**: the original author’s repository (`RichmondAlake/memorizz`).
- **`origin`**: **your** fork on GitHub.
- **`upstream-main`**: a local branch that **exactly mirrors** `upstream/main` (never commit here).
- **`my-changes`**: your long-lived integration branch; your commits sit **on top of** `upstream-main`.

---

## One-Time Setup

1. **Fork on GitHub**
   - Go to `https://github.com/RichmondAlake/memorizz`
   - Click **Fork** → choose your account → **Create fork** (copy `main` branch only is fine).

2. **Clone your fork**
   ```bash
   git clone https://github.com/<your-username>/memorizz.git
   cd memorizz
   ```

3. **Add the author’s repo as upstream**
   ```bash
   git remote add upstream https://github.com/RichmondAlake/memorizz.git
   git fetch upstream
   ```

4. **Create/track a read-only mirror of `upstream/main`**
   ```bash
   git checkout -B upstream-main upstream/main
   git branch --set-upstream-to=upstream/main upstream-main
   ```

5. **Create your integration branch for edits**
   ```bash
   git checkout -b my-changes upstream-main
   ```

6. **(Recommended) Enable conflict reuse**
   ```bash
   git config rerere.enabled true
   ```

7. **Begin committing and push your branch to your fork**
   ```bash
   # ...edit, git add, git commit...
   git push -u origin my-changes
   ```

**Verify remotes (sanity check):**
```bash
git remote -v
# origin   https://github.com/<you>/memorizz.git (fetch/push)
# upstream https://github.com/RichmondAlake/memorizz.git (fetch/push)
```

---

## Regular Update Cycle (keep your changes + pull upstream)

Whenever the author updates `main`:

```bash
# 1) Refresh upstream mirror
git fetch upstream
git checkout upstream-main
git reset --hard upstream/main    # keep this branch identical to upstream

# 2) Replay your commits on the new base
git checkout my-changes
git rebase upstream-main          # clean, linear history

# 3) Resolve conflicts, run tests, then update your fork
git push --force-with-lease
```

**Why rebase (not merge)?**  
Rebase keeps history as “`upstream/main` + your patches” without merge bubbles, which makes future updates simpler and diffs clearer. If you prefer merges, replace step (2) with:
```bash
git merge upstream-main
git push
```
…but accept messier history over time.

---

## Conflict Resolution Quick Guide

During `git rebase upstream-main`, if conflicts appear:

```bash
# edit files to resolve conflicts
git add <file(s)>
git rebase --continue
```

If you need to bail out:
```bash
git rebase --abort
```

If the rebase is done but you realize it’s wrong, you can return to the pre-rebase state using the ORIG_HEAD reference:
```bash
git reset --hard ORIG_HEAD
```

---

## Good Hygiene & Options as Your Patch Set Grows

### 1) Keep logical changes in topic branches
Instead of piling everything onto `my-changes`, create small focused branches off `upstream-main`:
```bash
git checkout upstream-main
git checkout -b feature/better-embedding
# ...commit...
git push -u origin feature/better-embedding
```
These are easier to rebase, test, and (optionally) contribute upstream.

### 2) Autosquash to keep history tidy
Use conventional fixup commits and autosquash:
```bash
git commit --fixup <commit>
git rebase -i --autosquash upstream-main
```

### 3) Patch-queue approach (advanced)
If you prefer a literal patch queue:
```bash
git format-patch upstream-main..my-changes -o /tmp/mypatches
# update upstream-main, then:
git checkout -B my-changes upstream-main
git am /tmp/mypatches/*.patch
```

### 4) Worktrees for side-by-side checkouts
Keep a clean view of upstream alongside your working branch:
```bash
git worktree add ../memorizz-upstream upstream-main
# ../memorizz-upstream contains the pristine upstream mirror
```

### 5) Protect `upstream-main` from accidental commits
If your Git supports it, mark it as a protected branch locally (advisory), and enforce the habit: **never** commit directly there. Practically, always develop on `my-changes` or feature branches.

---

## Contributing Back to the Author (Optional)

1. Branch off your integration or `upstream-main`:
   ```bash
   git checkout -b feature/xyz upstream-main
   git cherry-pick <commits-from-my-changes>   # or re-implement cleanly
   ```
2. Clean up commits (squash/reword).
3. Push and open a PR **from your fork** to the upstream repo.

---

## If You Already Cloned the Author’s Repo First

Convert in-place (no re-clone):
```bash
git remote rename origin upstream
# create your fork on GitHub, then:
git remote add origin https://github.com/<your-username>/memorizz.git

git fetch upstream
git checkout -B upstream-main upstream/main
git checkout -b my-changes upstream-main
git push -u origin my-changes
```

---

## Troubleshooting

- **“I pushed after rebase and GitHub shows it diverged.”**  
  Use `--force-with-lease` (not `--force`). It safely updates the remote while protecting others’ work:
  ```bash
  git push --force-with-lease
  ```

- **“I committed by mistake on `upstream-main`.”**  
  Move the commit to `my-changes`:
  ```bash
  git checkout my-changes
  git cherry-pick <bad-commit-sha>
  git checkout upstream-main
  git reset --hard upstream/main
  ```

- **“My local `main` exists—do I need it?”**  
  Not for this pattern. You can keep it unused or delete it locally to avoid confusion:
  ```bash
  git branch -D main
  ```

- **“I need to keep a long-running local patch and also bump it often.”**  
  Rebase frequently, keep commits small and well-scoped, and enable `rerere` (already done) to reduce future conflict effort.

---

## Minimal Daily/Weekly Command Set (Copy/Paste)

```bash
# Update upstream mirror
git fetch upstream
git checkout upstream-main
git reset --hard upstream/main

# Rebase your work
git checkout my-changes
git rebase upstream-main

# After tests pass
git push --force-with-lease
```

---

## Optional Quality-of-Life

- **Sign commits** (recommended for provenance):
  ```bash
  git config commit.gpgsign true
  git config user.signingkey <your-key-id>
  ```
- **Show upstream tracking in prompt**: use a Git-aware shell prompt or `git status -sb`.
- **Pre-commit hooks**: add linters/formatters to avoid rebasing conflicts caused by style drift.

---

### Summary

- Keep `upstream-main` **pristine** (a mirror of `upstream/main`).
- Do all edits on `my-changes` (or focused feature branches).
- **Rebase** your branches onto `upstream-main` when upstream changes.
- Push with `--force-with-lease` after rebases.
- Use `rerere`, small commits, and topic branches to keep the process painless.

