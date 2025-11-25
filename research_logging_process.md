# ML Research Logging Process

> One file per day. Everything in one place.

---

## The System

**One daily log** contains everything: goals, experiments, insights, and commits (auto-logged).

```
logs/
├── daily/              # One file per day (auto-created on first commit)
│   ├── 2025-11-25.md
│   └── 2025-11-26.md
├── commits.md          # Searchable history of all commits
└── insights.md         # Cross-cutting learnings
```

---

## Daily Log Template

Each day's file (`logs/daily/YYYY-MM-DD.md`) contains:

```markdown
# Research Log: 2025-11-25

## Today's Goal
> What are you trying to accomplish?

---

## Experiments

### EXP: `20251125_dia_001_baseline`

**Status:** 🟡 In Progress | 🟢 Complete | 🔴 Failed  
**Hypothesis:** Reducing LR to 2e-5 will stabilize training  
**Changes:** Modified config.json LR from 5e-5 to 2e-5  
**Config:** `lr=2e-5, batch=4, epochs=10`

**Results:**
| Metric | Value |
|--------|-------|
| Final Loss | 0.45 |
| Best Val | 0.42 |

**Conclusion:** Hypothesis confirmed. Training stable but slower.

**Next:** Try LR scheduler instead of fixed low LR.

---

### EXP: `20251125_dia_002_scheduler`

(copy the block above for each new experiment)

---

## Key Insights
- 

## Blockers / Questions
- 

## Tomorrow's Plan
- 

---

## Commits
(auto-appended by git hook)
```

---

## Experiment ID Convention

```
{YYYYMMDD}_{project}_{sequence}_{brief_desc}
```

**Examples:**
- `20251125_dia_001_baseline`
- `20251125_dia_002_lora_rank16`
- `20251125_dia_003_longer_context`

---

## Workflow

### 1. Start your day
Open today's log (or let the first commit create it).

### 2. Before each experiment
Add a new `### EXP:` block with:
- Hypothesis (what you expect)
- Changes (what you're doing differently)
- Config (key hyperparameters)

### 3. Commit your code
```bash
git add -A
git commit -m "20251125_dia_001: reduce LR to 2e-5"
```
→ Automatically appended to `## Commits` section

### 4. After the run
Fill in Results, Conclusion, and Next steps.

---

## Git Commit Convention

Include the experiment ID in commit messages:

```bash
git commit -m "20251125_dia_001: reduce LR to 2e-5 for stability"
git commit -m "20251125_dia_002: add LoRA with rank 16"
git commit -m "fix: resolve gradient accumulation bug"
```

### Search commits by experiment
```bash
git log --grep="dia_001" --oneline
grep "dia_001" logs/commits.md
```

---

## Auto-Logging (Git Hook)

A post-commit hook automatically:
1. Creates today's daily log if it doesn't exist
2. Appends each commit to the `## Commits` section
3. Logs to `logs/commits.md` for searchability

**What gets logged:**
```markdown
- `14:30` [`a1b2c3d`] **main**: 20251125_dia_001: reduce LR (3 files)
```

---

## Quick Reference

```bash
# View today's log
cat logs/daily/$(date +%Y-%m-%d).md

# Search all experiments for a term
grep -r "hypothesis" logs/daily/

# Find commits for an experiment
grep "dia_001" logs/commits.md

# Git search
git log --grep="dia_001" --oneline
```

---

## Best Practices

### Do ✅
- Write the hypothesis BEFORE running
- Fill in conclusions IMMEDIATELY after
- Log failed experiments (prevents repeating mistakes)
- Use descriptive experiment IDs

### Don't ❌
- Skip the hypothesis ("let's see what happens")
- Delete failed experiments
- Trust your memory
- Hardcode hyperparameters in code

---

## Checklist

**Before experiment:**
- [ ] Hypothesis written
- [ ] Config documented
- [ ] Code committed with EXP_ID

**After experiment:**
- [ ] Results filled in
- [ ] Conclusion written
- [ ] Next steps identified

---

*The goal isn't perfect documentation—it's capturing enough context that you can understand what you tried and why.*
