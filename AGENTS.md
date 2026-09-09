# DecisionCraft — Agent Working Guide

This directory (`/home/amirelite_ai/markovProcessTopic`) holds the
**DecisionCraft** course (private repo `shMotaharpour/DecisionCraft`):
a hands-on MILP → MDP → RL → hybrid-architectures course (37 lessons),
built lesson-by-lesson with the user in the "Markov Process" Telegram
topic. The same folder serves as the workspace for that topic.

## Layout

- `phase1_milp/`, `phase2_mdp/`, `phase3_rl/`, `phase4_hybrid/` — lesson
  scripts, one runnable demo per lesson, plus seeded `make_figures*.py`
  generators next to the lesson that owns each figure.
- `notes/` — one compact theory digest per lesson (applied, no proofs;
  theorems labeled *proven*). Figures are embedded after the numbers
  they illustrate.
- `assets/phase{1..4}/` — generated PNGs (all seeded/reproducible);
  Mermaid diagrams live inline in the notes.
- `docs/research/` — evidence files: raw live outputs backing every
  measured claim in a note.
- `data/` — shared datasets (e.g. `cvrp_15node.json`).
- `pyproject.toml` / `uv.lock` — environment (uv-managed).

## Hard conventions (user requirements)

- **English only** in all repo files; Persian is for chat replies only.
- **Toolchain:** OR-Tools + `scipy.optimize.milp`/`linprog` (the HiGHS
  engine inside scipy is fine; the standalone `highspy`/`pyomo` packages
  are NOT used). PyTorch CPU-only via the `pytorch-cpu` index.
- **Honesty rules:** every measured claim cites an evidence file under
  `docs/research/`; negative results are reported, never tuned away;
  figures are seeded; when a figure shows a loss (e.g. DW slower than
  direct MILP at toy scale), the title stays honest.
- **Commit style:** `chNN: <lesson/description>` — one commit (+ push)
  per lesson, per figure, or per fix, so any item can be reverted or
  amended independently.
- Figures: seeded generator checked in beside the lesson; embed in the
  note after the related numbers; append the figure's data to the
  evidence file; visually inspect (or numerically assert) before commit.
- Lesson scripts must stay small: run in minutes on a normal machine.

## Working rules

- Verify every command by running it before writing it into a note.
- Long runs (>2 min) go to background with output to `/tmp` and are
  waited on via the process tools.
- When adding a lesson or figure: script → run → evidence file → note
  embed → README row if new lesson → commit+push individually.
