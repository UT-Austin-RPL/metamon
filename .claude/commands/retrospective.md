Generate structured documentation of experiment learnings from the current conversation.

## How this works

After completing an experiment or training run, run `/retrospective` to have Claude:

1. **Review the conversation** to identify key learnings
2. **Extract insights** including:
   - What approach was tried
   - What worked and what didn't
   - Hyperparameters used and their effects
   - Errors encountered and solutions
   - Unexpected findings
3. **Generate a skill file** in `.claude/skills/` following the standard template
4. **Optionally create a git branch** for team review

## Usage

### After completing a training experiment:
```bash
/retrospective
```

Claude will analyze the conversation and create a skill file capturing all learnings.

### With a specific focus:
```bash
/retrospective focus on dynamic damping issues
```

Claude will emphasize particular aspects of the experiment when documenting.

### Update existing skill:
```bash
/retrospective update psro-nash-training
```

Claude will merge new findings into an existing skill file rather than creating a new one.

## What gets captured

The `/retrospective` command documents:

### What Worked ✅
- Successful approaches and configurations
- Hyperparameters that produced good results
- Metric ranges indicating healthy training
- Commands that ran successfully

### What Failed ❌
- Failed approaches and why they didn't work
- Error messages and their solutions
- Metric patterns indicating problems
- Configuration mistakes to avoid

### Key Parameters
- Exact hyperparameter values tested
- Sensitivity analysis results (if explored)
- Recommended ranges vs red flags
- Scaling relationships

### Reproducibility
- Complete command with all flags
- Environment setup requirements
- Dataset locations and versions
- Model checkpoints used

### Unexpected Findings
- Surprising results or behaviors
- Hypotheses generated during experiment
- Follow-up questions for future work

## Instructions for Claude

When the user runs `/retrospective`:

1. **Analyze the conversation** to identify:
   - What experiment was performed
   - What commands were run
   - What results were observed
   - What problems were encountered
   - What decisions were made and why

2. **Determine skill category**:
   - Training workflows → `.claude/skills/training/`
   - Configuration selection → `.claude/skills/config/`
   - Troubleshooting → `.claude/skills/troubleshooting/`
   - Evaluation methods → `.claude/skills/evaluation/`

3. **Choose skill name**:
   - Use kebab-case (e.g., `psro-nash-training`)
   - Be specific (not "training-experiments", but "gen1-sleep-strategy-training")
   - Include format/context if specialized

4. **Follow the template** in `.claude/SKILL_TEMPLATE.md`

5. **Emphasize failures**:
   - Document what DIDN'T work in detail
   - Include error messages verbatim
   - Explain root causes
   - Provide solutions

6. **Be specific**:
   - Include exact commands (copy-paste ready)
   - Show actual metric values, not "loss decreased"
   - Reference specific hardware/environment
   - List all prerequisites explicitly

7. **Present for review**:
   - Show the generated skill file content
   - Ask if user wants to commit it
   - Suggest related skills to cross-reference

**Proactive usage**: You should suggest running `/retrospective` when:
- An experiment completes (successful or failed)
- User solves a tricky bug or configuration issue
- User discovers a non-obvious insight about training
- User compares multiple approaches and reaches a conclusion
- An unexpected result requires investigation and resolution

**Quality check**: Before finalizing, ensure:
- "Failed Attempts" section is substantive (not just "tried X, didn't work")
- Hyperparameters include actual values, not descriptions
- Commands are complete and executable
- Prerequisites are explicit (env vars, servers, data paths)
- Skill name is specific and searchable
