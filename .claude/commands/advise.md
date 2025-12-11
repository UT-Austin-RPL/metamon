Search the metamon experiment skills registry to find relevant past experiments, learnings, and recommendations.

## How this works

When you run `/advise [optional query]`, Claude will:

1. **Search the skills registry** in `.claude/skills/` for relevant experiments
2. **Surface key findings** including:
   - Similar past experiments and their outcomes
   - Common failure patterns to avoid
   - Recommended hyperparameters and configurations
   - Prerequisite steps and gotchas
3. **Generate recommendations** for your specific experiment based on accumulated team knowledge

## Usage

### Before starting any new experiment:
```bash
/advise training a Gen1 OU specialist with sleep strategy
```

Claude will search for relevant skills about:
- Gen1 OU training workflows
- Sleep-focused reward functions
- Configuration selection for specialists
- Common pitfalls in similar experiments

### For troubleshooting:
```bash
/advise format filtering error
```

Claude will find skills related to data loading issues, format filtering requirements, and solutions.

### For general exploration:
```bash
/advise
```

Without a query, Claude will present an overview of the available skills organized by category.

## What to expect

The `/advise` command will return:

1. **Relevant skills** - Ordered by relevance to your query
2. **Key takeaways** - Quick summary of main findings from each skill
3. **Recommended approach** - Synthesized guidance combining multiple skills
4. **Warnings** - Critical failure modes to avoid from past experiments
5. **Starting point** - Concrete commands or configurations to begin with

## Instructions for Claude

When the user runs `/advise`:

1. **Parse the query** to understand the experiment goal, if provided
2. **Search `.claude/skills/` recursively** for relevant skill files
3. **Read relevant skills** (prioritize by keyword match and category)
4. **Synthesize findings**:
   - Extract applicable hyperparameters
   - Identify similar failure patterns
   - Note prerequisites and setup requirements
   - Highlight configuration recommendations
5. **Present concisely**:
   - 1-2 sentence summary per relevant skill
   - Bulleted key takeaways
   - Concrete next steps
6. **Ask clarifying questions** if the query is ambiguous

**Proactive usage**: You should automatically suggest running `/advise` when:
- User describes starting a new training experiment
- User asks "how do I..." questions about experiments
- User encounters an error that might have known solutions
- User asks about configuration or hyperparameter choices

**Specificity**: Always reference specific skill files by name (e.g., "See `psro-nash-training` skill") and include file paths to code/configs mentioned in skills.
