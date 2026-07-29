# Share-k AI Agent Instructions

## Shared product contract

Before AI planning or implementation, read:

1. `../docs/CONTEXT.md`
2. `../docs/product/governance/decision-log.md`
3. relevant ADRs under `../docs/adr/`
4. the current sprint under `../docs/product/sprints/`

The AI service returns structured analysis only. NestJS owns authorization,
business decisions, state transitions, and audit persistence.

## Agent skills

### Issue tracker

Issues are tracked in `ITI-Sharek/AI_Agents` GitHub Issues. See
`docs/agents/issue-tracker.md`.

### Triage labels

Use the five default triage labels. See `docs/agents/triage-labels.md`.

### Domain docs

Share-k has one canonical product context in the sibling Documentation
repository. See `docs/agents/domain.md`.
