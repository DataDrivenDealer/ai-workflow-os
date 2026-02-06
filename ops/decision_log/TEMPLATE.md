# Decision Log Template

**Document Type**: Decision Record  
**Template Version**: 1.0.0

---

## How to Use

1. Copy this template to create a new decision record
2. Name the file: `DEC_YYYYMMDD_<SHORT_NAME>.md`
3. Fill in all required sections
4. Commit to `ops/decision-log/`

---

## Template

```markdown
# Decision Record: [DEC_YYYYMMDD_SHORT_NAME]

**ID**: DEC_YYYYMMDD_XXX  
**Date**: YYYY-MM-DD  
**Status**: [proposed | accepted | deprecated | superseded]  
**Decider**: [Name/Role]

---

## Context

[Describe the situation that required a decision. What problem are we solving?]

## Decision

[State the decision that was made clearly and unambiguously]

## Rationale

[Explain why this decision was made. What alternatives were considered?]

### Alternatives Considered

1. **Alternative A**: [Description]
   - Pros: ...
   - Cons: ...
   
2. **Alternative B**: [Description]
   - Pros: ...
   - Cons: ...

### Why This Choice

[Explain the reasoning behind selecting this option]

## Consequences

### Positive

- [Benefit 1]
- [Benefit 2]

### Negative

- [Tradeoff 1]
- [Tradeoff 2]

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ... | Low/Med/High | Low/Med/High | ... |

## Related

- **Specs**: [List related specifications]
- **Tasks**: [List related TaskCards]
- **Previous Decisions**: [List superseded decisions]

## Acceptance

| Role | Name | Date | Status |
|------|------|------|--------|
| Owner | | | pending |
| Reviewer | | | pending |

---

**Change Log**:
| Date | Author | Change |
|------|--------|--------|
| YYYY-MM-DD | [Name] | Initial creation |
```
