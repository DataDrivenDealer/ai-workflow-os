# Freeze Record Template

**Record ID**: FREEZE_{SPEC_ID}_{DATE}  
**Date**: YYYY-MM-DD  
**Frozen By**: [Governance Officer Name]  
**Status**: 🔒 Frozen | 📝 Draft | ⚠️ Pending Review

---

## 1. Freeze Metadata

| Field | Value |
|-------|-------|
| **Spec ID** | [Specification being frozen] |
| **Spec Version** | [Semver, e.g., 1.0.0] |
| **Freeze Date** | [YYYY-MM-DD] |
| **Effective From** | [YYYY-MM-DD] |
| **Frozen By** | [Name/Role] |
| **Approved By** | [Governance Authority] |

---

## 2. Scope of Freeze

### 2.1 What is Frozen
- [ ] Full specification content
- [ ] Interface definitions
- [ ] Schema definitions
- [ ] API contracts
- [ ] Configuration defaults

### 2.2 What Remains Mutable
- [ ] Implementation details
- [ ] Documentation clarifications
- [ ] Non-breaking additions

---

## 3. Rationale

**Why is this freeze necessary?**

[Explain the business/technical reason for freezing this specification]

**Impact Assessment:**

| Impact Area | Level | Notes |
|-------------|-------|-------|
| Downstream Systems | High/Med/Low | |
| Development Velocity | High/Med/Low | |
| Compliance | High/Med/Low | |

---

## 4. Dependencies

### 4.1 Specs This Depends On
| Spec ID | Version | Status |
|---------|---------|--------|
| | | |

### 4.2 Specs That Depend On This
| Spec ID | Impact | Migration Required |
|---------|--------|-------------------|
| | | |

---

## 5. Change Control

### 5.1 Amendment Process
To modify a frozen spec:

1. Create proposal in `ops/proposals/`
2. Submit for governance review
3. Impact assessment required
4. Minimum 2 approvers for canon specs
5. Update freeze record with amendment

### 5.2 Exception Process
For emergency changes:

1. File deviation in `ops/deviations/`
2. Document business justification
3. Set expiration date for deviation
4. Plan remediation to return to compliance

---

## 6. Verification Checksum

```yaml
freeze_verification:
  spec_file: "[path/to/spec.md]"
  sha256: "[calculated checksum]"
  frozen_at: "[ISO8601 timestamp]"
  verified_by: "[Governance Officer]"
```

---

## 7. Audit Trail

| Timestamp | Actor | Action | Notes |
|-----------|-------|--------|-------|
| | | Freeze initiated | |
| | | Freeze approved | |
| | | Freeze effective | |

---

## 8. Signatures

### Requested By
- **Name**: 
- **Role**: 
- **Date**: 

### Approved By
- **Name**: 
- **Role**: 
- **Date**: 

---

*Template Version: 1.0.0*  
*Maintained by: Governance Officer*
