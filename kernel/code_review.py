"""
Code Review Engine for Pair Programming

Implements the Pair Programming workflow as defined in:
- PAIR_PROGRAMMING_STANDARD
- state_machine.yaml (pair_programming section)

This module provides:
- Review session management
- Four-dimensional code review (Quality, Requirements, Completeness, Optimization)
- Review report generation
- Revision tracking
- Expert persona simulation

All review operations respect:
- Role Mode permissions (reviewer mode required)
- Self-review prohibition
- Governance constraints
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml


class IssueSeverity(Enum):
    """Issue severity levels for code review."""
    CRITICAL = "CRITICAL"   # Security vulnerability, data loss risk, crash
    MAJOR = "MAJOR"         # Functional bug, missing requirement
    MINOR = "MINOR"         # Code smell, minor bug
    SUGGESTION = "SUGGESTION"  # Optimization opportunity


class ReviewVerdict(Enum):
    """Review verdict outcomes."""
    APPROVED = "APPROVED"           # All checks passed
    NEEDS_REVISION = "NEEDS_REVISION"  # Issues found, revision needed
    BLOCKED = "BLOCKED"             # Critical blocker found


class ReviewDimension(Enum):
    """Four dimensions of code review."""
    QUALITY = "quality_check"           # Q-Check: Code quality
    REQUIREMENTS = "requirements_check"  # R-Check: Requirements verification
    COMPLETENESS = "completeness_check"  # C-Check: Completeness
    OPTIMIZATION = "optimization_check"  # O-Check: Optimization opportunities


@dataclass
class ReviewIssue:
    """Represents a single issue found during code review."""
    issue_id: str
    dimension: ReviewDimension
    check_id: str
    severity: IssueSeverity
    description: str
    file_path: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    code_snippet: Optional[str] = None
    suggested_fix: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "dimension": self.dimension.value,
            "check_id": self.check_id,
            "severity": self.severity.value,
            "description": self.description,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "code_snippet": self.code_snippet,
            "suggested_fix": self.suggested_fix,
            "context": self.context,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewIssue":
        return cls(
            issue_id=data["issue_id"],
            dimension=ReviewDimension(data["dimension"]),
            check_id=data["check_id"],
            severity=IssueSeverity(data["severity"]),
            description=data["description"],
            file_path=data["file_path"],
            line_start=data.get("line_start"),
            line_end=data.get("line_end"),
            code_snippet=data.get("code_snippet"),
            suggested_fix=data.get("suggested_fix"),
            context=data.get("context", {}),
        )


@dataclass
class OptimizationSuggestion:
    """Represents an optimization suggestion (non-blocking)."""
    suggestion_id: str
    check_id: str
    description: str
    file_path: str
    line_start: Optional[int] = None
    current_code: Optional[str] = None
    suggested_code: Optional[str] = None
    rationale: str = ""
    estimated_impact: str = ""  # e.g., "minor", "moderate", "significant"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "suggestion_id": self.suggestion_id,
            "check_id": self.check_id,
            "description": self.description,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "current_code": self.current_code,
            "suggested_code": self.suggested_code,
            "rationale": self.rationale,
            "estimated_impact": self.estimated_impact,
        }


@dataclass
class DimensionResult:
    """Result of a single review dimension check."""
    dimension: ReviewDimension
    passed: bool
    issues: List[ReviewIssue] = field(default_factory=list)
    suggestions: List[OptimizationSuggestion] = field(default_factory=list)
    coverage_percentage: Optional[float] = None  # For requirements/completeness
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "suggestions": [s.to_dict() for s in self.suggestions],
            "coverage_percentage": self.coverage_percentage,
            "notes": self.notes,
        }


@dataclass
class ReviewReport:
    """Complete code review report."""
    report_id: str
    task_id: str
    reviewer_agent_id: str
    builder_agent_id: str
    artifacts_reviewed: List[str]
    reviewed_at: datetime
    revision_number: int
    
    verdict: ReviewVerdict
    
    # Dimension results
    quality_result: DimensionResult
    requirements_result: DimensionResult
    completeness_result: DimensionResult
    optimization_result: DimensionResult
    
    # Summary counts
    critical_count: int = 0
    major_count: int = 0
    minor_count: int = 0
    suggestion_count: int = 0
    
    # Metadata
    personas_used: List[str] = field(default_factory=list)
    review_duration_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "task_id": self.task_id,
            "reviewer_agent_id": self.reviewer_agent_id,
            "builder_agent_id": self.builder_agent_id,
            "artifacts_reviewed": self.artifacts_reviewed,
            "reviewed_at": self.reviewed_at.isoformat(),
            "revision_number": self.revision_number,
            "verdict": self.verdict.value,
            "summary": {
                "critical_count": self.critical_count,
                "major_count": self.major_count,
                "minor_count": self.minor_count,
                "suggestion_count": self.suggestion_count,
            },
            "quality_check": self.quality_result.to_dict(),
            "requirements_check": self.requirements_result.to_dict(),
            "completeness_check": self.completeness_result.to_dict(),
            "optimization_check": self.optimization_result.to_dict(),
            "personas_used": self.personas_used,
            "review_duration_seconds": self.review_duration_seconds,
        }
    
    def to_yaml(self) -> str:
        """Export report as YAML."""
        return yaml.safe_dump(self.to_dict(), default_flow_style=False, sort_keys=False)
    
    def to_markdown(self) -> str:
        """Export report as Markdown."""
        lines = [
            f"# Code Review Report: {self.report_id}",
            "",
            f"**Task**: {self.task_id}",
            f"**Reviewer**: {self.reviewer_agent_id}",
            f"**Builder**: {self.builder_agent_id}",
            f"**Revision**: #{self.revision_number}",
            f"**Reviewed At**: {self.reviewed_at.isoformat()}",
            "",
            "---",
            "",
            f"## Verdict: {self.verdict.value}",
            "",
            "### Issue Summary",
            "",
            f"| Severity | Count |",
            f"|----------|-------|",
            f"| CRITICAL | {self.critical_count} |",
            f"| MAJOR | {self.major_count} |",
            f"| MINOR | {self.minor_count} |",
            f"| SUGGESTION | {self.suggestion_count} |",
            "",
            "---",
            "",
        ]
        
        # Add dimension results
        for dim_result in [
            self.quality_result,
            self.requirements_result,
            self.completeness_result,
            self.optimization_result,
        ]:
            status = "✅ PASSED" if dim_result.passed else "❌ FAILED"
            lines.append(f"## {dim_result.dimension.value.replace('_', ' ').title()}: {status}")
            lines.append("")
            
            if dim_result.coverage_percentage is not None:
                lines.append(f"**Coverage**: {dim_result.coverage_percentage:.1f}%")
                lines.append("")
            
            if dim_result.issues:
                lines.append("### Issues")
                lines.append("")
                for issue in dim_result.issues:
                    lines.append(f"#### [{issue.severity.value}] {issue.check_id}: {issue.description}")
                    lines.append(f"- **File**: `{issue.file_path}`")
                    if issue.line_start:
                        lines.append(f"- **Line**: {issue.line_start}")
                    if issue.code_snippet:
                        lines.append(f"- **Code**:")
                        lines.append(f"  ```")
                        lines.append(f"  {issue.code_snippet}")
                        lines.append(f"  ```")
                    if issue.suggested_fix:
                        lines.append(f"- **Suggested Fix**: {issue.suggested_fix}")
                    lines.append("")
            
            if dim_result.suggestions:
                lines.append("### Optimization Suggestions")
                lines.append("")
                for suggestion in dim_result.suggestions:
                    lines.append(f"#### {suggestion.check_id}: {suggestion.description}")
                    lines.append(f"- **File**: `{suggestion.file_path}`")
                    lines.append(f"- **Impact**: {suggestion.estimated_impact}")
                    lines.append(f"- **Rationale**: {suggestion.rationale}")
                    if suggestion.suggested_code:
                        lines.append(f"- **Suggested Code**:")
                        lines.append(f"  ```")
                        lines.append(f"  {suggestion.suggested_code}")
                        lines.append(f"  ```")
                    lines.append("")
            
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)


class ReviewSession:
    """
    Manages a code review session for Pair Programming.
    
    Ensures:
    - Reviewer is different from builder (no self-review)
    - Proper state transitions
    - Revision tracking
    """
    
    def __init__(
        self,
        session_id: str,
        task_id: str,
        reviewer_agent_id: str,
        builder_agent_id: str,
        artifacts: List[str],
        taskcard_content: str,
        spec_references: List[str],
    ):
        if reviewer_agent_id == builder_agent_id:
            raise ValueError("Self-review prohibited: reviewer and builder must be different agents")
        
        self.session_id = session_id
        self.task_id = task_id
        self.reviewer_agent_id = reviewer_agent_id
        self.builder_agent_id = builder_agent_id
        self.artifacts = artifacts
        self.taskcard_content = taskcard_content
        self.spec_references = spec_references
        
        self.created_at = datetime.now(timezone.utc)
        self.revision_number = 1
        self.reports: List[ReviewReport] = []
        self.status = "active"
    
    def generate_report_id(self) -> str:
        """Generate unique report ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"REV-{self.task_id}-{timestamp}-R{self.revision_number}"
    
    def increment_revision(self) -> int:
        """Increment and return new revision number."""
        self.revision_number += 1
        return self.revision_number
    
    def add_report(self, report: ReviewReport) -> None:
        """Add a review report to the session."""
        self.reports.append(report)
    
    def get_latest_report(self) -> Optional[ReviewReport]:
        """Get the most recent review report."""
        return self.reports[-1] if self.reports else None


class CodeReviewEngine:
    """
    Main engine for conducting code reviews.
    
    This engine:
    - Creates review sessions
    - Coordinates the four review dimensions
    - Generates comprehensive review reports
    - Tracks revision cycles
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the code review engine."""
        self.config_path = config_path or Path(__file__).parent / "state_machine.yaml"
        self._config: Optional[Dict[str, Any]] = None
        self._sessions: Dict[str, ReviewSession] = {}
    
    @property
    def config(self) -> Dict[str, Any]:
        """Load and cache Pair Programming configuration."""
        if self._config is None:
            if self.config_path.exists():
                with self.config_path.open("r", encoding="utf-8") as f:
                    full_config = yaml.safe_load(f) or {}
                self._config = full_config.get("pair_programming", {})
            else:
                self._config = {}
        return self._config or {}
    
    @property
    def max_revision_cycles(self) -> int:
        """Get maximum allowed revision cycles."""
        return self.config.get("max_revision_cycles", 3)
    
    @property
    def self_review_prohibited(self) -> bool:
        """Check if self-review is prohibited."""
        return self.config.get("self_review_prohibited", True)
    
    def get_review_checks(self, dimension: ReviewDimension) -> List[Dict[str, Any]]:
        """Get check definitions for a review dimension."""
        dimensions = self.config.get("review_dimensions", {})
        dim_config = dimensions.get(dimension.value, {})
        return dim_config.get("checks", [])
    
    def get_severity_info(self, severity: IssueSeverity) -> Dict[str, Any]:
        """Get severity level information."""
        levels = self.config.get("severity_levels", [])
        for level in levels:
            if level.get("name") == severity.value:
                return level
        return {"name": severity.value, "blocks_merge": False, "description": ""}
    
    def is_blocking_severity(self, severity: IssueSeverity) -> bool:
        """Check if a severity level blocks merge."""
        info = self.get_severity_info(severity)
        return info.get("blocks_merge", False)
    
    def create_session(
        self,
        task_id: str,
        reviewer_agent_id: str,
        builder_agent_id: str,
        artifacts: List[str],
        taskcard_content: str,
        spec_references: Optional[List[str]] = None,
    ) -> ReviewSession:
        """
        Create a new code review session.
        
        Args:
            task_id: Task being reviewed
            reviewer_agent_id: Agent performing the review
            builder_agent_id: Agent who wrote the code
            artifacts: List of artifact paths to review
            taskcard_content: Content of the TaskCard
            spec_references: List of spec IDs referenced
        
        Returns:
            New ReviewSession
        
        Raises:
            ValueError: If self-review is attempted
        """
        if self.self_review_prohibited and reviewer_agent_id == builder_agent_id:
            raise ValueError(
                "Self-review prohibited: reviewer and builder must be different agents. "
                "This ensures objective code review quality."
            )
        
        session_id = self._generate_session_id(task_id, reviewer_agent_id)
        session = ReviewSession(
            session_id=session_id,
            task_id=task_id,
            reviewer_agent_id=reviewer_agent_id,
            builder_agent_id=builder_agent_id,
            artifacts=artifacts,
            taskcard_content=taskcard_content,
            spec_references=spec_references or [],
        )
        self._sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[ReviewSession]:
        """Get an existing review session."""
        return self._sessions.get(session_id)
    
    def _generate_session_id(self, task_id: str, reviewer_id: str) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        hash_input = f"{task_id}-{reviewer_id}-{timestamp}"
        short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        return f"RSESS-{task_id}-{short_hash}"
    
    def determine_verdict(self, issues: List[ReviewIssue]) -> ReviewVerdict:
        """
        Determine review verdict based on issues found.
        
        Logic:
        - APPROVED: No CRITICAL or MAJOR issues
        - NEEDS_REVISION: Has CRITICAL or MAJOR issues
        - BLOCKED: Used for special blocking conditions
        """
        critical_count = sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL)
        major_count = sum(1 for i in issues if i.severity == IssueSeverity.MAJOR)
        
        if critical_count > 0 or major_count > 0:
            return ReviewVerdict.NEEDS_REVISION
        
        return ReviewVerdict.APPROVED
    
    def count_issues_by_severity(
        self, issues: List[ReviewIssue]
    ) -> Tuple[int, int, int, int]:
        """Count issues by severity level."""
        critical = sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL)
        major = sum(1 for i in issues if i.severity == IssueSeverity.MAJOR)
        minor = sum(1 for i in issues if i.severity == IssueSeverity.MINOR)
        suggestion = sum(1 for i in issues if i.severity == IssueSeverity.SUGGESTION)
        return critical, major, minor, suggestion
    
    def create_review_report(
        self,
        session: ReviewSession,
        quality_result: DimensionResult,
        requirements_result: DimensionResult,
        completeness_result: DimensionResult,
        optimization_result: DimensionResult,
        personas_used: Optional[List[str]] = None,
        review_duration_seconds: Optional[float] = None,
    ) -> ReviewReport:
        """
        Create a complete review report from dimension results.
        
        Args:
            session: Active review session
            quality_result: Result of quality check
            requirements_result: Result of requirements check
            completeness_result: Result of completeness check
            optimization_result: Result of optimization check
            personas_used: Expert personas used in review
            review_duration_seconds: Time taken for review
        
        Returns:
            Complete ReviewReport
        """
        # Collect all issues
        all_issues = (
            quality_result.issues +
            requirements_result.issues +
            completeness_result.issues +
            optimization_result.issues
        )
        
        # Count by severity
        critical, major, minor, suggestion = self.count_issues_by_severity(all_issues)
        
        # Add suggestions from optimization
        suggestion += len(optimization_result.suggestions)
        
        # Determine verdict
        verdict = self.determine_verdict(all_issues)
        
        report = ReviewReport(
            report_id=session.generate_report_id(),
            task_id=session.task_id,
            reviewer_agent_id=session.reviewer_agent_id,
            builder_agent_id=session.builder_agent_id,
            artifacts_reviewed=session.artifacts,
            reviewed_at=datetime.now(timezone.utc),
            revision_number=session.revision_number,
            verdict=verdict,
            quality_result=quality_result,
            requirements_result=requirements_result,
            completeness_result=completeness_result,
            optimization_result=optimization_result,
            critical_count=critical,
            major_count=major,
            minor_count=minor,
            suggestion_count=suggestion,
            personas_used=personas_used or [],
            review_duration_seconds=review_duration_seconds,
        )
        
        session.add_report(report)
        return report
    
    def requires_review(self, artifact_path: str) -> bool:
        """Check if an artifact requires code review."""
        required_patterns = self.config.get("review_required_for", [])
        optional_patterns = self.config.get("review_optional_for", [])
        
        path = Path(artifact_path)
        suffix = f"*{path.suffix}"
        
        # Check required patterns
        for pattern in required_patterns:
            if self._matches_pattern(artifact_path, pattern):
                return True
        
        # Not in required list
        return False
    
    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches glob pattern."""
        from fnmatch import fnmatch
        return fnmatch(path, pattern) or fnmatch(Path(path).name, pattern)


class ReviewPromptGenerator:
    """
    Generates prompts for AI agents to conduct code reviews.
    
    This class creates structured prompts that guide the reviewer
    through the four-dimensional review process.
    """
    
    @staticmethod
    def generate_quality_review_prompt(
        code_content: str,
        file_path: str,
        taskcard_content: str,
        checks: List[Dict[str, Any]],
    ) -> str:
        """Generate prompt for quality review."""
        check_list = "\n".join([
            f"- {c['id']}: {c['name'].replace('_', ' ').title()}"
            for c in checks
        ])
        
        return f"""You are a Senior Code Reviewer conducting a **Quality Check** (Q-Check).

## Code to Review
File: `{file_path}`
```
{code_content}
```

## Task Context
{taskcard_content}

## Quality Checklist
{check_list}

## Instructions
For each issue found, provide:
1. Check ID (e.g., Q-001)
2. Severity: CRITICAL / MAJOR / MINOR / SUGGESTION
3. Line number(s) if applicable
4. Clear description of the issue
5. Suggested fix

Focus on:
- Syntax correctness
- Type safety
- Error handling
- Security vulnerabilities
- Performance anti-patterns
- Code duplication

Output your findings in structured format."""

    @staticmethod
    def generate_requirements_review_prompt(
        code_content: str,
        file_path: str,
        requirements: str,
        spec_content: str,
        checks: List[Dict[str, Any]],
    ) -> str:
        """Generate prompt for requirements review."""
        check_list = "\n".join([
            f"- {c['id']}: {c['name'].replace('_', ' ').title()}"
            for c in checks
        ])
        
        return f"""You are a Senior Code Reviewer conducting a **Requirements Check** (R-Check).

## Code to Review
File: `{file_path}`
```
{code_content}
```

## Requirements from TaskCard
{requirements}

## Referenced Specifications
{spec_content}

## Requirements Checklist
{check_list}

## Instructions
Verify each requirement is correctly implemented.

For each issue found, provide:
1. Check ID (e.g., R-001)
2. Severity: CRITICAL / MAJOR / MINOR
3. Requirement that is violated/missing
4. Description of the gap
5. What needs to be done to fix it

Calculate requirements coverage percentage.

Output your findings in structured format."""

    @staticmethod
    def generate_completeness_review_prompt(
        code_content: str,
        file_path: str,
        taskcard_content: str,
        acceptance_criteria: str,
        checks: List[Dict[str, Any]],
    ) -> str:
        """Generate prompt for completeness review."""
        check_list = "\n".join([
            f"- {c['id']}: {c['name'].replace('_', ' ').title()}"
            for c in checks
        ])
        
        return f"""You are a Senior Code Reviewer conducting a **Completeness Check** (C-Check).

## Code to Review
File: `{file_path}`
```
{code_content}
```

## TaskCard Content
{taskcard_content}

## Acceptance Criteria
{acceptance_criteria}

## Completeness Checklist
{check_list}

## Instructions
Ensure all requirements are addressed, not just partially implemented.

For each missing item, provide:
1. Check ID (e.g., C-001)
2. Severity: CRITICAL / MAJOR / MINOR
3. What is missing
4. What needs to be added

Calculate completeness percentage.

Output your findings in structured format."""

    @staticmethod
    def generate_optimization_review_prompt(
        code_content: str,
        file_path: str,
        language: str,
        checks: List[Dict[str, Any]],
    ) -> str:
        """Generate prompt for optimization review."""
        check_list = "\n".join([
            f"- {c['id']}: {c['name'].replace('_', ' ').title()}"
            for c in checks
        ])
        
        return f"""You are a Senior Code Reviewer conducting an **Optimization Review** (O-Check).

## Code to Review
File: `{file_path}`
Language: {language}
```
{code_content}
```

## Optimization Checklist
{check_list}

## Instructions
Identify opportunities to improve the code while maintaining functionality.

For each suggestion, provide:
1. Check ID (e.g., O-001)
2. Description of the improvement
3. Current code snippet
4. Suggested improved code
5. Rationale
6. Estimated impact: minor / moderate / significant

Focus on:
- Code simplification
- Algorithm efficiency
- Better abstractions
- Naming clarity
- Idiomatic {language} patterns
- Complexity reduction

Note: These are SUGGESTIONS, not blocking issues.

Output your findings in structured format."""

    @staticmethod
    def generate_persona_prompt(
        persona: str,
        persona_config: Dict[str, Any],
        code_content: str,
        file_path: str,
    ) -> str:
        """Generate prompt for a specific expert persona."""
        focus_checks = persona_config.get("focus", [])
        description = persona_config.get("description", "")
        
        return f"""You are a **{persona.replace('_', ' ').title()}** reviewing code.

## Your Expertise
{description}

## Focus Areas
{', '.join(focus_checks)}

## Code to Review
File: `{file_path}`
```
{code_content}
```

## Instructions
Apply your specialized expertise to identify issues in your focus areas.
Be thorough and specific.

For each finding, provide:
1. Related Check ID
2. Severity
3. Technical details
4. Remediation steps

Output your expert findings in structured format."""


# Convenience functions for module-level access

def create_review_engine(config_path: Optional[Path] = None) -> CodeReviewEngine:
    """Create and return a CodeReviewEngine instance."""
    return CodeReviewEngine(config_path)


def get_default_engine() -> CodeReviewEngine:
    """Get or create the default CodeReviewEngine."""
    global _default_engine
    if "_default_engine" not in globals() or _default_engine is None:
        _default_engine = CodeReviewEngine()
    return _default_engine


_default_engine: Optional[CodeReviewEngine] = None
