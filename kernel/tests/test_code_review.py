"""
Tests for Code Review Engine (Pair Programming)

Tests the code_review.py module functionality.
"""

import pytest
from datetime import datetime, timezone
from pathlib import Path

# Add kernel to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code_review import (
    CodeReviewEngine,
    ReviewSession,
    ReviewReport,
    ReviewIssue,
    OptimizationSuggestion,
    DimensionResult,
    IssueSeverity,
    ReviewVerdict,
    ReviewDimension,
    ReviewPromptGenerator,
)


class TestIssueSeverity:
    """Test issue severity enumeration."""
    
    def test_severity_values(self):
        """Test severity enum values."""
        assert IssueSeverity.CRITICAL.value == "CRITICAL"
        assert IssueSeverity.MAJOR.value == "MAJOR"
        assert IssueSeverity.MINOR.value == "MINOR"
        assert IssueSeverity.SUGGESTION.value == "SUGGESTION"


class TestReviewVerdict:
    """Test review verdict enumeration."""
    
    def test_verdict_values(self):
        """Test verdict enum values."""
        assert ReviewVerdict.APPROVED.value == "APPROVED"
        assert ReviewVerdict.NEEDS_REVISION.value == "NEEDS_REVISION"
        assert ReviewVerdict.BLOCKED.value == "BLOCKED"


class TestReviewIssue:
    """Test ReviewIssue dataclass."""
    
    def test_create_issue(self):
        """Test creating a review issue."""
        issue = ReviewIssue(
            issue_id="ISS-001",
            dimension=ReviewDimension.QUALITY,
            check_id="Q-001",
            severity=IssueSeverity.MAJOR,
            description="Type annotation missing",
            file_path="kernel/test.py",
            line_start=10,
            suggested_fix="Add type hints",
        )
        
        assert issue.issue_id == "ISS-001"
        assert issue.dimension == ReviewDimension.QUALITY
        assert issue.severity == IssueSeverity.MAJOR
        assert issue.line_start == 10
    
    def test_issue_to_dict(self):
        """Test converting issue to dictionary."""
        issue = ReviewIssue(
            issue_id="ISS-002",
            dimension=ReviewDimension.REQUIREMENTS,
            check_id="R-001",
            severity=IssueSeverity.CRITICAL,
            description="Missing validation",
            file_path="kernel/module.py",
        )
        
        result = issue.to_dict()
        
        assert result["issue_id"] == "ISS-002"
        assert result["dimension"] == "requirements_check"
        assert result["severity"] == "CRITICAL"
        assert result["check_id"] == "R-001"


class TestOptimizationSuggestion:
    """Test OptimizationSuggestion dataclass."""
    
    def test_create_suggestion(self):
        """Test creating an optimization suggestion."""
        suggestion = OptimizationSuggestion(
            suggestion_id="OPT-001",
            check_id="O-002",
            description="Use list comprehension",
            file_path="kernel/utils.py",
            current_code="result = []\nfor x in items:\n    result.append(x*2)",
            suggested_code="result = [x*2 for x in items]",
            rationale="More Pythonic and efficient",
            estimated_impact="minor",
        )
        
        assert suggestion.suggestion_id == "OPT-001"
        assert suggestion.estimated_impact == "minor"
    
    def test_suggestion_to_dict(self):
        """Test converting suggestion to dictionary."""
        suggestion = OptimizationSuggestion(
            suggestion_id="OPT-002",
            check_id="O-001",
            description="Simplify conditional",
            file_path="kernel/logic.py",
        )
        
        result = suggestion.to_dict()
        
        assert result["suggestion_id"] == "OPT-002"
        assert result["check_id"] == "O-001"


class TestDimensionResult:
    """Test DimensionResult dataclass."""
    
    def test_passed_dimension(self):
        """Test a dimension that passed."""
        result = DimensionResult(
            dimension=ReviewDimension.QUALITY,
            passed=True,
            issues=[],
            suggestions=[],
        )
        
        assert result.passed is True
        assert len(result.issues) == 0
    
    def test_failed_dimension(self):
        """Test a dimension that failed."""
        issue = ReviewIssue(
            issue_id="ISS-003",
            dimension=ReviewDimension.QUALITY,
            check_id="Q-006",
            severity=IssueSeverity.CRITICAL,
            description="SQL injection vulnerability",
            file_path="kernel/db.py",
        )
        
        result = DimensionResult(
            dimension=ReviewDimension.QUALITY,
            passed=False,
            issues=[issue],
        )
        
        assert result.passed is False
        assert len(result.issues) == 1
    
    def test_dimension_to_dict(self):
        """Test converting dimension result to dictionary."""
        result = DimensionResult(
            dimension=ReviewDimension.REQUIREMENTS,
            passed=True,
            coverage_percentage=95.5,
        )
        
        data = result.to_dict()
        
        assert data["dimension"] == "requirements_check"
        assert data["passed"] is True
        assert data["coverage_percentage"] == 95.5


class TestReviewSession:
    """Test ReviewSession class."""
    
    def test_create_session(self):
        """Test creating a review session."""
        session = ReviewSession(
            session_id="RSESS-TEST-001",
            task_id="TASK_TEST_001",
            reviewer_agent_id="reviewer-agent-001",
            builder_agent_id="builder-agent-001",
            artifacts=["kernel/test.py"],
            taskcard_content="# Test Task\n\nSummary here",
            spec_references=["SPEC_001"],
        )
        
        assert session.session_id == "RSESS-TEST-001"
        assert session.task_id == "TASK_TEST_001"
        assert session.reviewer_agent_id != session.builder_agent_id
        assert session.revision_number == 1
    
    def test_self_review_prohibited(self):
        """Test that self-review raises error."""
        with pytest.raises(ValueError, match="Self-review prohibited"):
            ReviewSession(
                session_id="RSESS-TEST-002",
                task_id="TASK_TEST_002",
                reviewer_agent_id="same-agent",
                builder_agent_id="same-agent",  # Same as reviewer
                artifacts=["test.py"],
                taskcard_content="Test",
                spec_references=[],
            )
    
    def test_generate_report_id(self):
        """Test report ID generation."""
        session = ReviewSession(
            session_id="RSESS-TEST-003",
            task_id="TASK_TEST_003",
            reviewer_agent_id="reviewer",
            builder_agent_id="builder",
            artifacts=[],
            taskcard_content="",
            spec_references=[],
        )
        
        report_id = session.generate_report_id()
        
        assert report_id.startswith("REV-TASK_TEST_003-")
        assert "-R1" in report_id
    
    def test_increment_revision(self):
        """Test revision number increment."""
        session = ReviewSession(
            session_id="RSESS-TEST-004",
            task_id="TASK_TEST_004",
            reviewer_agent_id="reviewer",
            builder_agent_id="builder",
            artifacts=[],
            taskcard_content="",
            spec_references=[],
        )
        
        assert session.revision_number == 1
        new_rev = session.increment_revision()
        assert new_rev == 2
        assert session.revision_number == 2


class TestCodeReviewEngine:
    """Test CodeReviewEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create a CodeReviewEngine instance."""
        return CodeReviewEngine()
    
    def test_create_session(self, engine):
        """Test creating a review session via engine."""
        session = engine.create_session(
            task_id="TASK_ENGINE_001",
            reviewer_agent_id="reviewer-engine",
            builder_agent_id="builder-engine",
            artifacts=["test.py"],
            taskcard_content="# Test\n\nContent",
        )
        
        assert session is not None
        assert session.task_id == "TASK_ENGINE_001"
    
    def test_create_session_self_review_prohibited(self, engine):
        """Test that engine prohibits self-review."""
        with pytest.raises(ValueError, match="Self-review prohibited"):
            engine.create_session(
                task_id="TASK_SELF_REVIEW",
                reviewer_agent_id="same-agent",
                builder_agent_id="same-agent",
                artifacts=[],
                taskcard_content="",
            )
    
    def test_determine_verdict_approved(self, engine):
        """Test verdict is APPROVED when no critical/major issues."""
        issues = [
            ReviewIssue(
                issue_id="1",
                dimension=ReviewDimension.QUALITY,
                check_id="Q-008",
                severity=IssueSeverity.MINOR,
                description="Minor issue",
                file_path="test.py",
            ),
        ]
        
        verdict = engine.determine_verdict(issues)
        assert verdict == ReviewVerdict.APPROVED
    
    def test_determine_verdict_needs_revision_critical(self, engine):
        """Test verdict is NEEDS_REVISION when critical issues exist."""
        issues = [
            ReviewIssue(
                issue_id="1",
                dimension=ReviewDimension.QUALITY,
                check_id="Q-006",
                severity=IssueSeverity.CRITICAL,
                description="Security vulnerability",
                file_path="test.py",
            ),
        ]
        
        verdict = engine.determine_verdict(issues)
        assert verdict == ReviewVerdict.NEEDS_REVISION
    
    def test_determine_verdict_needs_revision_major(self, engine):
        """Test verdict is NEEDS_REVISION when major issues exist."""
        issues = [
            ReviewIssue(
                issue_id="1",
                dimension=ReviewDimension.REQUIREMENTS,
                check_id="R-001",
                severity=IssueSeverity.MAJOR,
                description="Missing requirement",
                file_path="test.py",
            ),
        ]
        
        verdict = engine.determine_verdict(issues)
        assert verdict == ReviewVerdict.NEEDS_REVISION
    
    def test_count_issues_by_severity(self, engine):
        """Test counting issues by severity."""
        issues = [
            ReviewIssue("1", ReviewDimension.QUALITY, "Q-001", IssueSeverity.CRITICAL, "desc", "file.py"),
            ReviewIssue("2", ReviewDimension.QUALITY, "Q-002", IssueSeverity.CRITICAL, "desc", "file.py"),
            ReviewIssue("3", ReviewDimension.REQUIREMENTS, "R-001", IssueSeverity.MAJOR, "desc", "file.py"),
            ReviewIssue("4", ReviewDimension.COMPLETENESS, "C-001", IssueSeverity.MINOR, "desc", "file.py"),
            ReviewIssue("5", ReviewDimension.QUALITY, "Q-007", IssueSeverity.SUGGESTION, "desc", "file.py"),
        ]
        
        critical, major, minor, suggestion = engine.count_issues_by_severity(issues)
        
        assert critical == 2
        assert major == 1
        assert minor == 1
        assert suggestion == 1


class TestReviewPromptGenerator:
    """Test ReviewPromptGenerator class."""
    
    def test_generate_quality_prompt(self):
        """Test generating quality review prompt."""
        prompt = ReviewPromptGenerator.generate_quality_review_prompt(
            code_content="def test(): pass",
            file_path="test.py",
            taskcard_content="# Task\n\nTest task",
            checks=[
                {"id": "Q-001", "name": "syntax_correctness"},
                {"id": "Q-002", "name": "type_safety"},
            ],
        )
        
        assert "Quality Check" in prompt
        assert "def test(): pass" in prompt
        assert "Q-001" in prompt
        assert "Q-002" in prompt
    
    def test_generate_requirements_prompt(self):
        """Test generating requirements review prompt."""
        prompt = ReviewPromptGenerator.generate_requirements_review_prompt(
            code_content="class Handler: pass",
            file_path="handler.py",
            requirements="Implement handler class",
            spec_content="SPEC-001",
            checks=[{"id": "R-001", "name": "functional_requirements"}],
        )
        
        assert "Requirements Check" in prompt
        assert "class Handler: pass" in prompt
        assert "R-001" in prompt
    
    def test_generate_optimization_prompt(self):
        """Test generating optimization review prompt."""
        prompt = ReviewPromptGenerator.generate_optimization_review_prompt(
            code_content="for i in range(len(items)): print(items[i])",
            file_path="loop.py",
            language="Python",
            checks=[{"id": "O-001", "name": "simplification_possible"}],
        )
        
        assert "Optimization Review" in prompt
        assert "Python" in prompt
        assert "O-001" in prompt
    
    def test_generate_persona_prompt(self):
        """Test generating expert persona prompt."""
        prompt = ReviewPromptGenerator.generate_persona_prompt(
            persona="security_expert",
            persona_config={
                "focus": ["Q-006", "R-002"],
                "description": "OWASP security expert",
            },
            code_content="password = 'hardcoded'",
            file_path="auth.py",
        )
        
        assert "Security Expert" in prompt
        assert "OWASP" in prompt
        assert "Q-006" in prompt


class TestReviewReportGeneration:
    """Test review report generation."""
    
    def test_create_review_report(self):
        """Test creating a complete review report."""
        engine = CodeReviewEngine()
        
        session = engine.create_session(
            task_id="TASK_REPORT_001",
            reviewer_agent_id="reviewer",
            builder_agent_id="builder",
            artifacts=["test.py"],
            taskcard_content="Test task",
        )
        
        # Create dimension results
        quality_result = DimensionResult(
            dimension=ReviewDimension.QUALITY,
            passed=True,
            issues=[],
        )
        requirements_result = DimensionResult(
            dimension=ReviewDimension.REQUIREMENTS,
            passed=True,
            issues=[],
            coverage_percentage=100.0,
        )
        completeness_result = DimensionResult(
            dimension=ReviewDimension.COMPLETENESS,
            passed=True,
            issues=[],
        )
        optimization_result = DimensionResult(
            dimension=ReviewDimension.OPTIMIZATION,
            passed=True,
            suggestions=[],
        )
        
        report = engine.create_review_report(
            session=session,
            quality_result=quality_result,
            requirements_result=requirements_result,
            completeness_result=completeness_result,
            optimization_result=optimization_result,
        )
        
        assert report.verdict == ReviewVerdict.APPROVED
        assert report.critical_count == 0
        assert report.major_count == 0
    
    def test_report_to_yaml(self):
        """Test exporting report to YAML."""
        engine = CodeReviewEngine()
        
        session = engine.create_session(
            task_id="TASK_YAML_001",
            reviewer_agent_id="reviewer",
            builder_agent_id="builder",
            artifacts=[],
            taskcard_content="",
        )
        
        report = engine.create_review_report(
            session=session,
            quality_result=DimensionResult(ReviewDimension.QUALITY, True),
            requirements_result=DimensionResult(ReviewDimension.REQUIREMENTS, True),
            completeness_result=DimensionResult(ReviewDimension.COMPLETENESS, True),
            optimization_result=DimensionResult(ReviewDimension.OPTIMIZATION, True),
        )
        
        yaml_output = report.to_yaml()
        
        assert "report_id:" in yaml_output
        assert "verdict: APPROVED" in yaml_output
    
    def test_report_to_markdown(self):
        """Test exporting report to Markdown."""
        engine = CodeReviewEngine()
        
        session = engine.create_session(
            task_id="TASK_MD_001",
            reviewer_agent_id="reviewer",
            builder_agent_id="builder",
            artifacts=["test.py"],
            taskcard_content="",
        )
        
        # Add an issue
        issue = ReviewIssue(
            issue_id="ISS-MD-001",
            dimension=ReviewDimension.QUALITY,
            check_id="Q-003",
            severity=IssueSeverity.MAJOR,
            description="Missing error handling",
            file_path="test.py",
            line_start=15,
            suggested_fix="Add try-except block",
        )
        
        quality_result = DimensionResult(
            dimension=ReviewDimension.QUALITY,
            passed=False,
            issues=[issue],
        )
        
        report = engine.create_review_report(
            session=session,
            quality_result=quality_result,
            requirements_result=DimensionResult(ReviewDimension.REQUIREMENTS, True),
            completeness_result=DimensionResult(ReviewDimension.COMPLETENESS, True),
            optimization_result=DimensionResult(ReviewDimension.OPTIMIZATION, True),
        )
        
        md_output = report.to_markdown()
        
        assert "# Code Review Report" in md_output
        assert "NEEDS_REVISION" in md_output
        assert "MAJOR" in md_output
        assert "Q-003" in md_output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
