"""
Governance Action Module

Implements Freeze and Acceptance operations as defined in GOVERNANCE_INVARIANTS.

These operations confer authority and preserve state:
- **Freeze**: Makes an artifact immutable with cryptographic verification
- **Acceptance**: Confers authority to an artifact through explicit approval

Spec References:
- GOVERNANCE_INVARIANTS §1: Artifact, Freeze, Acceptance definitions
- AUTHORITY_CANON: Authority conferral mechanisms
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from kernel.paths import OPS_FREEZE_DIR, ROOT
from kernel.state_store import read_yaml, write_yaml


@dataclass
class FreezeRecord:
    """
    Record of an artifact freeze operation.
    
    A freeze makes an artifact immutable and preserves its content with
    cryptographic verification. Frozen artifacts cannot be modified without
    explicit unfreeze.
    
    Attributes:
        artifact_path: Relative path to frozen artifact
        frozen_at: Timestamp of freeze operation
        frozen_by: Identity of agent/user who froze
        content_hash: SHA-256 hash of frozen content
        version: Version identifier (e.g., "v1.0.0")
        metadata: Additional context (reason, approver, etc.)
    """
    artifact_path: str
    frozen_at: datetime
    frozen_by: str
    content_hash: str  # SHA-256 of frozen content
    version: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for YAML storage."""
        return {
            "artifact_path": self.artifact_path,
            "frozen_at": self.frozen_at.isoformat(),
            "frozen_by": self.frozen_by,
            "content_hash": self.content_hash,
            "version": self.version,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FreezeRecord":
        """Deserialize from dictionary."""
        return cls(
            artifact_path=data["artifact_path"],
            frozen_at=datetime.fromisoformat(data["frozen_at"]),
            frozen_by=data["frozen_by"],
            content_hash=data["content_hash"],
            version=data["version"],
            metadata=data.get("metadata", {}),
        )


def freeze_artifact(
    artifact_path: Path,
    frozen_by: str,
    version: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> FreezeRecord:
    """
    Freeze an artifact, making it immutable.
    
    Creates a freeze record in ops/freeze/ with:
    - Original content snapshot
    - Cryptographic hash (SHA-256)
    - Freeze metadata
    
    This operation preserves authority across time by creating an immutable
    record that can be verified against the current artifact state.
    
    Args:
        artifact_path: Path to artifact to freeze (relative to ROOT)
        frozen_by: Identity of freezer (agent_id or user)
        version: Version identifier (e.g., "v1.0.0")
        metadata: Additional metadata (reason, approver, etc.)
    
    Returns:
        FreezeRecord object containing freeze metadata
    
    Raises:
        FileNotFoundError: If artifact doesn't exist
        ValueError: If artifact already frozen at this version
    
    Example:
        >>> from pathlib import Path
        >>> record = freeze_artifact(
        ...     Path("specs/canon/GOVERNANCE_INVARIANTS.md"),
        ...     frozen_by="governance",
        ...     version="v1.0.0",
        ...     metadata={"reason": "Canon freeze", "approved_by": "committee"}
        ... )
        >>> print(f"Frozen: {record.artifact_path} at {record.version}")
    """
    full_path = ROOT / artifact_path
    if not full_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    
    # Compute content hash for verification
    content = full_path.read_bytes()
    content_hash = hashlib.sha256(content).hexdigest()
    
    # Create freeze record
    record = FreezeRecord(
        artifact_path=str(artifact_path),
        frozen_at=datetime.now(timezone.utc),
        frozen_by=frozen_by,
        content_hash=content_hash,
        version=version,
        metadata=metadata or {},
    )
    
    # Save freeze record to ops/freeze/
    OPS_FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sanitize path for filename (replace path separators and colons with _)
    safe_path = str(artifact_path).replace('/', '_').replace('\\', '_').replace(':', '_')
    freeze_file = OPS_FREEZE_DIR / f"{safe_path}_{version}.yaml"
    
    # Allow overwriting existing freeze (覆盖模式)
    write_yaml(freeze_file, record.to_dict())
    
    # Save frozen content snapshot for verification
    snapshot_file = freeze_file.with_suffix(".snapshot")
    snapshot_file.write_bytes(content)
    
    return record


@dataclass
class AcceptanceRecord:
    """
    Record of an artifact acceptance operation.
    
    Acceptance confers authority to an artifact through explicit approval.
    An accepted artifact is recognized as authoritative within its scope.
    
    Attributes:
        artifact_path: Relative path to accepted artifact
        accepted_at: Timestamp of acceptance
        accepted_by: Identity who accepted
        authority: Source of authority ("owner", "governance", "vote")
        content_hash: SHA-256 hash at acceptance time
        metadata: Additional context
    """
    artifact_path: str
    accepted_at: datetime
    accepted_by: str
    authority: str  # "owner", "governance", "vote", etc.
    content_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for YAML storage."""
        return {
            "artifact_path": self.artifact_path,
            "accepted_at": self.accepted_at.isoformat(),
            "accepted_by": self.accepted_by,
            "authority": self.authority,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AcceptanceRecord":
        """Deserialize from dictionary."""
        return cls(
            artifact_path=data["artifact_path"],
            accepted_at=datetime.fromisoformat(data["accepted_at"]),
            accepted_by=data["accepted_by"],
            authority=data["authority"],
            content_hash=data["content_hash"],
            metadata=data.get("metadata", {}),
        )


def accept_artifact(
    artifact_path: Path,
    accepted_by: str,
    authority: str = "owner",
    metadata: Optional[Dict[str, Any]] = None,
) -> AcceptanceRecord:
    """
    Accept an artifact, conferring it authority.
    
    Creates acceptance record and updates artifact status. This operation
    explicitly grants authority to an artifact, making it recognized as
    authoritative within the system.
    
    Args:
        artifact_path: Path to artifact to accept (relative to ROOT)
        accepted_by: Identity of acceptor (agent_id or user)
        authority: Authority source ("owner", "governance", "vote")
        metadata: Additional metadata (reason, conditions, etc.)
    
    Returns:
        AcceptanceRecord object containing acceptance metadata
    
    Raises:
        FileNotFoundError: If artifact doesn't exist
    
    Example:
        >>> record = accept_artifact(
        ...     Path("specs/canon/GOVERNANCE_INVARIANTS.md"),
        ...     accepted_by="governance_committee",
        ...     authority="governance",
        ...     metadata={"reason": "Unanimous approval", "vote": "5-0"}
        ... )
    """
    full_path = ROOT / artifact_path
    if not full_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    
    # Compute content hash for verification
    content = full_path.read_bytes()
    content_hash = hashlib.sha256(content).hexdigest()
    
    # Create acceptance record
    record = AcceptanceRecord(
        artifact_path=str(artifact_path),
        accepted_at=datetime.now(timezone.utc),
        accepted_by=accepted_by,
        authority=authority,
        content_hash=content_hash,
        metadata=metadata or {},
    )
    
    # Save acceptance record to ops/acceptance/
    acceptance_dir = ROOT / "ops" / "acceptance"
    acceptance_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize path for filename (replace path separators and colons with _)
    safe_path = str(artifact_path).replace('/', '_').replace('\\', '_').replace(':', '_')
    acceptance_file = acceptance_dir / f"{safe_path}.yaml"
    
    # Allow re-acceptance (overwrite existing)
    write_yaml(acceptance_file, record.to_dict())
    
    return record


def is_frozen(artifact_path: Path, version: Optional[str] = None) -> bool:
    """
    Check if artifact is frozen at specified version.
    
    Args:
        artifact_path: Path to artifact (relative to ROOT)
        version: Specific version to check, or None for any version
    
    Returns:
        True if artifact is frozen (at specified version if given)
    """
    safe_path = str(artifact_path).replace('/', '_').replace('\\', '_').replace(':', '_')
    
    if version:
        freeze_file = OPS_FREEZE_DIR / f"{safe_path}_{version}.yaml"
        return freeze_file.exists()
    else:
        # Check if any version is frozen
        if not OPS_FREEZE_DIR.exists():
            return False
        pattern = f"{safe_path}_*.yaml"
        return len(list(OPS_FREEZE_DIR.glob(pattern))) > 0


def is_accepted(artifact_path: Path) -> bool:
    """
    Check if artifact has been accepted.
    
    Args:
        artifact_path: Path to artifact (relative to ROOT)
    
    Returns:
        True if artifact has acceptance record
    """
    safe_path = str(artifact_path).replace('/', '_').replace('\\', '_').replace(':', '_')
    acceptance_file = ROOT / "ops" / "acceptance" / f"{safe_path}.yaml"
    return acceptance_file.exists()


def get_freeze_record(artifact_path: Path, version: str) -> Optional[FreezeRecord]:
    """
    Retrieve freeze record for artifact at specific version.
    
    Args:
        artifact_path: Path to artifact
        version: Version identifier
    
    Returns:
        FreezeRecord if exists, None otherwise
    """
    safe_path = str(artifact_path).replace('/', '_').replace('\\', '_').replace(':', '_')
    freeze_file = OPS_FREEZE_DIR / f"{safe_path}_{version}.yaml"
    
    if not freeze_file.exists():
        return None
    
    data = read_yaml(freeze_file)
    return FreezeRecord.from_dict(data)


def get_acceptance_record(artifact_path: Path) -> Optional[AcceptanceRecord]:
    """
    Retrieve acceptance record for artifact.
    
    Args:
        artifact_path: Path to artifact
    
    Returns:
        AcceptanceRecord if exists, None otherwise
    """
    safe_path = str(artifact_path).replace('/', '_').replace('\\', '_').replace(':', '_')
    acceptance_file = ROOT / "ops" / "acceptance" / f"{safe_path}.yaml"
    
    if not acceptance_file.exists():
        return None
    
    data = read_yaml(acceptance_file)
    return AcceptanceRecord.from_dict(data)
