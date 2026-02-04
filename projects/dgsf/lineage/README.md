# DGSF Lineage Tracking
# This folder tracks spec changes and their impact on experiments

# Files:
# - spec_changes.yaml: Log of all spec changes with timestamps and affected experiments

# Format:
# changes:
#   - id: SCH-YYYYMMDD-HHMMSS
#     spec_path: path/to/spec.yaml
#     change_type: add | modify | deprecate
#     timestamp: ISO-8601
#     git_commit: hash (if committed)
#     affected_experiments: [list of experiment IDs]
