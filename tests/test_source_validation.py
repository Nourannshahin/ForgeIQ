import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from validate_sources import validate_source_citations


retrieved_chunks = [
    {
        "source": "maintenance_manual.txt",
        "section": "4. Hydraulic System Monitoring",
    },
    {
        "source": "inspection_checklist.txt",
        "section": "4. Hydraulic System Observations",
    },
]

cited_sources = [
    {
        "source": "maintenance_manual.txt",
        "section": "4. Hydraulic System Monitoring",
    },
    {
        "source": "unknown_manual.txt",
        "section": "2. Pressure Limits",
    },
]

report = validate_source_citations(cited_sources, retrieved_chunks)

print("SOURCE VALIDATION TEST")
print("=" * 35)
print(f"Total citations: {report['total_citations']}")
print(f"Valid citations: {report['valid_citations']}")
print(f"Invalid citations: {report['invalid_citations']}")
print(f"All citations valid: {report['all_valid']}")

for citation in report["citations"]:
    print(f"\n{citation['source']} | {citation['section']}")
    print(f"Valid: {citation['valid']}")
    print(f"Reason: {citation['reason']}")

assert report["total_citations"] == 2
assert report["valid_citations"] == 1
assert report["invalid_citations"] == 1
assert report["all_valid"] is False

print("\nAll validation tests passed.")