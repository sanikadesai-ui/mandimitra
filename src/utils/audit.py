"""
Audit logging utilities for MANDIMITRA.
Creates detailed audit summaries in Markdown format.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class AuditLogger:
    """
    Generate audit summary reports in Markdown format.
    
    Example:
        >>> audit = AuditLogger("mandi_download", Path("logs"))
        >>> audit.add_section("API Configuration", {"endpoint": "...", "state": "Maharashtra"})
        >>> audit.add_metric("Total Rows", 50000)
        >>> audit.save()
    """
    
    def __init__(
        self,
        operation_name: str,
        output_dir: Path,
        timestamp: Optional[str] = None,
    ):
        """
        Initialize audit logger.
        
        Args:
            operation_name: Name of the operation being audited
            output_dir: Directory for audit files
            timestamp: Optional timestamp (auto-generated if None)
        """
        self.operation_name = operation_name
        self.output_dir = Path(output_dir)
        self.timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        self.sections: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.start_time = datetime.now(timezone.utc)
    
    def add_section(self, title: str, content: Dict[str, Any]) -> None:
        """Add a section to the audit report."""
        self.sections.append({"title": title, "content": content})
    
    def add_metric(self, name: str, value: Any) -> None:
        """Add a metric to the summary."""
        self.metrics[name] = value
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
    
    def _format_dict(self, d: Dict[str, Any], indent: int = 0) -> str:
        """Format dictionary for Markdown display."""
        lines = []
        prefix = "  " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}- **{key}**:")
                lines.append(self._format_dict(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}- **{key}**: {len(value)} items")
                if len(value) <= 10:
                    for item in value:
                        lines.append(f"{prefix}  - {item}")
                else:
                    for item in value[:5]:
                        lines.append(f"{prefix}  - {item}")
                    lines.append(f"{prefix}  - ... and {len(value) - 5} more")
            else:
                lines.append(f"{prefix}- **{key}**: {value}")
        return "\n".join(lines)
    
    def generate_markdown(self) -> str:
        """Generate the full Markdown audit report."""
        end_time = datetime.now(timezone.utc)
        duration = (end_time - self.start_time).total_seconds()
        
        lines = [
            f"# MANDIMITRA Audit Report: {self.operation_name}",
            "",
            "---",
            "",
            "## Overview",
            "",
            f"- **Operation**: {self.operation_name}",
            f"- **Timestamp**: {self.timestamp}",
            f"- **Start Time (UTC)**: {self.start_time.isoformat()}",
            f"- **End Time (UTC)**: {end_time.isoformat()}",
            f"- **Duration**: {duration:.2f} seconds",
            f"- **Status**: {'⚠️ WARNINGS' if self.warnings else '❌ ERRORS' if self.errors else '✅ SUCCESS'}",
            "",
            "---",
            "",
        ]
        
        # Summary metrics
        if self.metrics:
            lines.append("## Summary Metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for name, value in self.metrics.items():
                if isinstance(value, (int, float)) and value >= 1000:
                    formatted = f"{value:,.0f}" if isinstance(value, int) else f"{value:,.2f}"
                else:
                    formatted = str(value)
                lines.append(f"| {name} | {formatted} |")
            lines.append("")
            lines.append("---")
            lines.append("")
        
        # Sections
        for section in self.sections:
            lines.append(f"## {section['title']}")
            lines.append("")
            lines.append(self._format_dict(section["content"]))
            lines.append("")
            lines.append("---")
            lines.append("")
        
        # Warnings
        if self.warnings:
            lines.append("## ⚠️ Warnings")
            lines.append("")
            for warning in self.warnings:
                lines.append(f"- {warning}")
            lines.append("")
            lines.append("---")
            lines.append("")
        
        # Errors
        if self.errors:
            lines.append("## ❌ Errors")
            lines.append("")
            for error in self.errors:
                lines.append(f"- {error}")
            lines.append("")
            lines.append("---")
            lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Generated by MANDIMITRA Data Pipeline*")
        lines.append(f"*Maharashtra-Only | Version 2.0.0*")
        
        return "\n".join(lines)
    
    def save(self) -> Path:
        """Save audit report to file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"maharashtra_{self.operation_name}_{self.timestamp}.md"
        output_path = self.output_dir / filename
        
        content = self.generate_markdown()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        return output_path
