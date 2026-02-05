"""
Progress tracking and resumability utilities for MANDIMITRA.
Enables safe chunked downloads with resume capability.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from enum import Enum


class ChunkStatus(str, Enum):
    """Status of a download chunk."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProgressTracker:
    """
    Track download progress for resumable operations.
    
    Stores progress in a JSON file for crash recovery and resumability.
    
    Example:
        >>> tracker = ProgressTracker(Path("data/metadata/progress.json"))
        >>> tracker.start_session("mandi_download", chunks=["Pune", "Nashik", "Nagpur"])
        >>> tracker.mark_completed("Pune", rows=5000)
        >>> tracker.save()
    """
    
    def __init__(self, progress_file: Path):
        """
        Initialize progress tracker.
        
        Args:
            progress_file: Path to JSON file for storing progress
        """
        self.progress_file = Path(progress_file)
        self.data: Dict[str, Any] = {}
        self._load()
    
    def _load(self) -> None:
        """Load existing progress from file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.data = {}
        else:
            self.data = {}
    
    def save(self) -> None:
        """Save progress to file."""
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Update last saved timestamp
        self.data["last_saved_utc"] = datetime.now(timezone.utc).isoformat()
        
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def start_session(
        self,
        session_type: str,
        chunks: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        force_restart: bool = False,
    ) -> None:
        """
        Start a new download session or resume existing one.
        
        Args:
            session_type: Type of session (e.g., "mandi_download")
            chunks: List of chunk identifiers to process
            metadata: Additional metadata to store
            force_restart: If True, discard previous progress
        """
        if force_restart or session_type not in self.data:
            self.data[session_type] = {
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "total_chunks": len(chunks),
                "chunks": {
                    chunk: {"status": ChunkStatus.PENDING, "rows": 0, "attempts": 0}
                    for chunk in chunks
                },
                "metadata": metadata or {},
                "completed_count": 0,
                "failed_count": 0,
                "total_rows": 0,
            }
        else:
            # Resume: add any new chunks
            existing_chunks = set(self.data[session_type]["chunks"].keys())
            for chunk in chunks:
                if chunk not in existing_chunks:
                    self.data[session_type]["chunks"][chunk] = {
                        "status": ChunkStatus.PENDING,
                        "rows": 0,
                        "attempts": 0,
                    }
            self.data[session_type]["total_chunks"] = len(self.data[session_type]["chunks"])
        
        self.save()
    
    def get_pending_chunks(self, session_type: str) -> List[str]:
        """Get list of chunks not yet completed."""
        if session_type not in self.data:
            return []
        
        chunks = self.data[session_type].get("chunks", {})
        return [
            chunk_id
            for chunk_id, info in chunks.items()
            if info["status"] in (ChunkStatus.PENDING, ChunkStatus.FAILED)
        ]
    
    def get_completed_chunks(self, session_type: str) -> List[str]:
        """Get list of completed chunks."""
        if session_type not in self.data:
            return []
        
        chunks = self.data[session_type].get("chunks", {})
        return [
            chunk_id
            for chunk_id, info in chunks.items()
            if info["status"] == ChunkStatus.COMPLETED
        ]
    
    def mark_in_progress(self, session_type: str, chunk_id: str) -> None:
        """Mark a chunk as in progress."""
        if session_type in self.data and chunk_id in self.data[session_type]["chunks"]:
            chunk = self.data[session_type]["chunks"][chunk_id]
            chunk["status"] = ChunkStatus.IN_PROGRESS
            chunk["attempts"] = chunk.get("attempts", 0) + 1
            chunk["started_utc"] = datetime.now(timezone.utc).isoformat()
            self.save()
    
    def mark_completed(
        self,
        session_type: str,
        chunk_id: str,
        rows: int = 0,
        output_file: Optional[str] = None,
    ) -> None:
        """Mark a chunk as completed."""
        if session_type in self.data and chunk_id in self.data[session_type]["chunks"]:
            chunk = self.data[session_type]["chunks"][chunk_id]
            chunk["status"] = ChunkStatus.COMPLETED
            chunk["rows"] = rows
            chunk["completed_utc"] = datetime.now(timezone.utc).isoformat()
            if output_file:
                chunk["output_file"] = output_file
            
            # Update session totals
            session = self.data[session_type]
            session["completed_count"] = len(self.get_completed_chunks(session_type))
            session["total_rows"] = sum(
                c.get("rows", 0) for c in session["chunks"].values()
            )
            
            self.save()
    
    def mark_failed(
        self,
        session_type: str,
        chunk_id: str,
        error: str,
    ) -> None:
        """Mark a chunk as failed."""
        if session_type in self.data and chunk_id in self.data[session_type]["chunks"]:
            chunk = self.data[session_type]["chunks"][chunk_id]
            chunk["status"] = ChunkStatus.FAILED
            chunk["error"] = error
            chunk["failed_utc"] = datetime.now(timezone.utc).isoformat()
            
            # Update session totals
            session = self.data[session_type]
            session["failed_count"] = sum(
                1 for c in session["chunks"].values()
                if c["status"] == ChunkStatus.FAILED
            )
            
            self.save()
    
    def get_session_summary(self, session_type: str) -> Dict[str, Any]:
        """Get summary of session progress."""
        if session_type not in self.data:
            return {}
        
        session = self.data[session_type]
        completed = self.get_completed_chunks(session_type)
        pending = self.get_pending_chunks(session_type)
        
        return {
            "started": session.get("started_utc"),
            "total_chunks": session.get("total_chunks", 0),
            "completed_chunks": len(completed),
            "pending_chunks": len(pending),
            "failed_chunks": session.get("failed_count", 0),
            "total_rows": session.get("total_rows", 0),
            "is_complete": len(pending) == 0 and session.get("failed_count", 0) == 0,
        }
    
    def clear_session(self, session_type: str) -> None:
        """Clear all progress for a session."""
        if session_type in self.data:
            del self.data[session_type]
            self.save()
