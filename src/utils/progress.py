"""
Production-grade progress tracking for MANDIMITRA data pipeline.

Features:
- Batched saves to reduce I/O overhead
- Atomic writes (temp file + rename) to prevent corruption
- Thread-safe for concurrent operations
- Automatic flush on exit via atexit
- Resume capability with chunk-level granularity

Author: MANDIMITRA Team
Version: 2.0.0 (Production Refactor)
"""

import atexit
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class ChunkStatus(str, Enum):
    """Status of a download chunk."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProgressTracker:
    """
    Thread-safe progress tracker with batched saves and atomic writes.
    
    Optimizations:
    - Saves only every N updates (configurable batch_size)
    - Uses atomic write (temp file + rename) to prevent corruption
    - Thread-safe with internal locking
    - Auto-flushes on exit via atexit
    
    Example:
        >>> tracker = ProgressTracker(Path("progress.json"), batch_size=10)
        >>> tracker.start_session("download", chunks=["A", "B", "C"])
        >>> tracker.mark_completed("download", "A", rows=1000)
        >>> # Saves automatically after 10 updates or on exit
    """
    
    # Class-level registry for atexit cleanup
    _instances: List["ProgressTracker"] = []
    _atexit_registered: bool = False
    _class_lock: threading.Lock = threading.Lock()
    
    def __init__(
        self,
        progress_file: Path,
        batch_size: int = 10,
        auto_save: bool = True,
    ):
        """
        Initialize progress tracker.
        
        Args:
            progress_file: Path to JSON progress file
            batch_size: Number of updates before auto-save (0 = save every time)
            auto_save: Whether to auto-flush on exit
        """
        self.progress_file = Path(progress_file)
        self.batch_size = batch_size
        self.auto_save = auto_save
        
        self._data: Dict[str, Any] = {}
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._pending_updates = 0
        self._dirty = False
        
        # Load existing progress
        self._load()
        
        # Register for atexit cleanup
        if auto_save:
            self._register_for_cleanup()
    
    def _register_for_cleanup(self) -> None:
        """Register instance for atexit cleanup."""
        with ProgressTracker._class_lock:
            ProgressTracker._instances.append(self)
            
            if not ProgressTracker._atexit_registered:
                atexit.register(ProgressTracker._flush_all_instances)
                ProgressTracker._atexit_registered = True
    
    @classmethod
    def _flush_all_instances(cls) -> None:
        """Flush all registered instances (called at exit)."""
        with cls._class_lock:
            for instance in cls._instances:
                try:
                    instance.flush()
                except Exception:
                    pass  # Best effort on exit
    
    def _load(self) -> None:
        """Load existing progress from file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._data = {}
        else:
            self._data = {}
    
    def _atomic_save(self) -> None:
        """
        Save progress atomically using temp file + rename.
        
        This prevents corruption if the process is killed during save.
        """
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Update timestamp
        self._data["last_saved_utc"] = datetime.now(timezone.utc).isoformat()
        
        # Write to temp file first
        fd, temp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix="progress_",
            dir=self.progress_file.parent
        )
        
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename (on same filesystem)
            os.replace(temp_path, self.progress_file)
            
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise
    
    def _maybe_save(self) -> None:
        """Save if batch threshold reached."""
        if self.batch_size == 0:
            # Save every time
            self._atomic_save()
            self._pending_updates = 0
            self._dirty = False
        elif self._pending_updates >= self.batch_size:
            self._atomic_save()
            self._pending_updates = 0
            self._dirty = False
        else:
            self._dirty = True
    
    def flush(self) -> None:
        """Force save all pending changes."""
        with self._lock:
            if self._dirty or self._pending_updates > 0:
                self._atomic_save()
                self._pending_updates = 0
                self._dirty = False
    
    def save(self) -> None:
        """Explicit save (for backward compatibility)."""
        self.flush()
    
    def start_session(
        self,
        session_type: str,
        chunks: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        force_restart: bool = False,
    ) -> None:
        """
        Start a new session or resume existing one.
        
        Args:
            session_type: Session identifier (e.g., "mandi_download")
            chunks: List of chunk IDs to process
            metadata: Additional metadata to store
            force_restart: If True, discard previous progress
        """
        with self._lock:
            if force_restart or session_type not in self._data:
                self._data[session_type] = {
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                    "total_chunks": len(chunks),
                    "chunks": {
                        chunk: {
                            "status": ChunkStatus.PENDING.value,
                            "rows": 0,
                            "attempts": 0,
                        }
                        for chunk in chunks
                    },
                    "metadata": metadata or {},
                    "completed_count": 0,
                    "failed_count": 0,
                    "total_rows": 0,
                }
            else:
                # Resume: add any new chunks
                existing = set(self._data[session_type]["chunks"].keys())
                for chunk in chunks:
                    if chunk not in existing:
                        self._data[session_type]["chunks"][chunk] = {
                            "status": ChunkStatus.PENDING.value,
                            "rows": 0,
                            "attempts": 0,
                        }
                self._data[session_type]["total_chunks"] = len(self._data[session_type]["chunks"])
            
            self._pending_updates += 1
            self._maybe_save()
    
    def get_pending_chunks(self, session_type: str) -> List[str]:
        """Get list of chunks not yet completed."""
        with self._lock:
            if session_type not in self._data:
                return []
            
            chunks = self._data[session_type].get("chunks", {})
            return [
                chunk_id
                for chunk_id, info in chunks.items()
                if info.get("status") in (ChunkStatus.PENDING.value, ChunkStatus.FAILED.value)
            ]
    
    def get_completed_chunks(self, session_type: str) -> List[str]:
        """Get list of completed chunks."""
        with self._lock:
            if session_type not in self._data:
                return []
            
            chunks = self._data[session_type].get("chunks", {})
            return [
                chunk_id
                for chunk_id, info in chunks.items()
                if info.get("status") == ChunkStatus.COMPLETED.value
            ]
    
    def mark_in_progress(self, session_type: str, chunk_id: str) -> None:
        """Mark a chunk as in progress."""
        with self._lock:
            if session_type in self._data and chunk_id in self._data[session_type]["chunks"]:
                chunk = self._data[session_type]["chunks"][chunk_id]
                chunk["status"] = ChunkStatus.IN_PROGRESS.value
                chunk["attempts"] = chunk.get("attempts", 0) + 1
                chunk["started_utc"] = datetime.now(timezone.utc).isoformat()
                
                self._pending_updates += 1
                self._maybe_save()
    
    def mark_completed(
        self,
        session_type: str,
        chunk_id: str,
        rows: int = 0,
        output_file: Optional[str] = None,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """Mark a chunk as completed."""
        with self._lock:
            if session_type in self._data and chunk_id in self._data[session_type]["chunks"]:
                chunk = self._data[session_type]["chunks"][chunk_id]
                chunk["status"] = ChunkStatus.COMPLETED.value
                chunk["rows"] = rows
                chunk["completed_utc"] = datetime.now(timezone.utc).isoformat()
                
                if output_file:
                    chunk["output_file"] = output_file
                if duration_seconds is not None:
                    chunk["duration_seconds"] = duration_seconds
                
                # Update session totals
                self._update_session_totals(session_type)
                
                self._pending_updates += 1
                self._maybe_save()
    
    def mark_failed(
        self,
        session_type: str,
        chunk_id: str,
        error: str,
    ) -> None:
        """Mark a chunk as failed."""
        with self._lock:
            if session_type in self._data and chunk_id in self._data[session_type]["chunks"]:
                chunk = self._data[session_type]["chunks"][chunk_id]
                chunk["status"] = ChunkStatus.FAILED.value
                chunk["error"] = error
                chunk["failed_utc"] = datetime.now(timezone.utc).isoformat()
                
                # Update session totals
                self._update_session_totals(session_type)
                
                self._pending_updates += 1
                self._maybe_save()
    
    def _update_session_totals(self, session_type: str) -> None:
        """Update session-level totals (must be called with lock held)."""
        session = self._data[session_type]
        chunks = session["chunks"]
        
        session["completed_count"] = sum(
            1 for c in chunks.values()
            if c.get("status") == ChunkStatus.COMPLETED.value
        )
        session["failed_count"] = sum(
            1 for c in chunks.values()
            if c.get("status") == ChunkStatus.FAILED.value
        )
        session["total_rows"] = sum(
            c.get("rows", 0) for c in chunks.values()
        )
    
    def get_session_summary(self, session_type: str) -> Dict[str, Any]:
        """Get summary of session progress."""
        with self._lock:
            if session_type not in self._data:
                return {}
            
            session = self._data[session_type]
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
        with self._lock:
            if session_type in self._data:
                del self._data[session_type]
                self._pending_updates += 1
                self._maybe_save()
    
    def has_session(self, session_type: str) -> bool:
        """Check if a session exists."""
        with self._lock:
            return session_type in self._data
    
    def get_chunk_info(self, session_type: str, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get info for a specific chunk."""
        with self._lock:
            if session_type in self._data and chunk_id in self._data[session_type].get("chunks", {}):
                return self._data[session_type]["chunks"][chunk_id].copy()
            return None


# Backward compatibility
__all__ = ["ChunkStatus", "ProgressTracker"]
