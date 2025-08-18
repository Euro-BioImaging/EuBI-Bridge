"""
Real-time progress monitoring with throughput metrics and ETA calculations.
Provides detailed statistics for large-scale conversion operations.
"""

import asyncio
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import deque
import psutil


@dataclass
class ConversionStats:
    """Statistics for conversion progress."""
    start_time: float = field(default_factory=time.time)
    current_time: float = field(default_factory=time.time)
    
    # Data progress
    total_bytes: int = 0
    processed_bytes: int = 0
    
    # Level progress
    total_levels: int = 0
    completed_levels: int = 0
    current_level: int = 0
    
    # Chunk progress
    total_chunks: int = 0
    completed_chunks: int = 0
    failed_chunks: int = 0
    
    # Performance metrics
    throughput_history: deque = field(default_factory=lambda: deque(maxlen=100))
    memory_usage_history: deque = field(default_factory=lambda: deque(maxlen=100))
    cpu_usage_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ProgressMonitor:
    """Real-time progress monitor with advanced analytics."""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.stats = ConversionStats()
        self.is_active = False
        self.callbacks: List[Callable] = []
        
        # Threading for background monitoring
        self.monitor_thread = None
        self.stop_event = threading.Event()
        
        # Performance tracking
        self.last_update_time = time.time()
        self.last_processed_bytes = 0
        
        # ETA calculation
        self.eta_samples = deque(maxlen=20)  # Last 20 samples for ETA
        self.smooth_throughput = 0.0
        self.alpha = 0.1  # Exponential smoothing factor
    
    async def start(self):
        """Start progress monitoring."""
        self.is_active = True
        self.stats.start_time = time.time()
        self.stats.current_time = time.time()
        self.stop_event.clear()
        
        # Start background monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    async def stop(self):
        """Stop progress monitoring."""
        self.is_active = False
        self.stop_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self.stop_event.wait(self.update_interval):
            try:
                self._update_system_metrics()
                self._calculate_throughput()
                self._notify_callbacks()
            except Exception as e:
                self.stats.errors.append(f"Monitor error: {e}")
    
    def _update_system_metrics(self):
        """Update system resource metrics."""
        current_time = time.time()
        
        # Update current time
        self.stats.current_time = current_time
        
        # Memory usage
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            self.stats.memory_usage_history.append({
                "timestamp": current_time,
                "memory_mb": memory_mb,
                "memory_percent": process.memory_percent()
            })
        except Exception:
            pass
        
        # CPU usage
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            self.stats.cpu_usage_history.append({
                "timestamp": current_time,
                "cpu_percent": cpu_percent
            })
        except Exception:
            pass
    
    def _calculate_throughput(self):
        """Calculate current throughput and update ETA."""
        current_time = time.time()
        time_delta = current_time - self.last_update_time
        
        if time_delta >= self.update_interval:
            bytes_delta = self.stats.processed_bytes - self.last_processed_bytes
            
            if time_delta > 0:
                current_throughput = bytes_delta / time_delta  # bytes/second
                
                # Exponential smoothing for stable throughput
                if self.smooth_throughput == 0:
                    self.smooth_throughput = current_throughput
                else:
                    self.smooth_throughput = (
                        self.alpha * current_throughput + 
                        (1 - self.alpha) * self.smooth_throughput
                    )
                
                # Add to history
                self.stats.throughput_history.append({
                    "timestamp": current_time,
                    "throughput_mb_s": current_throughput / (1024 * 1024),
                    "smooth_throughput_mb_s": self.smooth_throughput / (1024 * 1024)
                })
                
                # Update ETA samples
                if self.stats.total_bytes > 0:
                    remaining_bytes = self.stats.total_bytes - self.stats.processed_bytes
                    if self.smooth_throughput > 0:
                        eta_seconds = remaining_bytes / self.smooth_throughput
                        self.eta_samples.append(eta_seconds)
            
            self.last_update_time = current_time
            self.last_processed_bytes = self.stats.processed_bytes
    
    def _notify_callbacks(self):
        """Notify registered callbacks of progress updates."""
        for callback in self.callbacks:
            try:
                callback(self.get_current_stats())
            except Exception as e:
                self.stats.errors.append(f"Callback error: {e}")
    
    async def initialize_conversion(self, analysis: Dict[str, Any]):
        """Initialize conversion with analysis data."""
        self.stats.total_bytes = analysis.get("file_size", 0)
        self.stats.total_levels = len(analysis.get("pyramid_levels", []))
        
        # Calculate total chunks across all levels
        total_chunks = 0
        for level_info in analysis.get("pyramid_levels", []):
            level_shape = level_info["shape"]
            # Estimate chunks per level (simplified)
            chunks_per_level = 1
            for dim in level_shape:
                chunks_per_level *= max(1, dim // 256)  # Assume 256 chunk size
            total_chunks += chunks_per_level
        
        self.stats.total_chunks = total_chunks
    
    async def update_progress(self, level: int = None, chunks_completed: int = None, 
                            total_chunks: int = None, bytes_processed: int = None):
        """Update progress information."""
        if level is not None:
            self.stats.current_level = level
            if level > self.stats.completed_levels:
                self.stats.completed_levels = level
        
        if chunks_completed is not None and total_chunks is not None:
            level_progress = chunks_completed / max(total_chunks, 1)
            # Update total chunk progress (simplified)
            chunks_in_level = total_chunks
            self.stats.completed_chunks += chunks_completed
        
        if bytes_processed is not None:
            self.stats.processed_bytes = bytes_processed
    
    async def report_error(self, error_message: str):
        """Report an error during conversion."""
        self.stats.errors.append(f"{time.time()}: {error_message}")
    
    async def report_warning(self, warning_message: str):
        """Report a warning during conversion."""
        self.stats.warnings.append(f"{time.time()}: {warning_message}")
    
    def add_callback(self, callback: Callable):
        """Add progress update callback."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove progress update callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    async def get_current_stats(self) -> Dict[str, Any]:
        """Get comprehensive current statistics."""
        current_time = time.time()
        elapsed_time = current_time - self.stats.start_time
        
        # Calculate progress percentages
        byte_progress = 0.0
        if self.stats.total_bytes > 0:
            byte_progress = (self.stats.processed_bytes / self.stats.total_bytes) * 100
        
        level_progress = 0.0
        if self.stats.total_levels > 0:
            level_progress = (self.stats.completed_levels / self.stats.total_levels) * 100
        
        chunk_progress = 0.0
        if self.stats.total_chunks > 0:
            chunk_progress = (self.stats.completed_chunks / self.stats.total_chunks) * 100
        
        # Calculate ETA
        eta_seconds = self._calculate_eta()
        
        # Current throughput
        current_throughput = 0.0
        if self.stats.throughput_history:
            current_throughput = self.stats.throughput_history[-1]["smooth_throughput_mb_s"]
        
        # Average throughput
        avg_throughput = 0.0
        if elapsed_time > 0 and self.stats.processed_bytes > 0:
            avg_throughput = (self.stats.processed_bytes / (1024 * 1024)) / elapsed_time
        
        # System resource usage
        current_memory = 0.0
        current_cpu = 0.0
        
        if self.stats.memory_usage_history:
            current_memory = self.stats.memory_usage_history[-1]["memory_mb"]
        
        if self.stats.cpu_usage_history:
            current_cpu = self.stats.cpu_usage_history[-1]["cpu_percent"]
        
        return {
            "is_active": self.is_active,
            "elapsed_time": elapsed_time,
            "eta_seconds": eta_seconds,
            
            # Progress
            "byte_progress_percent": byte_progress,
            "level_progress_percent": level_progress,
            "chunk_progress_percent": chunk_progress,
            "overall_progress_percent": (byte_progress + level_progress + chunk_progress) / 3,
            
            # Data statistics
            "total_bytes": self.stats.total_bytes,
            "processed_bytes": self.stats.processed_bytes,
            "remaining_bytes": self.stats.total_bytes - self.stats.processed_bytes,
            
            # Level statistics
            "total_levels": self.stats.total_levels,
            "completed_levels": self.stats.completed_levels,
            "current_level": self.stats.current_level,
            
            # Chunk statistics
            "total_chunks": self.stats.total_chunks,
            "completed_chunks": self.stats.completed_chunks,
            "failed_chunks": self.stats.failed_chunks,
            
            # Performance
            "current_throughput_mb_s": current_throughput,
            "average_throughput_mb_s": avg_throughput,
            "current_memory_mb": current_memory,
            "current_cpu_percent": current_cpu,
            
            # Error tracking
            "error_count": len(self.stats.errors),
            "warning_count": len(self.stats.warnings),
            "recent_errors": self.stats.errors[-5:],  # Last 5 errors
            "recent_warnings": self.stats.warnings[-5:],  # Last 5 warnings
            
            # Time formatting
            "elapsed_time_formatted": self._format_duration(elapsed_time),
            "eta_formatted": self._format_duration(eta_seconds) if eta_seconds else "Unknown"
        }
    
    def _calculate_eta(self) -> Optional[float]:
        """Calculate estimated time of arrival."""
        if not self.eta_samples:
            return None
        
        # Use median of recent samples for stability
        sorted_samples = sorted(self.eta_samples)
        median_eta = sorted_samples[len(sorted_samples) // 2]
        
        return max(0, median_eta)
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 0:
            return "Unknown"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    async def get_performance_history(self, samples: int = 100) -> Dict[str, List]:
        """Get performance history for visualization."""
        return {
            "throughput": list(self.stats.throughput_history)[-samples:],
            "memory_usage": list(self.stats.memory_usage_history)[-samples:],
            "cpu_usage": list(self.stats.cpu_usage_history)[-samples:]
        }
    
    async def export_stats(self, file_path: str):
        """Export detailed statistics to file."""
        import json
        
        export_data = {
            "conversion_stats": await self.get_current_stats(),
            "performance_history": await self.get_performance_history(1000),
            "all_errors": self.stats.errors,
            "all_warnings": self.stats.warnings,
            "export_timestamp": time.time()
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def reset(self):
        """Reset all statistics for new conversion."""
        self.stats = ConversionStats()
        self.last_update_time = time.time()
        self.last_processed_bytes = 0
        self.eta_samples.clear()
        self.smooth_throughput = 0.0


class LiveProgressDisplay:
    """Live console progress display with rich formatting."""
    
    def __init__(self, progress_monitor: ProgressMonitor):
        self.progress_monitor = progress_monitor
        self.display_active = False
        
        try:
            from rich.console import Console
            from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
            from rich.live import Live
            from rich.table import Table
            
            self.console = Console()
            self.rich_available = True
        except ImportError:
            self.rich_available = False
    
    async def start_display(self):
        """Start live progress display."""
        if not self.rich_available:
            # Fallback to simple console output
            self.progress_monitor.add_callback(self._simple_progress_callback)
            return
        
        # Rich-based live display
        self.display_active = True
        
        with Live(self._create_progress_table(), refresh_per_second=2) as live:
            self.live_display = live
            
            while self.display_active:
                live.update(self._create_progress_table())
                await asyncio.sleep(0.5)
    
    def stop_display(self):
        """Stop live progress display."""
        self.display_active = False
    
    def _simple_progress_callback(self, stats: Dict[str, Any]):
        """Simple progress callback for systems without rich."""
        progress = stats.get("overall_progress_percent", 0)
        throughput = stats.get("current_throughput_mb_s", 0)
        eta = stats.get("eta_formatted", "Unknown")
        
        print(f"\rProgress: {progress:.1f}% | "
              f"Speed: {throughput:.1f} MB/s | "
              f"ETA: {eta}", end="", flush=True)
    
    def _create_progress_table(self):
        """Create rich progress table."""
        if not self.rich_available:
            return "Progress monitoring active..."
        
        from rich.table import Table
        from rich.progress import Progress
        
        # Get current stats
        stats = asyncio.run(self.progress_monitor.get_current_stats())
        
        table = Table(title="BigTIFF to OME-NGFF Conversion Progress")
        
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        table.add_column("Details", style="white")
        
        # Overall progress
        table.add_row(
            "Overall Progress",
            f"{stats['overall_progress_percent']:.1f}%",
            f"Elapsed: {stats['elapsed_time_formatted']}"
        )
        
        # Throughput
        table.add_row(
            "Throughput",
            f"{stats['current_throughput_mb_s']:.1f} MB/s",
            f"Average: {stats['average_throughput_mb_s']:.1f} MB/s"
        )
        
        # ETA
        table.add_row(
            "ETA",
            stats['eta_formatted'],
            f"Remaining: {stats['remaining_bytes'] / (1024**3):.2f} GB"
        )
        
        # System resources
        table.add_row(
            "Memory",
            f"{stats['current_memory_mb']:.1f} MB",
            f"CPU: {stats['current_cpu_percent']:.1f}%"
        )
        
        # Levels and chunks
        table.add_row(
            "Levels",
            f"{stats['completed_levels']}/{stats['total_levels']}",
            f"Current: {stats['current_level']}"
        )
        
        table.add_row(
            "Chunks",
            f"{stats['completed_chunks']}/{stats['total_chunks']}",
            f"Failed: {stats['failed_chunks']}"
        )
        
        # Errors and warnings
        if stats['error_count'] > 0 or stats['warning_count'] > 0:
            table.add_row(
                "Issues",
                f"Errors: {stats['error_count']}",
                f"Warnings: {stats['warning_count']}"
            )
        
        return table


# Utility function for CLI integration
async def monitor_conversion_progress(progress_monitor: ProgressMonitor, 
                                   use_rich_display: bool = True):
    """Monitor conversion progress with optional rich display."""
    if use_rich_display:
        display = LiveProgressDisplay(progress_monitor)
        await display.start_display()
    else:
        # Simple callback-based monitoring
        def simple_callback(stats):
            print(f"Progress: {stats['overall_progress_percent']:.1f}% | "
                  f"Speed: {stats['current_throughput_mb_s']:.1f} MB/s")
        
        progress_monitor.add_callback(simple_callback)
