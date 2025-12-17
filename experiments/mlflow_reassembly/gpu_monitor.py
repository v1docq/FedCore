"""
Real-time GPU monitoring for MLflow experiments.

Tracks:
- GPU utilization (%)
- GPU memory usage (MB)
- GPU temperature (°C)
- Power consumption (W)
"""

import time
import threading
from typing import List, Dict, Optional
from dataclasses import dataclass
import torch


@dataclass
class GPUSnapshot:
    """Snapshot of GPU state at a specific moment in time."""
    timestamp: float
    utilization: float  # %
    memory_used: float  # MB
    memory_total: float  # MB
    memory_percent: float  # %
    temperature: Optional[float] = None  # °C
    power_usage: Optional[float] = None  # W


class GPUMonitor:
    """
    GPU monitoring in a background thread.

    Collects GPU metrics every N seconds and stores them for subsequent
    logging in MLflow.

    Usage:
        monitor = GPUMonitor(device_id=0, interval=5.0)
        monitor.start()

        # ... your training code ...

        monitor.stop()
        snapshots = monitor.get_snapshots()
    """
    
    def __init__(self, device_id: int = 0, interval: float = 5.0):
        """
        Args:
            device_id: GPU ID to monitor
            interval: Metrics collection interval in seconds
        """
        self.device_id = device_id
        self.interval = interval
        self.snapshots: List[GPUSnapshot] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time = None
    
    def _collect_snapshot(self) -> Optional[GPUSnapshot]:
        """Collects a single GPU state snapshot."""
        if not torch.cuda.is_available():
            return None
        
        try:
            # Use pynvml for detailed metrics (if available)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
                
                # Get detailed metrics
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = None
                
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                except:
                    power_usage = None
                
                snapshot = GPUSnapshot(
                    timestamp=time.time() - self._start_time,
                    utilization=utilization.gpu,
                    memory_used=memory_info.used / (1024**2),  # bytes to MB
                    memory_total=memory_info.total / (1024**2),
                    memory_percent=(memory_info.used / memory_info.total) * 100,
                    temperature=temperature,
                    power_usage=power_usage
                )
                
            except ImportError:
                # Fallback: use only torch.cuda
                memory_allocated = torch.cuda.memory_allocated(self.device_id) / (1024**2)
                memory_reserved = torch.cuda.memory_reserved(self.device_id) / (1024**2)
                memory_total = torch.cuda.get_device_properties(self.device_id).total_memory / (1024**2)
                
                snapshot = GPUSnapshot(
                    timestamp=time.time() - self._start_time,
                    utilization=0.0,  # Not available without pynvml
                    memory_used=memory_allocated,
                    memory_total=memory_total,
                    memory_percent=(memory_allocated / memory_total) * 100,
                    temperature=None,
                    power_usage=None
                )
            
            return snapshot
            
        except Exception as e:
            print(f"Warning: Failed to collect GPU snapshot: {e}")
            return None
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            snapshot = self._collect_snapshot()
            if snapshot:
                self.snapshots.append(snapshot)
            time.sleep(self.interval)
    
    def start(self):
        """Starts monitoring in a background thread."""
        if self._running:
            return
        
        self._running = True
        self._start_time = time.time()
        self.snapshots = []
        
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        
        print(f"GPU monitor started (device {self.device_id}, interval {self.interval}s)")
    
    def stop(self):
        """Stops monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.interval + 1.0)
        
        print(f"GPU monitor stopped. Collected {len(self.snapshots)} snapshots")
    
    def get_snapshots(self) -> List[GPUSnapshot]:
        """Returns collected snapshots."""
        return self.snapshots
    
    def get_summary_stats(self) -> Dict[str, float]:
        """
        Computes summary statistics across all snapshots.

        Returns:
            Dict with avg/max/min metric values
        """
        if not self.snapshots:
            return {}
        
        utilizations = [s.utilization for s in self.snapshots]
        memory_percents = [s.memory_percent for s in self.snapshots]
        memory_used = [s.memory_used for s in self.snapshots]
        
        stats = {
            'gpu_util_avg': sum(utilizations) / len(utilizations),
            'gpu_util_max': max(utilizations),
            'gpu_util_min': min(utilizations),
            'gpu_memory_percent_avg': sum(memory_percents) / len(memory_percents),
            'gpu_memory_percent_max': max(memory_percents),
            'gpu_memory_mb_avg': sum(memory_used) / len(memory_used),
            'gpu_memory_mb_max': max(memory_used),
        }
        
        # Add temperature if available
        temperatures = [s.temperature for s in self.snapshots if s.temperature is not None]
        if temperatures:
            stats['gpu_temp_avg'] = sum(temperatures) / len(temperatures)
            stats['gpu_temp_max'] = max(temperatures)
        
        # Add power usage if available
        powers = [s.power_usage for s in self.snapshots if s.power_usage is not None]
        if powers:
            stats['gpu_power_avg'] = sum(powers) / len(powers)
            stats['gpu_power_max'] = max(powers)
        
        return stats


# pynvml setup for detailed monitoring (optional)
def install_pynvml_if_needed():
    """Checks and installs pynvml for detailed GPU monitoring."""
    try:
        import pynvml
        pynvml.nvmlInit()
        return True
    except ImportError:
        print("Note: pynvml not installed. GPU utilization monitoring will be limited.")
        print("Install with: pip install nvidia-ml-py3")
        return False
    except Exception as e:
        print(f"Note: pynvml available but failed to initialize: {e}")
        return False


if __name__ == "__main__":
    # Monitoring test
    print("Testing GPU Monitor...")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)
    
    # Check pynvml
    has_pynvml = install_pynvml_if_needed()
    print(f"Detailed monitoring: {'Available' if has_pynvml else 'Limited'}")
    print()
    
    # Monitoring test
    monitor = GPUMonitor(device_id=0, interval=2.0)
    monitor.start()
    
    print("Monitoring for 10 seconds...")
    time.sleep(10)
    
    monitor.stop()
    
    # Show results
    snapshots = monitor.get_snapshots()
    print(f"\nCollected {len(snapshots)} snapshots:")
    
    for i, snapshot in enumerate(snapshots[:3], 1):  # First 3
        print(f"\n  Snapshot {i} (t={snapshot.timestamp:.1f}s):")
        print(f"    Utilization: {snapshot.utilization:.1f}%")
        print(f"    Memory: {snapshot.memory_used:.0f}/{snapshot.memory_total:.0f} MB ({snapshot.memory_percent:.1f}%)")
        if snapshot.temperature:
            print(f"    Temperature: {snapshot.temperature}°C")
        if snapshot.power_usage:
            print(f"    Power: {snapshot.power_usage:.1f}W")
    
    # Summary statistics
    stats = monitor.get_summary_stats()
    print(f"\nSummary Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    print("\n GPU Monitor test complete!")


