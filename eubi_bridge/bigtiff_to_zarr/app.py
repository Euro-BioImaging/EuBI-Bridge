"""
Web interface for the High-Performance BigTIFF to OME-NGFF Converter
Provides real-time monitoring, configuration, and conversion management.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uuid

from core.converter import HighPerformanceConverter
from core.progress_monitor import ProgressMonitor
from utils.hardware_detection import HardwareDetector
from utils.config_profiles import ConfigProfileManager

def create_app():
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="High-Performance BigTIFF to OME-NGFF Converter",
        description="Web interface for optimized terabyte-scale data conversion",
        version="1.0.0"
    )
    
    # Mount static files
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    # Global state management
    active_conversions: Dict[str, Dict[str, Any]] = {}
    websocket_connections: Dict[str, WebSocket] = {}
    
    class ConversionRequest(BaseModel):
        input_path: str
        output_path: str
        profile: str = "auto"
        workers: Optional[int] = None
        chunk_memory_mb: Optional[int] = None
        compression: str = "blosc2-lz4"
        compression_level: int = 3
        use_numa: bool = False
        use_async_io: bool = True
        pyramid_levels: Optional[int] = None
        downsample_factor: int = 2
    
    class SystemInfoResponse(BaseModel):
        cpu_info: Dict[str, Any]
        memory_info: Dict[str, Any]
        storage_info: Dict[str, Any]
        numa_info: Dict[str, Any]
        recommended_profile: str
    
    @app.get("/", response_class=HTMLResponse)
    async def get_index():
        """Serve the main web interface."""
        return FileResponse("static/index.html")
    
    @app.get("/api/system-info", response_model=SystemInfoResponse)
    async def get_system_info():
        """Get system hardware information and recommendations."""
        detector = HardwareDetector()
        await detector.detect_hardware()
        
        return SystemInfoResponse(
            cpu_info=detector.get_cpu_info(),
            memory_info=detector.get_memory_info(),
            storage_info=detector.hardware_info.get("storage", {}),
            numa_info=detector.hardware_info.get("numa", {}),
            recommended_profile=detector.recommend_profile()
        )
    
    @app.get("/api/profiles")
    async def get_profiles():
        """Get available configuration profiles."""
        config_manager = ConfigProfileManager()
        return config_manager.get_all_profiles()
    
    @app.get("/api/profile/{profile_name}")
    async def get_profile_config(profile_name: str):
        """Get configuration for a specific profile."""
        config_manager = ConfigProfileManager()
        try:
            config = config_manager.get_profile_config(profile_name)
            return config
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    
    @app.post("/api/conversion/start")
    async def start_conversion(request: ConversionRequest):
        """Start a new conversion process."""
        conversion_id = str(uuid.uuid4())
        
        # Validate input file
        input_path = Path(request.input_path)
        if not input_path.exists():
            raise HTTPException(status_code=400, detail=f"Input file not found: {input_path}")
        
        # Load configuration
        config_manager = ConfigProfileManager()
        config = config_manager.get_profile_config(request.profile)
        
        # Override with request parameters
        if request.workers:
            config["workers"] = request.workers
        if request.chunk_memory_mb:
            config["chunk_memory_mb"] = request.chunk_memory_mb
        config["compression"] = request.compression
        config["compression_level"] = request.compression_level
        config["use_numa"] = request.use_numa
        config["use_async_io"] = request.use_async_io
        
        # Initialize converter and progress monitor
        converter = HighPerformanceConverter(config)
        progress_monitor = ProgressMonitor()
        
        # Store conversion info
        active_conversions[conversion_id] = {
            "request": request.dict(),
            "converter": converter,
            "progress_monitor": progress_monitor,
            "status": "starting",
            "start_time": time.time(),
            "error": None
        }
        
        # Start conversion in background
        asyncio.create_task(run_conversion_background(conversion_id, converter, progress_monitor, request))
        
        return {"conversion_id": conversion_id, "status": "started"}
    
    @app.get("/api/conversion/{conversion_id}/status")
    async def get_conversion_status(conversion_id: str):
        """Get status of a conversion process."""
        if conversion_id not in active_conversions:
            raise HTTPException(status_code=404, detail="Conversion not found")
        
        conversion = active_conversions[conversion_id]
        
        # Get progress from monitor
        progress_data = {}
        if conversion["progress_monitor"]:
            progress_data = await conversion["progress_monitor"].get_current_stats()
        
        return {
            "conversion_id": conversion_id,
            "status": conversion["status"],
            "start_time": conversion["start_time"],
            "error": conversion["error"],
            "progress": progress_data
        }
    
    @app.delete("/api/conversion/{conversion_id}")
    async def cancel_conversion(conversion_id: str):
        """Cancel a running conversion."""
        if conversion_id not in active_conversions:
            raise HTTPException(status_code=404, detail="Conversion not found")
        
        conversion = active_conversions[conversion_id]
        
        # Cancel the conversion
        if conversion["converter"]:
            await conversion["converter"].cancel()
        
        if conversion["progress_monitor"]:
            await conversion["progress_monitor"].stop()
        
        conversion["status"] = "cancelled"
        
        return {"status": "cancelled"}
    
    @app.get("/api/conversions")
    async def list_conversions():
        """List all active and recent conversions."""
        result = []
        for conversion_id, conversion in active_conversions.items():
            progress_data = {}
            if conversion["progress_monitor"]:
                progress_data = await conversion["progress_monitor"].get_current_stats()
            
            result.append({
                "conversion_id": conversion_id,
                "status": conversion["status"],
                "input_path": conversion["request"]["input_path"],
                "output_path": conversion["request"]["output_path"],
                "start_time": conversion["start_time"],
                "progress": progress_data
            })
        
        return result
    
    @app.websocket("/ws/{conversion_id}")
    async def websocket_endpoint(websocket: WebSocket, conversion_id: str):
        """WebSocket endpoint for real-time progress updates."""
        await websocket.accept()
        websocket_connections[conversion_id] = websocket
        
        try:
            while conversion_id in active_conversions:
                conversion = active_conversions[conversion_id]
                
                # Send progress update
                progress_data = {}
                if conversion["progress_monitor"]:
                    progress_data = await conversion["progress_monitor"].get_current_stats()
                
                await websocket.send_json({
                    "type": "progress_update",
                    "conversion_id": conversion_id,
                    "status": conversion["status"],
                    "progress": progress_data
                })
                
                # Check if conversion is complete
                if conversion["status"] in ["completed", "failed", "cancelled"]:
                    break
                
                await asyncio.sleep(1)  # Update every second
                
        except WebSocketDisconnect:
            pass
        finally:
            if conversion_id in websocket_connections:
                del websocket_connections[conversion_id]
    
    async def run_conversion_background(conversion_id: str, converter: HighPerformanceConverter, 
                                      progress_monitor: ProgressMonitor, request: ConversionRequest):
        """Run conversion in background and update status."""
        conversion = active_conversions[conversion_id]
        
        try:
            # Start progress monitoring
            await progress_monitor.start()
            conversion["status"] = "running"
            
            # Run conversion
            success = await converter.convert(
                input_path=request.input_path,
                output_path=request.output_path,
                progress_monitor=progress_monitor
            )
            
            if success:
                conversion["status"] = "completed"
            else:
                conversion["status"] = "failed"
                conversion["error"] = "Conversion failed"
                
        except Exception as e:
            conversion["status"] = "failed"
            conversion["error"] = str(e)
        finally:
            await progress_monitor.stop()
    
    return app

if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=5000)
