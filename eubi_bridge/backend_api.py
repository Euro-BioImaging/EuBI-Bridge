"""FastAPI backend server for the React GUI."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Dict, Any

# Import the real EuBI-Bridge library
from eubi_bridge.ebridge import EuBIBridge

app = FastAPI(title="EuBI-Bridge API", version="0.1.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage for saved configs
saved_configs: Dict[str, Dict[str, Any]] = {}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "EuBI-Bridge API is running"
    }


@app.get("/api/browse")
async def browse_path(path: str = "C:\\", page: int = 1, page_size: int = 50):
    """Browse file system."""
    try:
        # Normalize path
        if not path or path == "/":
            path = "C:\\"
        
        path = path.replace("/", "\\")
        
        if len(path) == 2 and path[1] == ":":
            path = path + "\\"
        
        browse_path_obj = Path(path)
        
        if not browse_path_obj.exists():
            return {
                "error": f"Path does not exist: {path}",
                "items": [],
                "path": path,
            }
        
        if not browse_path_obj.is_dir():
            return {
                "error": f"Path is not a directory: {path}",
                "items": [],
                "path": path,
            }
        
        items = []
        dirs = []
        files = []
        
        try:
            for item in browse_path_obj.iterdir():
                try:
                    item_info = {
                        "name": item.name,
                        "path": str(item),
                        "is_dir": item.is_dir(),
                        "size": item.stat().st_size if item.is_file() else 0,
                    }
                    if item.is_dir():
                        dirs.append(item_info)
                    else:
                        files.append(item_info)
                except (OSError, PermissionError):
                    pass
        except (OSError, PermissionError) as e:
            return {
                "error": f"Cannot read directory: {str(e)}",
                "items": [],
                "path": path,
            }
        
        dirs.sort(key=lambda x: x["name"].lower())
        files.sort(key=lambda x: x["name"].lower())
        items = dirs + files
        
        start = (page - 1) * page_size
        end = start + page_size
        paginated_items = items[start:end]
        
        return {
            "items": paginated_items,
            "total": len(items),
            "page": page,
            "page_size": page_size,
            "path": str(browse_path_obj.absolute()),
        }
    except Exception as e:
        return {
            "error": f"Error browsing path: {str(e)}",
            "items": [],
            "path": path,
        }


@app.post("/api/conversion/start")
async def start_conversion(data: dict):
    """Start a conversion job - calls EuBI-Bridge directly."""
    try:
        input_path = data.get("input_path", "").strip()
        output_path = data.get("output_path", "").strip()
        
        if not input_path or not output_path:
            return {"status": "error", "message": "Input and output paths required"}
        
        # Validate paths
        input_path_obj = Path(input_path)
        output_path_obj = Path(output_path)
        
        if not input_path_obj.exists():
            return {"status": "error", "message": f"Input path does not exist: {input_path}"}
        
        if not input_path_obj.is_dir():
            return {"status": "error", "message": f"Input path is not a directory: {input_path}"}
        
        # Create output directory
        try:
            output_path_obj.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return {"status": "error", "message": f"Cannot create output directory: {str(e)}"}
        
        # Call EuBI-Bridge directly - it handles everything
        try:
            bridge = EuBIBridge()
            
            kwargs = {
                "input_path": input_path,
                "output_path": output_path,
            }
            
            # Add optional parameters from config
            if data.get("max_workers"):
                kwargs["max_workers"] = data["max_workers"]
            
            if data.get("conversion_chunk_size"):
                kwargs["auto_chunk"] = False
                kwargs["y_chunk"] = data["conversion_chunk_size"]
                kwargs["x_chunk"] = data["conversion_chunk_size"]
            
            if data.get("metadata_scale"):
                kwargs["y_scale"] = data["metadata_scale"]
                kwargs["x_scale"] = data["metadata_scale"]
            
            if data.get("metadata_unit"):
                kwargs["y_unit"] = data["metadata_unit"]
                kwargs["x_unit"] = data["metadata_unit"]
            
            # Call the conversion - EuBI-Bridge handles all threading/multiprocessing
            print(f"Starting conversion with params: {kwargs}")
            result = bridge.to_zarr(**kwargs)
            
            return {
                "status": "completed",
                "message": "Conversion completed successfully",
                "output_path": str(output_path_obj.absolute()),
            }
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"Conversion failed: {str(e)}"
            }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error starting conversion: {str(e)}"
        }


@app.get("/api/config")
async def get_config():
    """Get default configuration."""
    return {
        "input_path": "",
        "output_path": "",
        "max_workers": 1,
        "conversion_chunk_size": 256,
        "metadata_scale": 1.0,
        "metadata_unit": "micrometer",
    }


@app.post("/api/config/save")
async def save_config(data: dict):
    """Save a configuration."""
    try:
        name = data.get("name", "default")
        saved_configs[name] = data
        return {"status": "ok", "message": f"Configuration '{name}' saved"}
    except Exception as e:
        return {"status": "error", "message": f"Error saving config: {str(e)}"}


@app.post("/api/config/load")
async def load_config(data: dict):
    """Load a saved configuration."""
    try:
        name = data.get("name", "default")
        if name not in saved_configs:
            return {"status": "error", "message": f"Configuration '{name}' not found"}
        return {"status": "ok", "config": saved_configs[name]}
    except Exception as e:
        return {"status": "error", "message": f"Error loading config: {str(e)}"}


@app.post("/api/config/reset")
async def reset_config():
    """Reset configuration to defaults."""
    return {
        "status": "ok",
        "config": {
            "input_path": "",
            "output_path": "",
            "max_workers": 1,
            "conversion_chunk_size": 256,
            "metadata_scale": 1.0,
            "metadata_unit": "micrometer",
        }
    }
