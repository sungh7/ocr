import os
import logging
import json
from io import BytesIO
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic_settings import BaseSettings
from pydantic import Field
import uvicorn

# Import local DeepSeek VL model
from deepseek_vl import get_model_instance, DeepSeekVLModel

# Import Excel utilities
from excel_utils import create_excel_from_tables, validate_table_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_name: str = Field(
        default="deepseek-ai/deepseek-vl-1.3b-chat",
        env="MODEL_NAME",
        description="HuggingFace model name (1.3b or 7b)"
    )
    device: str = Field(
        default="auto",
        env="DEVICE",
        description="Device to run model on (auto, cuda, cpu)"
    )
    load_in_8bit: bool = Field(
        default=False,
        env="LOAD_IN_8BIT",
        description="Load model in 8-bit precision to save memory"
    )
    load_in_4bit: bool = Field(
        default=False,
        env="LOAD_IN_4BIT",
        description="Load model in 4-bit precision to save more memory"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

app = FastAPI(
    title="DeepSeek OCR Table Extraction (Local)",
    description="Web application for extracting tables from images using local DeepSeek VL model",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global model instance (loaded on startup)
model: Optional[DeepSeekVLModel] = None


@app.on_event("startup")
async def startup_event():
    """
    Load the model on application startup.
    This ensures the model is ready before handling requests.
    """
    global model
    logger.info("Loading DeepSeek VL model...")
    logger.info(f"Model: {settings.model_name}")
    logger.info(f"Device: {settings.device}")
    logger.info(f"8-bit: {settings.load_in_8bit}, 4-bit: {settings.load_in_4bit}")

    try:
        model = get_model_instance(
            model_name=settings.model_name,
            device=settings.device,
            load_in_8bit=settings.load_in_8bit,
            load_in_4bit=settings.load_in_4bit
        )
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Application will start but model inference will fail.")
        logger.error("Make sure you have downloaded the model using: python download_model.py")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global model
    if model is not None:
        logger.info("Unloading model...")
        model.unload_model()


def extract_table_with_local_model(image: Image.Image) -> dict:
    """
    Extract table from image using local DeepSeek VL model.

    Args:
        image: PIL Image object

    Returns:
        Dictionary containing extracted table data
    """
    global model

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please restart the application or check logs."
        )

    try:
        result = model.extract_table_from_image(
            image=image,
            max_new_tokens=2048,
            temperature=0.1
        )
        return result

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image with local model: {str(e)}"
        )


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: Frontend files not found</h1>",
            status_code=404
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    import torch

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": settings.model_name,
        "device": settings.device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }


@app.post("/api/extract-table")
async def extract_table(file: UploadFile = File(...)):
    """
    Extract table from uploaded image using local model.

    Args:
        file: Uploaded image file

    Returns:
        JSON response with extracted table data
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file."
        )

    try:
        # Read image data
        image_data = await file.read()

        # Open and validate image
        try:
            img = Image.open(BytesIO(image_data))
            img.load()  # Ensure image is fully loaded

            # Convert to RGB if necessary
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid or corrupted image file: {str(e)}"
            )

        # Extract table using local model
        logger.info(f"Processing image: {file.filename}")
        result = extract_table_with_local_model(img)

        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "data": result
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/api/export-excel")
async def export_to_excel(table_data: str = Form(...)):
    """
    Export extracted table data to Excel file.

    Args:
        table_data: JSON string containing extracted table data

    Returns:
        Excel file as a download
    """
    try:
        # Parse JSON data
        try:
            data = json.loads(table_data)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON data: {str(e)}"
            )

        # Validate data structure
        if not validate_table_data(data):
            raise HTTPException(
                status_code=400,
                detail="Invalid table data structure"
            )

        # Generate filename
        filename = "extracted_tables.xlsx"

        # Create Excel file
        logger.info("Creating Excel file from table data")
        excel_file = create_excel_from_tables(data, filename)

        # Return as download
        return StreamingResponse(
            excel_file,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating Excel file: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error creating Excel file: {str(e)}"
        )


if __name__ == "__main__":
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)

    # Run the application
    uvicorn.run(
        "app_local:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # Disable reload to keep model in memory
    )
