import os
import base64
import json
from io import BytesIO
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from openai import OpenAI
from pydantic_settings import BaseSettings
from pydantic import Field
import uvicorn


class Settings(BaseSettings):
    deepseek_api_key: str = Field(default="", env="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com", env="DEEPSEEK_BASE_URL"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

app = FastAPI(
    title="DeepSeek OCR Table Extraction",
    description="Web application for extracting tables from images using DeepSeek OCR",
    version="1.0.0"
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


def encode_image_to_base64(image_data: bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_data).decode('utf-8')


def extract_table_with_deepseek(image_data: bytes) -> dict:
    """
    Extract table from image using DeepSeek OCR API.

    Args:
        image_data: Image file as bytes

    Returns:
        Dictionary containing extracted table data
    """
    if not settings.deepseek_api_key:
        raise HTTPException(
            status_code=500,
            detail="DeepSeek API key not configured. Please set DEEPSEEK_API_KEY in .env file"
        )

    try:
        # Initialize OpenAI client with DeepSeek endpoint
        client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url
        )

        # Encode image to base64
        base64_image = encode_image_to_base64(image_data)

        # Create the prompt for table extraction
        prompt = """Analyze this image and extract any tables present.

For each table found:
1. Identify the table structure (rows and columns)
2. Extract all text content from each cell
3. Preserve the table layout and relationships

Return the result in JSON format with this structure:
{
    "tables": [
        {
            "table_number": 1,
            "rows": [
                ["cell1", "cell2", "cell3"],
                ["cell4", "cell5", "cell6"]
            ],
            "headers": ["Header1", "Header2", "Header3"],
            "description": "Brief description of what this table contains"
        }
    ],
    "total_tables": 1
}

If no tables are found, return:
{
    "tables": [],
    "total_tables": 0,
    "message": "No tables detected in the image"
}
"""

        # Call DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4096,
            temperature=0.1
        )

        # Extract response content
        content = response.choices[0].message.content

        # Try to parse JSON from the response
        # Sometimes the model wraps JSON in markdown code blocks
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # If JSON parsing fails, return raw content
            result = {
                "tables": [],
                "total_tables": 0,
                "raw_response": content,
                "message": "Unable to parse structured table data. Raw response included."
            }

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image with DeepSeek: {str(e)}"
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
    return {
        "status": "healthy",
        "api_configured": bool(settings.deepseek_api_key)
    }


@app.post("/api/extract-table")
async def extract_table(file: UploadFile = File(...)):
    """
    Extract table from uploaded image.

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

        # Validate image can be opened
        try:
            img = Image.open(BytesIO(image_data))
            img.verify()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid or corrupted image file: {str(e)}"
            )

        # Extract table using DeepSeek
        result = extract_table_with_deepseek(image_data)

        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "data": result
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


if __name__ == "__main__":
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)

    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
