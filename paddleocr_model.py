"""
PaddleOCR Model for Table Recognition

This module provides a wrapper for PaddleOCR's table recognition
capabilities for extracting structured table data from images.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaddleOCRModel:
    """
    Wrapper for PaddleOCR table recognition.

    PaddleOCR provides excellent table structure recognition and OCR
    capabilities, making it ideal for extracting tables from images.
    """

    def __init__(
        self,
        use_gpu: bool = True,
        lang: str = "en"
    ):
        """
        Initialize the PaddleOCR model.

        Args:
            use_gpu: Whether to use GPU for inference
            lang: Language for OCR (en, ch, korean, japanese, etc.)
        """
        self.use_gpu = use_gpu
        self.lang = lang
        self.ocr = None
        self.table_engine = None

        logger.info(f"Initializing PaddleOCR model (GPU: {use_gpu}, Lang: {lang})")

    def load_model(self):
        """Load the PaddleOCR model and table engine."""
        if self.ocr is not None:
            logger.info("Model already loaded")
            return

        try:
            from paddleocr import PPStructure, save_structure_res

            logger.info("Loading PaddleOCR table recognition model...")

            # Initialize PPStructure for table recognition
            self.table_engine = PPStructure(
                table=True,
                ocr=True,
                show_log=False,
                use_gpu=self.use_gpu,
                lang=self.lang,
                layout=False,  # We only need table recognition
                recovery=False
            )

            logger.info("✓ PaddleOCR model loaded successfully!")

        except ImportError as e:
            logger.error(f"PaddleOCR not installed: {e}")
            logger.error("Install with: pip install paddleocr paddlepaddle")
            raise
        except Exception as e:
            logger.error(f"Error loading PaddleOCR model: {e}")
            raise

    def extract_table_from_image(
        self,
        image: Image.Image,
        max_new_tokens: int = 2048,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Extract table data from an image using PaddleOCR.

        Args:
            image: PIL Image object
            max_new_tokens: Not used for PaddleOCR (kept for interface compatibility)
            temperature: Not used for PaddleOCR (kept for interface compatibility)

        Returns:
            Dictionary containing extracted table data
        """
        if self.table_engine is None:
            self.load_model()

        try:
            # Convert PIL Image to numpy array
            img_array = np.array(image)

            logger.info("Processing image with PaddleOCR...")

            # Run table recognition
            result = self.table_engine(img_array)

            logger.info(f"PaddleOCR found {len(result)} regions")

            # Parse results and extract tables
            tables = self._parse_paddleocr_result(result)

            return {
                "tables": tables,
                "total_tables": len(tables)
            }

        except Exception as e:
            logger.error(f"Error during PaddleOCR inference: {e}")
            return {
                "tables": [],
                "total_tables": 0,
                "error": str(e),
                "message": f"Error during table extraction: {str(e)}"
            }

    def _parse_paddleocr_result(self, result: List[Dict]) -> List[Dict[str, Any]]:
        """
        Parse PaddleOCR result into our standard table format.

        Args:
            result: Raw result from PaddleOCR

        Returns:
            List of table dictionaries
        """
        tables = []
        table_num = 0

        for item in result:
            if item.get('type') == 'table':
                table_num += 1

                # Get table HTML result
                res_html = item.get('res', {}).get('html', '')

                # Parse table cells from result
                table_data = self._parse_table_html(res_html)

                if table_data:
                    tables.append({
                        "table_number": table_num,
                        "rows": table_data.get('rows', []),
                        "headers": table_data.get('headers', []),
                        "description": f"Table extracted by PaddleOCR"
                    })

        return tables

    def _parse_table_html(self, html: str) -> Optional[Dict[str, Any]]:
        """
        Parse HTML table into rows and columns.

        Args:
            html: HTML string of the table

        Returns:
            Dictionary with headers and rows
        """
        if not html:
            return None

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, 'html.parser')
            table = soup.find('table')

            if not table:
                return None

            headers = []
            rows = []

            # Extract headers
            thead = table.find('thead')
            if thead:
                header_row = thead.find('tr')
                if header_row:
                    headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]

            # Extract rows
            tbody = table.find('tbody') or table
            for tr in tbody.find_all('tr'):
                # Skip if this is the header row
                if thead and tr.parent == thead:
                    continue

                cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                if cells:  # Only add non-empty rows
                    rows.append(cells)

            # If no explicit headers, use first row as headers
            if not headers and rows:
                headers = rows[0]
                rows = rows[1:]

            return {
                "headers": headers,
                "rows": rows
            }

        except ImportError:
            logger.warning("BeautifulSoup not installed. Installing beautifulsoup4...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'beautifulsoup4'])
            # Retry after installation
            return self._parse_table_html(html)
        except Exception as e:
            logger.error(f"Error parsing table HTML: {e}")
            return None

    def unload_model(self):
        """Unload the model from memory."""
        if self.table_engine is not None:
            del self.table_engine
            self.table_engine = None
            logger.info("PaddleOCR model unloaded from memory")


# Singleton instance for model reuse
_model_instance: Optional[PaddleOCRModel] = None


def get_model_instance(
    use_gpu: bool = True,
    lang: str = "en",
    **kwargs  # Accept but ignore other parameters for interface compatibility
) -> PaddleOCRModel:
    """
    Get or create a singleton model instance.

    Args:
        use_gpu: Whether to use GPU
        lang: Language for OCR
        **kwargs: Other parameters (ignored, for compatibility)

    Returns:
        PaddleOCRModel instance
    """
    global _model_instance

    if _model_instance is None:
        _model_instance = PaddleOCRModel(
            use_gpu=use_gpu,
            lang=lang
        )
        _model_instance.load_model()

    return _model_instance
