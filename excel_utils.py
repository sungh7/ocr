"""
Excel export utilities for table data.

This module provides functions to convert extracted table data to Excel files.
"""

from io import BytesIO
from typing import Dict, Any, List
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


def create_excel_from_tables(table_data: Dict[str, Any], filename: str = "extracted_tables.xlsx") -> BytesIO:
    """
    Create an Excel file from extracted table data.

    Args:
        table_data: Dictionary containing table extraction results
        filename: Name for the Excel file

    Returns:
        BytesIO object containing the Excel file
    """
    # Create workbook
    wb = Workbook()

    # Remove default sheet
    if "Sheet" in wb.sheetnames:
        wb.remove(wb["Sheet"])

    # Check if tables exist
    tables = table_data.get("tables", [])

    if not tables or len(tables) == 0:
        # Create a single sheet with "No tables found" message
        ws = wb.create_sheet("결과")
        ws["A1"] = "표를 찾을 수 없습니다"
        ws["A1"].font = Font(bold=True, size=12)
        message = table_data.get("message", "이미지에서 표를 찾을 수 없습니다.")
        ws["A2"] = message
    else:
        # Create a sheet for each table
        for idx, table in enumerate(tables, 1):
            table_number = table.get("table_number", idx)
            sheet_name = f"표{table_number}"

            # Ensure sheet name is valid (max 31 chars, no special chars)
            sheet_name = sheet_name[:31].replace("[", "(").replace("]", ")").replace("*", "").replace(":", "").replace("?", "").replace("/", "").replace("\\", "")

            ws = wb.create_sheet(sheet_name)

            # Add table description if available
            description = table.get("description", "")
            if description:
                ws["A1"] = description
                ws["A1"].font = Font(bold=True, italic=True, size=11)
                ws.merge_cells("A1:D1")
                current_row = 2
            else:
                current_row = 1

            # Get headers and rows
            headers = table.get("headers", [])
            rows = table.get("rows", [])

            # Style definitions
            header_font = Font(bold=True, color="FFFFFF", size=11)
            header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

            cell_alignment = Alignment(vertical="top", wrap_text=True)
            border = Border(
                left=Side(style="thin", color="000000"),
                right=Side(style="thin", color="000000"),
                top=Side(style="thin", color="000000"),
                bottom=Side(style="thin", color="000000")
            )

            # Write headers
            if headers and len(headers) > 0:
                for col_idx, header in enumerate(headers, 1):
                    cell = ws.cell(row=current_row, column=col_idx)
                    cell.value = str(header)
                    cell.font = header_font
                    cell.fill = header_fill
                    cell.alignment = header_alignment
                    cell.border = border
                current_row += 1
                start_row_for_data = current_row
            else:
                # Use first row as headers if no headers provided
                if rows and len(rows) > 0:
                    for col_idx, header in enumerate(rows[0], 1):
                        cell = ws.cell(row=current_row, column=col_idx)
                        cell.value = str(header)
                        cell.font = header_font
                        cell.fill = header_fill
                        cell.alignment = header_alignment
                        cell.border = border
                    current_row += 1
                    rows = rows[1:]  # Skip first row in data
                start_row_for_data = current_row

            # Write data rows
            for row_data in rows:
                for col_idx, cell_value in enumerate(row_data, 1):
                    cell = ws.cell(row=current_row, column=col_idx)
                    cell.value = str(cell_value) if cell_value is not None else ""
                    cell.alignment = cell_alignment
                    cell.border = border
                current_row += 1

            # Auto-adjust column widths
            for col_idx in range(1, len(headers or rows[0] if rows else []) + 1):
                column_letter = get_column_letter(col_idx)
                max_length = 0

                for row in ws[column_letter]:
                    try:
                        if row.value:
                            cell_length = len(str(row.value))
                            if cell_length > max_length:
                                max_length = cell_length
                    except:
                        pass

                # Set width (with limits)
                adjusted_width = min(max(max_length + 2, 10), 50)
                ws.column_dimensions[column_letter].width = adjusted_width

            # Set row heights for data rows
            for row_idx in range(start_row_for_data, current_row):
                ws.row_dimensions[row_idx].height = 20

    # Save to BytesIO
    excel_file = BytesIO()
    wb.save(excel_file)
    excel_file.seek(0)

    return excel_file


def validate_table_data(table_data: Dict[str, Any]) -> bool:
    """
    Validate that table data has the expected structure.

    Args:
        table_data: Dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(table_data, dict):
        return False

    if "tables" not in table_data:
        return False

    tables = table_data["tables"]
    if not isinstance(tables, list):
        return False

    # Validate each table structure
    for table in tables:
        if not isinstance(table, dict):
            return False

        # Must have either headers or rows
        if "headers" not in table and "rows" not in table:
            return False

    return True
