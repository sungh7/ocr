// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const clearBtn = document.getElementById('clearBtn');
const extractBtn = document.getElementById('extractBtn');
const extractBtnText = document.getElementById('extractBtnText');
const spinner = document.getElementById('spinner');
const resultsSection = document.getElementById('resultsSection');
const resultsContainer = document.getElementById('resultsContainer');
const exportExcelBtn = document.getElementById('exportExcelBtn');

let selectedFile = null;
let extractedTableData = null;

// Event Listeners
uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
clearBtn.addEventListener('click', clearSelection);
extractBtn.addEventListener('click', extractTable);
exportExcelBtn.addEventListener('click', exportToExcel);

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

uploadArea.addEventListener('click', (e) => {
    if (e.target === uploadArea || e.target.closest('.upload-area')) {
        fileInput.click();
    }
});

// Functions
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('이미지 파일만 업로드할 수 있습니다.');
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        alert('파일 크기는 10MB를 초과할 수 없습니다.');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        previewSection.style.display = 'block';
        resultsSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function clearSelection() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    uploadArea.style.display = 'block';
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
}

async function extractTable() {
    if (!selectedFile) {
        alert('먼저 이미지를 선택해주세요.');
        return;
    }

    // Show loading state
    extractBtn.disabled = true;
    extractBtnText.textContent = '처리 중...';
    spinner.style.display = 'inline-block';

    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Call API
        const response = await fetch('/api/extract-table', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '서버 오류가 발생했습니다.');
        }

        const result = await response.json();

        // Display results
        displayResults(result);

    } catch (error) {
        console.error('Error:', error);
        displayError(error.message);
    } finally {
        // Reset button state
        extractBtn.disabled = false;
        extractBtnText.textContent = '표 추출하기';
        spinner.style.display = 'none';
    }
}

function displayResults(result) {
    resultsSection.style.display = 'block';
    resultsContainer.innerHTML = '';

    if (!result.success) {
        displayError('표 추출에 실패했습니다.');
        exportExcelBtn.style.display = 'none';
        return;
    }

    const data = result.data;

    // Store extracted data for Excel export
    extractedTableData = data;

    // Check if tables were found
    if (!data.tables || data.tables.length === 0) {
        resultsContainer.innerHTML = `
            <div class="no-tables">
                <p>${data.message || '이미지에서 표를 찾을 수 없습니다.'}</p>
            </div>
        `;

        // Hide Excel button if no tables found
        exportExcelBtn.style.display = 'none';

        // Show raw response if available
        if (data.raw_response) {
            resultsContainer.innerHTML += `
                <div class="table-result">
                    <h4>원본 응답</h4>
                    <div class="raw-response">${escapeHtml(data.raw_response)}</div>
                </div>
            `;
        }
        return;
    }

    // Display each table
    data.tables.forEach((table, index) => {
        const tableHtml = createTableHtml(table, index);
        resultsContainer.innerHTML += tableHtml;
    });

    // Show Excel export button
    exportExcelBtn.style.display = 'inline-flex';

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function createTableHtml(table, index) {
    let html = `
        <div class="table-result">
            <h4>표 ${table.table_number || (index + 1)}</h4>
    `;

    if (table.description) {
        html += `<p class="table-description">${escapeHtml(table.description)}</p>`;
    }

    if (table.rows && table.rows.length > 0) {
        html += '<table class="extracted-table"><thead>';

        // Add headers if available
        if (table.headers && table.headers.length > 0) {
            html += '<tr>';
            table.headers.forEach(header => {
                html += `<th>${escapeHtml(header)}</th>`;
            });
            html += '</tr>';
        } else {
            // Use first row as headers
            html += '<tr>';
            table.rows[0].forEach(cell => {
                html += `<th>${escapeHtml(cell)}</th>`;
            });
            html += '</tr>';
        }

        html += '</thead><tbody>';

        // Add data rows
        const startRow = (table.headers && table.headers.length > 0) ? 0 : 1;
        for (let i = startRow; i < table.rows.length; i++) {
            html += '<tr>';
            table.rows[i].forEach(cell => {
                html += `<td>${escapeHtml(cell)}</td>`;
            });
            html += '</tr>';
        }

        html += '</tbody></table>';
    }

    html += '</div>';
    return html;
}

function displayError(message) {
    resultsSection.style.display = 'block';
    resultsContainer.innerHTML = `
        <div class="error-message">
            <strong>오류:</strong> ${escapeHtml(message)}
        </div>
    `;
}

function escapeHtml(text) {
    if (typeof text !== 'string') {
        text = String(text);
    }
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function exportToExcel() {
    if (!extractedTableData) {
        alert('추출된 표 데이터가 없습니다.');
        return;
    }

    try {
        // Disable button during export
        exportExcelBtn.disabled = true;
        const originalText = exportExcelBtn.innerHTML;
        exportExcelBtn.innerHTML = '<span class="spinner"></span> 생성 중...';

        // Create form data
        const formData = new FormData();
        formData.append('table_data', JSON.stringify(extractedTableData));

        // Call export API
        const response = await fetch('/api/export-excel', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '엑셀 생성에 실패했습니다.');
        }

        // Download file
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'extracted_tables.xlsx';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        // Show success message
        console.log('엑셀 파일 다운로드 완료');

    } catch (error) {
        console.error('Error:', error);
        alert('엑셀 파일 생성 중 오류가 발생했습니다: ' + error.message);
    } finally {
        // Reset button
        exportExcelBtn.disabled = false;
        exportExcelBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="7 10 12 15 17 10"></polyline>
                <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
            엑셀로 다운로드
        `;
    }
}

// Check API health on page load
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const health = await response.json();

        if (!health.api_configured) {
            console.warn('DeepSeek API key not configured');
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

checkHealth();
