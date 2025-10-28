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

let selectedFile = null;

// Event Listeners
uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
clearBtn.addEventListener('click', clearSelection);
extractBtn.addEventListener('click', extractTable);

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
        return;
    }

    const data = result.data;

    // Check if tables were found
    if (!data.tables || data.tables.length === 0) {
        resultsContainer.innerHTML = `
            <div class="no-tables">
                <p>${data.message || '이미지에서 표를 찾을 수 없습니다.'}</p>
            </div>
        `;

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
