// Theme toggle functionality
const themeToggle = document.getElementById('themeToggle');
const themeIcon = themeToggle.querySelector('i');
const body = document.body;

// Check for saved theme preference or respect OS preference
const savedTheme = localStorage.getItem('theme') || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
if (savedTheme === 'dark') {
    body.classList.add('dark-theme');
    themeIcon.classList.remove('fa-moon');
    themeIcon.classList.add('fa-sun');
}

themeToggle.addEventListener('click', () => {
    body.classList.toggle('dark-theme');
    const isDark = body.classList.contains('dark-theme');
    
    if (isDark) {
        themeIcon.classList.remove('fa-moon');
        themeIcon.classList.add('fa-sun');
        localStorage.setItem('theme', 'dark');
    } else {
        themeIcon.classList.remove('fa-sun');
        themeIcon.classList.add('fa-moon');
        localStorage.setItem('theme', 'light');
    }
});

// File upload functionality
const modelDropZone = document.getElementById('modelDropZone');
const modelFileInput = document.getElementById('modelFileInput');
const modelSelectBtn = document.getElementById('modelSelectBtn');
const modelFileList = document.getElementById('modelFileList');

const archDropZone = document.getElementById('archDropZone');
const archFileInput = document.getElementById('archFileInput');
const archSelectBtn = document.getElementById('archSelectBtn');
const archFileList = document.getElementById('archFileList');

// Model file handling
modelSelectBtn.addEventListener('click', () => {
    modelFileInput.click();
});

modelFileInput.addEventListener('change', (e) => {
    handleFileUpload(e.target.files, 'model');
});

modelDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    modelDropZone.classList.add('dragover');
});

modelDropZone.addEventListener('dragleave', () => {
    modelDropZone.classList.remove('dragover');
});

modelDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    modelDropZone.classList.remove('dragover');
    handleFileUpload(e.dataTransfer.files, 'model');
});

// Architecture file handling
archSelectBtn.addEventListener('click', () => {
    archFileInput.click();
});

archFileInput.addEventListener('change', (e) => {
    handleFileUpload(e.target.files, 'arch');
});

archDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    archDropZone.classList.add('dragover');
});

archDropZone.addEventListener('dragleave', () => {
    archDropZone.classList.remove('dragover');
});

archDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    archDropZone.classList.remove('dragover');
    handleFileUpload(e.dataTransfer.files, 'arch');
});

async function handleFileUpload(files, type) {
    if (files.length > 0) {
        const file = files[0];
        const fileName = file.name;
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.error) {
                showNotification(`Upload error: ${result.error}`, 'error');
                return;
            }
            
            // Обновить список файлов
            if (type === 'model') {
                const modelFileItem = document.createElement('div');
                modelFileItem.className = 'file-item';
                modelFileItem.innerHTML = `
                    <div class="file-name">
                        <i class="fas fa-file-code"></i>
                        <span>${result.original_name}</span>
                    </div>
                    <div class="action-buttons">
                        <button class="action-btn" onclick="removeFile('${type}', this)">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                `;
                modelFileList.appendChild(modelFileItem);
                document.getElementById('selectedModel').innerHTML += `<option value="${result.filename}">${result.original_name}</option>`;
            } else {
                const archFileItem = document.createElement('div');
                archFileItem.className = 'file-item';
                archFileItem.innerHTML = `
                    <div class="file-name">
                        <i class="fas fa-file-code"></i>
                        <span>${result.original_name}</span>
                    </div>
                    <div class="action-buttons">
                        <button class="action-btn" onclick="removeFile('${type}', this)">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                `;
                archFileList.appendChild(archFileItem);
                document.getElementById('selectedArch').innerHTML += `<option value="${result.filename}">${result.original_name}</option>`;
            }
            
            showNotification(`Uploaded: ${fileName}`, 'success');
        } catch (error) {
            showNotification(`Upload error: ${error.message}`, 'error');
        }
    }
}

function removeFile(type, button) {
    const fileItem = button.parentElement.parentElement;
    const fileName = fileItem.querySelector('.file-name span').textContent;
    
    if (type === 'model') {
        modelFileList.removeChild(fileItem);
        const select = document.getElementById('selectedModel');
        const options = select.options;
        for (let i = 0; i < options.length; i++) {
            if (options[i].text === fileName) {
                select.remove(i);
                break;
            }
        }
    } else {
        archFileList.removeChild(fileItem);
        const select = document.getElementById('selectedArch');
        const options = select.options;
        for (let i = 0; i < options.length; i++) {
            if (options[i].text === fileName) {
                select.remove(i);
                break;
            }
        }
    }
    
    showNotification(`Removed: ${fileName}`, 'info');
}

// Model analysis
const analyzeBtn = document.getElementById('analyzeBtn');
analyzeBtn.addEventListener('click', analyzeModel);

async function analyzeModel() {
    const selectedModel = document.getElementById('selectedModel').value;
    const selectedArch = document.getElementById('selectedArch').value;
    
    if (!selectedModel) {
        showNotification('Please select a model to analyze', 'error');
        return;
    }
    
    if (!selectedArch) {
        showNotification('Please select an architecture', 'error');
        return;
    }
    
    showNotification('Analyzing model...', 'info');
    
    try {
        // Формируем путь к модели
        const modelPath = `results/models/${selectedModel}`;
        
        // Call API to analyze model
        const response = await fetch('/analyze_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_path: modelPath
            })
        });
        
        const result = await response.json();
        
        if (result.error) {
            showNotification(`Analysis error: ${result.error}`, 'error');
            return;
        }
        
        // Update stats
        document.getElementById('totalLayers').textContent = result.model_analysis.total_layers || 0;
        document.getElementById('supportedLayers').textContent = result.model_analysis.supported_layers || 0;
        document.getElementById('unsupportedLayers').textContent = result.model_analysis.unsupported_layers || 0;
        
        // Update graph visualization
        updateModelGraph(result.model_analysis);
        
        showNotification('Model analysis complete', 'success');
    } catch (error) {
        showNotification(`Analysis error: ${error.message}`, 'error');
    }
}

// Update model graph visualization
function updateModelGraph(analysis) {
    const graphContainer = document.getElementById('modelGraph');
    graphContainer.innerHTML = `
        <svg width="100%" height="100%" viewBox="0 0 800 400">
            <rect x="50" y="50" width="700" height="300" fill="#f0f0f0" stroke="#ccc" stroke-width="1"/>
            <text x="400" y="200" font-family="Arial" font-size="20" text-anchor="middle" fill="#333">Model Structure Visualization</text>
            <text x="400" y="230" font-family="Arial" font-size="14" text-anchor="middle" fill="#666">Model analysis complete</text>
        </svg>
    `;
}

// Export functionality
const exportBtn = document.getElementById('exportBtn');
exportBtn.addEventListener('click', exportModel);

async function exportModel() {
    const selectedModel = document.getElementById('selectedModel').value;
    const exportFormat = document.getElementById('exportFormat').value;
    const exportDir = document.getElementById('exportDir').value;
    const exportModelName = document.getElementById('exportModelName').value;
    
    if (!selectedModel) {
        showNotification('Please select a model to export', 'error');
        return;
    }
    
    showNotification(`Exporting ${selectedModel} to ${exportFormat} format...`, 'info');
    
    try {
        // Формируем путь к модели
        const modelPath = `results/models/${selectedModel}`;
        
        // Call API to export model
        const response = await fetch('/export', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_path: modelPath,
                format: exportFormat,
                export_dir: exportDir,
                model_name: exportModelName
            })
        });
        
        const result = await response.json();
        
        if (result.error) {
            showNotification(`Export error: ${result.error}`, 'error');
            return;
        }
        
        showNotification(`Export completed successfully!`, 'success');
        
        // Add to results
        addExportResult(exportModelName, exportFormat, exportDir);
    } catch (error) {
        showNotification(`Export error: ${error.message}`, 'error');
    }
}
function addExportResult(name, format, dir) {
    const resultsContainer = document.getElementById('exportResults');
    const resultItem = document.createElement('div');
    resultItem.className = 'result-item';
    resultItem.innerHTML = `
        <div>
            <div class="result-name">${name} (${format})</div>
            <div class="result-path">${dir}/model_export.${format}</div>
        </div>
        <div class="result-actions">
            <button><i class="fas fa-download"></i></button>
            <button><i class="fas fa-trash"></i></button>
        </div>
    `;
    resultsContainer.appendChild(resultItem);
}

// Split points functionality
const addSplitPointBtn = document.getElementById('addSplitPointBtn');
const splitPointInput = document.getElementById('splitPointInput');
const splitPointsList = document.getElementById('splitPointsList');

addSplitPointBtn.addEventListener('click', addSplitPoint);
splitPointInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') addSplitPoint();
});

function addSplitPoint() {
    const pointValue = splitPointInput.value.trim();
    if (!pointValue) return;
    
    const splitPoint = document.createElement('div');
    splitPoint.className = 'split-point';
    splitPoint.innerHTML = `
        <div class="split-point-info">
            <div class="split-point-name">${pointValue}</div>
            <div class="split-point-type">Layer Index</div>
        </div>
        <div class="split-point-actions">
            <button class="action-btn"><i class="fas fa-edit"></i></button>
            <button class="action-btn"><i class="fas fa-trash"></i></button>
        </div>
    `;
    
    splitPointsList.appendChild(splitPoint);
    splitPointInput.value = '';
    
    showNotification(`Added split point: ${pointValue}`, 'success');
}

// Export parts functionality
const exportPartsBtn = document.getElementById('exportPartsBtn');
exportPartsBtn.addEventListener('click', exportParts);

async function exportParts() {
    const selectedModel = document.getElementById('selectedModel').value;
    
    if (!selectedModel) {
        showNotification('Please select a model to export parts', 'error');
        return;
    }
    
    showNotification('Exporting model parts...', 'info');
    
    try {
        // Формируем путь к модели
        const modelPath = `results/models/${selectedModel}`;
        
        // Call API to export parts
        const response = await fetch('/export_parts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_path: modelPath,
                export_dir: 'results/exports'
            })
        });
        
        const result = await response.json();
        
        if (result.error) {
            showNotification(`Parts export error: ${result.error}`, 'error');
            return;
        }
        
        showNotification('Parts exported successfully!', 'success');
        addExportResult('model_parts', 'parts', 'results/exports');
    } catch (error) {
        showNotification(`Parts export error: ${error.message}`, 'error');
    }
}


// Tab switching
const tabs = document.querySelectorAll('.tab');
tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        // Remove active class from all tabs
        tabs.forEach(t => t.classList.remove('active'));
        
        // Add active class to clicked tab
        tab.classList.add('active');
        
        // Get tab content
        const tabName = tab.getAttribute('data-tab');
        const tabContents = document.querySelectorAll('.tab-content');
        
        // Hide all tab contents
        tabContents.forEach(content => {
            content.classList.remove('active');
        });
        
        // Show selected tab content
        document.getElementById(`${tabName}Content`).classList.add('active');
    });
});

// Notification system
const notification = document.getElementById('notification');
const notificationMessage = document.getElementById('notificationMessage');

function showNotification(message, type = 'info') {
    notificationMessage.textContent = message;
    notification.className = `notification ${type} show`;
    
    setTimeout(() => {
        notification.classList.remove('show');
    }, 3000);
}

// Load architectures on page load
async function loadArchitectures() {
    try {
        const response = await fetch('/architectures');
        const result = await response.json();
        
        if (result.architectures) {
            const select = document.getElementById('selectedArch');
            select.innerHTML = '<option value="">-- Select an architecture --</option>';
            result.architectures.forEach(arch => {
                select.innerHTML += `<option value="${arch.file}">${arch.name}</option>`;
            });
        }
    } catch (error) {
        console.error('Error loading architectures:', error);
    }
}
async function loadFiles() {
    try {
        const response = await fetch('/files');
        const result = await response.json();
        
        if (result.files) {
            // Очищаем текущие списки
            modelFileList.innerHTML = '';
            archFileList.innerHTML = '';
            
            // Заполняем список файлов
            result.files.forEach(file => {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                fileItem.innerHTML = `
                    <div class="file-name">
                        <i class="fas fa-file-code"></i>
                        <span>${file.name}</span>
                    </div>
                    <div class="action-buttons">
                        <button class="action-btn" onclick="removeFile('model', this)">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                `;
                modelFileList.appendChild(fileItem);
                document.getElementById('selectedModel').innerHTML += `<option value="${file.name}">${file.name}</option>`;
            });
        }
    } catch (error) {
        console.error('Error loading files:', error);
    }
}
// Initialize page
document.addEventListener('DOMContentLoaded', () => {
    loadArchitectures();
    loadFiles(); // Добавить загрузку файлов
});
