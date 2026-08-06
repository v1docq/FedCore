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

const loaderDropZone = document.getElementById('loaderDropZone');
const loaderFileInput = document.getElementById('loaderFileInput');
const loaderSelectBtn = document.getElementById('loaderSelectBtn');
const loaderFileList = document.getElementById('loaderFileList');

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

if (loaderSelectBtn && loaderFileInput) {
    loaderSelectBtn.addEventListener('click', () => loaderFileInput.click());
    loaderFileInput.addEventListener('change', (e) => handleLoaderUpload(e.target.files));
    loaderDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        loaderDropZone.classList.add('dragover');
    });
    loaderDropZone.addEventListener('dragleave', () => loaderDropZone.classList.remove('dragover'));
    loaderDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        loaderDropZone.classList.remove('dragover');
        handleLoaderUpload(e.dataTransfer.files);
    });
}

const inspectLoaderBtn = document.getElementById('inspectLoaderBtn');
if (inspectLoaderBtn) {
    inspectLoaderBtn.addEventListener('click', inspectSelectedLoader);
}

async function handleLoaderUpload(files) {
    if (!files || !files.length) return;
    const file = files[0];
    try {
        const formData = new FormData();
        formData.append('file', file);
        const response = await fetch('/upload_loader', { method: 'POST', body: formData });
        const result = await response.json();
        if (result.error) {
            showNotification(`Loader upload error: ${result.error}`, 'error');
            return;
        }

        const item = document.createElement('div');
        item.className = 'file-item';
        item.innerHTML = `
            <div class="file-name">
                <i class="fas fa-database"></i>
                <span>${result.original_name}</span>
            </div>
            <div class="action-buttons">
                <button class="action-btn" onclick="removeFile('loader', this)">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `;
        loaderFileList.appendChild(item);
        document.getElementById('selectedLoader').innerHTML +=
            `<option value="${result.filename}">${result.original_name}</option>`;
        document.getElementById('selectedLoader').value = result.filename;
        renderLoaderInfo(result.meta);
        showNotification(`Loader uploaded: ${file.name}`, 'success');
    } catch (error) {
        showNotification(`Loader upload error: ${error.message}`, 'error');
    }
}

async function inspectSelectedLoader() {
    const filename = document.getElementById('selectedLoader').value;
    if (!filename) {
        showNotification('Select a loader first', 'error');
        return;
    }
    try {
        const response = await fetch('/inspect_loader', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename })
        });
        const result = await response.json();
        if (result.error) {
            showNotification(result.error, 'error');
            return;
        }
        renderLoaderInfo(result.meta);
        showNotification('Loader info loaded', 'success');
    } catch (error) {
        showNotification(`Inspect error: ${error.message}`, 'error');
    }
}

function renderLoaderInfo(meta) {
    const panel = document.getElementById('loaderInfoPanel');
    if (!panel || !meta) return;
    panel.innerHTML = `
        <div class="loader-meta-grid">
            <div><strong>Name</strong><span>${meta.name || meta.file_name || '—'}</span></div>
            <div><strong>Samples</strong><span>${meta.num_samples ?? '—'}</span></div>
            <div><strong>Batch</strong><span>${meta.batch_size ?? '—'}</span></div>
            <div><strong>Shape</strong><span>${(meta.sample_shape || []).join('×') || '—'}</span></div>
            <div><strong>Classes</strong><span>${meta.num_classes ?? '—'}</span></div>
            <div><strong>Dtype</strong><span>${meta.dtype || '—'}</span></div>
            <div><strong>File</strong><span>${meta.file_name || '—'}</span></div>
        </div>
    `;
}

function updateModelProfilePair() {
    const modelSelect = document.getElementById('selectedModel');
    const archSelect = document.getElementById('selectedArch');
    const modelLabel = document.getElementById('pairModelName');
    const jsonLabel = document.getElementById('pairJsonName');
    if (!modelLabel || !jsonLabel) return;

    const modelOpt = modelSelect?.selectedOptions?.[0];
    const archOpt = archSelect?.selectedOptions?.[0];
    const modelName = (modelOpt && modelOpt.value)
        ? (modelOpt.textContent || modelOpt.value).trim()
        : 'model.pt';
    let jsonName = 'device_profile.json';
    if (archOpt && archOpt.value) {
        jsonName = archOpt.dataset.filename
            || archOpt.value.split(/[\\/]/).pop()
            || 'device_profile.json';
    }
    modelLabel.textContent = modelName;
    jsonLabel.textContent = jsonName;
}

async function loadDeviceProfileOps() {
    const archSelect = document.getElementById('selectedArch');
    const list = document.getElementById('deviceOpsList');
    const countEl = document.getElementById('deviceOpsCount');
    if (!list || !countEl || !archSelect) return;

    const archFile = archSelect.value;
    if (!archFile) {
        countEl.textContent = '0';
        list.innerHTML = '<p class="layers-empty">Select a device profile</p>';
        return;
    }

    try {
        const response = await fetch('/device_profile_ops', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ architecture_file: archFile }),
        });
        const result = await response.json();
        if (result.error) {
            countEl.textContent = '0';
            list.innerHTML = `<p class="layers-empty">${result.error}</p>`;
            return;
        }

        const ops = result.supported_ops || [];
        countEl.textContent = String(ops.length);
        if (!ops.length) {
            list.innerHTML = '<p class="layers-empty">No supported_ops in this profile</p>';
            return;
        }
        list.innerHTML = ops
            .map((op) => `<span class="device-op-chip">${op}</span>`)
            .join('');
    } catch (error) {
        countEl.textContent = '0';
        list.innerHTML = `<p class="layers-empty">${error.message}</p>`;
    }
}

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
            
            const modelFileItem = document.createElement('div');
            modelFileItem.className = 'file-item';
            modelFileItem.innerHTML = `
                <div class="file-name">
                    <i class="fas fa-file-code"></i>
                    <span>${result.original_name}</span>
                </div>
                <div class="action-buttons">
                    <button class="action-btn" onclick="removeFile('model', this)">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            `;
            modelFileList.appendChild(modelFileItem);
            document.getElementById('selectedModel').innerHTML +=
                `<option value="${result.filename}">${result.original_name}</option>`;
            document.getElementById('selectedModel').value = result.filename;
            updateModelProfilePair();

            showNotification(`Uploaded: ${fileName}`, 'success');
        } catch (error) {
            showNotification(`Upload error: ${error.message}`, 'error');
        }
    }
}

function removeFile(type, button) {
    const fileItem = button.parentElement.parentElement;
    const fileName = fileItem.querySelector('.file-name span').textContent;

    let listEl;
    let selectId;
    if (type === 'loader') {
        listEl = loaderFileList;
        selectId = 'selectedLoader';
    } else {
        listEl = modelFileList;
        selectId = 'selectedModel';
    }

    listEl.removeChild(fileItem);
    const select = document.getElementById(selectId);
    for (let i = 0; i < select.options.length; i++) {
        if (optionsTextMatches(select.options[i].text, fileName)) {
            select.remove(i);
            break;
        }
    }

    showNotification(`Removed: ${fileName}`, 'info');
}

function optionsTextMatches(optionText, fileName) {
    return optionText === fileName || optionText.startsWith(fileName + ' ');
}

// Model analysis
const analyzeBtn = document.getElementById('analyzeBtn');
analyzeBtn.addEventListener('click', analyzeModel);

document.getElementById('selectedModel')?.addEventListener('change', updateModelProfilePair);
document.getElementById('selectedArch')?.addEventListener('change', () => {
    updateModelProfilePair();
    loadDeviceProfileOps();
});

async function analyzeModel() {
    const selectedModel = document.getElementById('selectedModel').value;
    const selectedArch = document.getElementById('selectedArch').value;
    
    if (!selectedModel) {
        showNotification('Please select a model to analyze', 'error');
        return;
    }
    
    if (!selectedArch) {
        showNotification('Please select a device profile (.json)', 'error');
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
                model_path: modelPath,
                architecture_file: selectedArch
            })
        });
        
        const result = await response.json();
        
        if (result.error) {
            showNotification(`Analysis error: ${result.error}`, 'error');
            return;
        }
        
        const analysis = result.model_analysis || {};
        const graph = result.graph || null;

        document.getElementById('totalLayers').textContent =
            result.total_layers ?? analysis.total_layers ?? (graph && graph.total_layers) ?? 0;
        document.getElementById('supportedLayers').textContent =
            result.supported_layers ?? analysis.supported_layers ?? 0;
        document.getElementById('unsupportedLayers').textContent =
            result.unsupported_layers ?? analysis.unsupported_layers ?? 0;

        updateModelGraph(graph, analysis);
        updateLayersList(graph, analysis);

        showNotification('Model analysis complete', 'success');
    } catch (error) {
        showNotification(`Analysis error: ${error.message}`, 'error');
    }
}

function _resolveModules(graph, analysis) {
    if (graph && Array.isArray(graph.modules) && graph.modules.length) {
        return graph.modules;
    }
    if (graph && Array.isArray(graph.layers) && graph.layers.length) {
        return graph.layers;
    }
    if (analysis && Array.isArray(analysis.model_layers)) {
        return analysis.model_layers.map((layer, index) => ({
            id: `n${index}`,
            index,
            name: layer.name || `module_${index}`,
            type: layer.type || 'Unknown',
            supported: layer.supported,
            param_count: layer.param_count,
            group: (layer.name || '').split('.')[0] || 'root',
        }));
    }
    return [];
}

// Compact hierarchical SVG: module cards in tight groups (not called "layers")
function updateModelGraph(graph, analysis) {
    const graphContainer = document.getElementById('modelGraph');
    const modules = _resolveModules(graph, analysis);
    const modelClass = (graph && graph.model_class) || 'Model';

    if (!modules.length) {
        graphContainer.innerHTML = `
            <svg width="100%" height="100%" viewBox="0 0 800 320">
                <rect x="40" y="40" width="720" height="240" fill="#f0f0f0" stroke="#ccc" stroke-width="1"/>
                <text x="400" y="160" font-family="Arial" font-size="18" text-anchor="middle" fill="#333">No modules found</text>
            </svg>
        `;
        return;
    }

    const groups = [];
    const groupIndex = new Map();
    modules.forEach((mod) => {
        const key = mod.group || (mod.name || '').split('.')[0] || 'root';
        if (!groupIndex.has(key)) {
            groupIndex.set(key, groups.length);
            groups.push({ title: key, modules: [] });
        }
        groups[groupIndex.get(key)].modules.push(mod);
    });

    const nodeW = 140;
    const nodeH = 44;
    const gapX = 16;
    const gapY = 12;
    const padX = 14;
    const padY = 26;
    const blockGap = 12;
    const cols = 5;

    let cursorY = 40;
    const positions = new Map();
    const blockRects = [];

    groups.forEach((group) => {
        const n = group.modules.length;
        const rows = Math.ceil(n / cols);
        const usedCols = Math.min(cols, n);
        const innerW = usedCols * nodeW + Math.max(0, usedCols - 1) * gapX;
        const blockW = Math.max(innerW + padX * 2, 200);
        const blockH = rows * nodeH + Math.max(0, rows - 1) * gapY + padY + 10;
        const blockX = 24;
        const blockY = cursorY;

        blockRects.push({ x: blockX, y: blockY, w: blockW, h: blockH, title: group.title });

        group.modules.forEach((mod, i) => {
            const col = i % cols;
            const row = Math.floor(i / cols);
            positions.set(mod.id || `n${mod.index}`, {
                x: blockX + padX + col * (nodeW + gapX),
                y: blockY + padY + row * (nodeH + gapY),
                mod,
            });
        });

        cursorY += blockH + blockGap;
    });

    const width = Math.max(760, 24 + cols * nodeW + (cols - 1) * gapX + padX * 2 + 40);
    const height = Math.max(320, cursorY + 12);

    let edgesSvg = '';
    groups.forEach((group) => {
        for (let i = 0; i < group.modules.length - 1; i++) {
            const a = positions.get(group.modules[i].id);
            const b = positions.get(group.modules[i + 1].id);
            if (!a || !b) continue;
            if (Math.abs(a.y - b.y) < 1) {
                edgesSvg += `<line x1="${a.x + nodeW}" y1="${a.y + nodeH / 2}" x2="${b.x}" y2="${b.y + nodeH / 2}" stroke="#94a3b8" stroke-width="1.5"/>`;
            } else {
                edgesSvg += `<path d="M ${a.x + nodeW / 2} ${a.y + nodeH} L ${b.x + nodeW / 2} ${b.y}" fill="none" stroke="#94a3b8" stroke-width="1.5"/>`;
            }
        }
    });

    const blocksSvg = blockRects.map((b) => `
        <rect x="${b.x}" y="${b.y}" width="${b.w}" height="${b.h}" rx="8"
              fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
        <text x="${b.x + 10}" y="${b.y + 16}" font-family="Arial" font-size="12" font-weight="600" fill="#334155">${b.title}</text>
    `).join('');

    const nodesSvg = modules.map((mod, i) => {
        const pos = positions.get(mod.id || `n${mod.index}`);
        if (!pos) return '';
        const { x, y } = pos;
        let fill = '#e2e8f0';
        let stroke = '#64748b';
        if (mod.supported === true) {
            fill = '#dcfce7';
            stroke = '#16a34a';
        } else if (mod.supported === false) {
            fill = '#fee2e2';
            stroke = '#dc2626';
        }
        const short = (mod.short_name || mod.name || `m${i}`).split('.').pop();
        const label = short.length > 16 ? `${short.slice(0, 14)}…` : short;
        return `
            <g>
                <rect x="${x}" y="${y}" width="${nodeW}" height="${nodeH}" rx="6" fill="${fill}" stroke="${stroke}" stroke-width="2"/>
                <text x="${x + nodeW / 2}" y="${y + 18}" font-family="Arial" font-size="11" text-anchor="middle" fill="#0f172a">${label}</text>
                <text x="${x + nodeW / 2}" y="${y + 34}" font-family="Arial" font-size="10" text-anchor="middle" fill="#475569">${mod.type || ''}</text>
            </g>
        `;
    }).join('');

    graphContainer.innerHTML = `
        <svg width="100%" height="100%" viewBox="0 0 ${width} ${height}">
            <text x="24" y="22" font-family="Arial" font-size="13" fill="#334155">${modelClass} · ${modules.length} modules · ${groups.length} groups</text>
            ${blocksSvg}
            ${edgesSvg}
            ${nodesSvg}
        </svg>
    `;
}

function updateLayersList(graph, analysis) {
    const list = document.getElementById('layersList');
    if (!list) return;

    const modules = _resolveModules(graph, analysis);
    if (!modules.length) {
        list.innerHTML = '<p class="layers-empty">No modules yet. Run Analyze Model.</p>';
        return;
    }

    list.innerHTML = modules.map((mod, i) => {
        const supported = mod.supported;
        let badge = 'unknown';
        let badgeText = 'n/a';
        if (supported === true) {
            badge = 'ok';
            badgeText = 'supported';
        } else if (supported === false) {
            badge = 'bad';
            badgeText = 'unsupported';
        }
        const params = mod.param_count != null ? ` · ${mod.param_count} params` : '';
        return `
            <div class="layer-row layer-${badge}">
                <span class="layer-idx">${mod.index != null ? mod.index : i}</span>
                <span class="layer-name">${mod.name || '—'}</span>
                <span class="layer-type">${mod.type || '—'}</span>
                <span class="layer-meta">${badgeText}${params}</span>
            </div>
        `;
    }).join('');
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
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        const tabName = tab.getAttribute('data-tab');
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
            content.style.display = 'none';
        });

        const panel = document.getElementById(`${tabName}Content`);
        if (panel) {
            panel.classList.add('active');
            panel.style.display = 'block';
        }
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
        const select = document.getElementById('selectedArch');
        if (!select || !result.architectures) return;

        select.innerHTML = '';
        result.architectures.forEach((arch) => {
            const label = arch.alias || arch.name;
            select.innerHTML += `<option value="${arch.file}" data-filename="${arch.filename || ''}">${label}</option>`;
        });

        // Prefer Rockchip RK3588S when present
        const rk = result.architectures.find((a) =>
            (a.file || '').includes('rk3588s') || (a.alias || '').includes('RK3588')
        );
        if (rk) select.value = rk.file;
        else if (result.architectures.length) select.value = result.architectures[0].file;

        updateModelProfilePair();
        await loadDeviceProfileOps();
    } catch (error) {
        console.error('Error loading architectures:', error);
    }
}
async function loadFiles() {
    try {
        const response = await fetch('/files');
        const result = await response.json();
        
        if (result.files && modelFileList) {
            modelFileList.innerHTML = '';
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

async function loadLoaders() {
    try {
        const response = await fetch('/loaders');
        const result = await response.json();
        if (!result.loaders || !loaderFileList) return;

        loaderFileList.innerHTML = '';
        const select = document.getElementById('selectedLoader');
        select.innerHTML = '<option value="">-- Select a loader --</option>';

        result.loaders.forEach((loader) => {
            const item = document.createElement('div');
            item.className = 'file-item';
            item.innerHTML = `
                <div class="file-name">
                    <i class="fas fa-database"></i>
                    <span>${loader.name}</span>
                </div>
            `;
            loaderFileList.appendChild(item);
            select.innerHTML += `<option value="${loader.name}">${loader.name}</option>`;
        });
    } catch (error) {
        console.error('Error loading loaders:', error);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    // Uploads are session-only (cleared on server restart); do not restore old files.
    loadArchitectures().then(updateModelProfilePair);
    updateModelProfilePair();
});
