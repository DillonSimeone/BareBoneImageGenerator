document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const modelSelect = document.getElementById('model-select');
    const modelDesc = document.getElementById('model-desc');
    const samplerSelect = document.getElementById('sampler-select');
    const promptInput = document.getElementById('prompt');
    const negPromptInput = document.getElementById('neg-prompt');
    const stepsInput = document.getElementById('steps');
    const stepsVal = document.getElementById('steps-val');
    const cfgInput = document.getElementById('cfg');
    const cfgVal = document.getElementById('cfg-val');
    const widthSelect = document.getElementById('width');
    const heightSelect = document.getElementById('height');
    const seedInput = document.getElementById('seed');
    
    // LoRA UI Elements
    const loraSelect = document.getElementById('lora-select');
    const scanLorasBtn = document.getElementById('scan-loras-btn');
    const customLoraInput = document.getElementById('custom-lora-input');
    const loraWeightInput = document.getElementById('lora-weight-input');
    const loraWeightVal = document.getElementById('lora-weight-val');
    const addLoraBtn = document.getElementById('add-lora-btn');
    const activeLorasList = document.getElementById('active-loras-list');
    const generateBtn = document.getElementById('generate-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const generatingBadge = document.getElementById('generating-badge');
    const badgeTitle = document.getElementById('badge-title');
    const badgeTimer = document.getElementById('badge-timer');

    const placeholderView = document.getElementById('placeholder-view');
    const imageStage = document.getElementById('image-stage');
    const resultImg = document.getElementById('result-img');
    const incomingImg = document.getElementById('incoming-img');
    const imageActions = document.getElementById('image-actions');
    const downloadLink = document.getElementById('download-link');
    const reuseSeedBtn = document.getElementById('reuse-seed-btn');
    const historyGallery = document.getElementById('history-gallery');
    
    // Img2Img UI Elements
    const img2imgDropzone = document.getElementById('img2img-dropzone');
    const img2imgInput = document.getElementById('img2img-input');
    const dropzoneEmpty = document.getElementById('dropzone-empty');
    const dropzonePreview = document.getElementById('dropzone-preview');
    const img2imgPreviewImg = document.getElementById('img2img-preview-img');
    const clearImg2imgBtn = document.getElementById('clear-img2img-btn');
    const img2imgControls = document.getElementById('img2img-controls');
    const img2imgBadge = document.getElementById('img2img-badge');
    const strengthInput = document.getElementById('strength');
    const strengthVal = document.getElementById('strength-val');
    const editModeSelect = document.getElementById('edit-mode');
    const qwenModeGroup = document.getElementById('qwen-mode-group');
    const sendToImg2imgBtn = document.getElementById('send-to-img2img-btn');

    // Fullscreen View Elements
    const zoomBackdrop = document.getElementById('zoom-backdrop');
    const zoomImg = document.getElementById('zoom-img');
    const zoomIncomingImg = document.getElementById('zoom-incoming-img');
    const zoomCloseBtn = document.getElementById('zoom-close-btn');
    const zoomUseImg2imgBtn = document.getElementById('zoom-use-img2img-btn');
    const zoomRegenBtn = document.getElementById('zoom-regen-btn');
    const zoomDownloadBtn = document.getElementById('zoom-download-btn');
    const zoomMetaPrompt = document.getElementById('zoom-meta-prompt');
    const zoomMetaDetails = document.getElementById('zoom-meta-details');
    const zoomSideCol = document.getElementById('zoom-side-col');
    const zoomSideToggle = document.getElementById('zoom-side-toggle');
    const zoomNegSection = document.getElementById('zoom-neg-section');
    const zoomMetaNeg = document.getElementById('zoom-meta-neg');

    let currentInitImage = null;
    let allHistoryItems = [];
    let currentHistoryIndex = -1;
    let generateStartTime = 0;
    let generateTimerInterval = null;

    // Performance stats elements
    const perfStats = document.getElementById('perf-stats');
    const statTime = document.getElementById('stat-time');
    const statSpeed = document.getElementById('stat-speed');

    // Performance Mode elements
    const performanceSelect = document.getElementById('performance-select');
    const compileCheckbox = document.getElementById('compile-checkbox');
    const turboCheckbox = document.getElementById('turbo-checkbox');
    const previewCheckbox = document.getElementById('preview-checkbox');
    const cfgCutoffInput = document.getElementById('cfg-cutoff');
    const cfgCutoffVal = document.getElementById('cfg-cutoff-val');
    const statBreakdown = document.getElementById('stat-breakdown');
    const cancelBtn = document.getElementById('cancel-btn');
 
    let configLoaded = false;
    let progressInterval = null;
    let lastPreviewId = -1;
    let previewShown = false;
    let modelsData = {};
    let currentImageSeed = -1;
    let activeLoras = [];

    // Load active LoRAs from localStorage
    try {
        const stored = localStorage.getItem('active_loras');
        if (stored) activeLoras = JSON.parse(stored);
    } catch (e) {
        console.error('Error parsing active LoRAs:', e);
    }

    // Listeners for slider updates
    stepsInput.addEventListener('input', () => {
        stepsVal.textContent = stepsInput.value;
    });

    if (strengthInput && strengthVal) {
        strengthInput.addEventListener('input', () => {
            strengthVal.textContent = parseFloat(strengthInput.value).toFixed(2);
        });
    }

    // Img2Img helpers and listeners
    function setInitImage(sourceUrlOrData) {
        if (!sourceUrlOrData) return;
        currentInitImage = sourceUrlOrData;
        img2imgPreviewImg.src = sourceUrlOrData;
        dropzoneEmpty.style.display = 'none';
        dropzonePreview.style.display = 'flex';
        clearImg2imgBtn.style.display = 'inline-block';
        img2imgControls.style.display = 'block';
        img2imgBadge.textContent = 'ACTIVE';
        img2imgBadge.classList.add('active');
        checkQwenModeVisibility();
    }

    function clearInitImage() {
        currentInitImage = null;
        img2imgPreviewImg.src = '';
        if (img2imgInput) img2imgInput.value = '';
        dropzonePreview.style.display = 'none';
        dropzoneEmpty.style.display = 'flex';
        clearImg2imgBtn.style.display = 'none';
        img2imgControls.style.display = 'none';
        img2imgBadge.textContent = 'OPTIONAL';
        img2imgBadge.classList.remove('active');
    }

    function checkQwenModeVisibility() {
        if (!qwenModeGroup) return;
        const isQwen = modelSelect && modelSelect.value && modelSelect.value.toLowerCase().includes('qwen');
        qwenModeGroup.style.display = isQwen ? 'block' : 'none';
    }

    if (clearImg2imgBtn) {
        clearImg2imgBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            clearInitImage();
        });
    }

    if (img2imgDropzone && img2imgInput) {
        img2imgDropzone.addEventListener('click', () => {
            img2imgInput.click();
        });

        img2imgInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file && file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (evt) => setInitImage(evt.target.result);
                reader.readAsDataURL(file);
            }
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            img2imgDropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                img2imgDropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            img2imgDropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                img2imgDropzone.classList.remove('dragover');
            });
        });

        img2imgDropzone.addEventListener('drop', (e) => {
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (evt) => setInitImage(evt.target.result);
                reader.readAsDataURL(file);
            }
        });
    }

    // Paste from clipboard support
    window.addEventListener('paste', (e) => {
        const items = (e.clipboardData || (e.originalEvent && e.originalEvent.clipboardData))?.items;
        if (!items) return;
        for (const item of items) {
            if (item.kind === 'file' && item.type.startsWith('image/')) {
                const file = item.getAsFile();
                const reader = new FileReader();
                reader.onload = (evt) => setInitImage(evt.target.result);
                reader.readAsDataURL(file);
                break;
            }
        }
    });

    if (sendToImg2imgBtn) {
        sendToImg2imgBtn.addEventListener('click', () => {
            if (resultImg.src && !resultImg.classList.contains('hidden')) {
                setInitImage(resultImg.src);
            }
        });
    }

    if (zoomUseImg2imgBtn) {
        zoomUseImg2imgBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (zoomImg.src) {
                setInitImage(zoomImg.src);
                zoomBackdrop.classList.add('hidden');
            }
        });
    }

    if (zoomCloseBtn) {
        zoomCloseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            zoomBackdrop.classList.add('hidden');
        });
    }

    cfgInput.addEventListener('input', () => {
        cfgVal.textContent = parseFloat(cfgInput.value).toFixed(1);
    });

    loraWeightInput.addEventListener('input', () => {
        loraWeightVal.textContent = parseFloat(loraWeightInput.value).toFixed(1);
    });

    // Fetch models and set defaults
    async function loadModels() {
        try {
            const response = await fetch('/api/models');
            modelsData = await response.json();
            
            modelSelect.innerHTML = '';
            for (const [key, model] of Object.entries(modelsData)) {
                const option = document.createElement('option');
                option.value = key;
                option.textContent = model.name;
                modelSelect.appendChild(option);
            }
            
            // Fetch configuration from backend
            try {
                const configResponse = await fetch('/api/config');
                const configData = await configResponse.json();
                if (configData.preload_model && modelsData[configData.preload_model]) {
                    modelSelect.value = configData.preload_model;
                }
                if (configData.performance_mode && performanceSelect) {
                    performanceSelect.value = configData.performance_mode;
                }
                if (configData.torch_compile !== undefined && compileCheckbox) {
                    compileCheckbox.checked = configData.torch_compile;
                }
                if (configData.turbo_mode !== undefined && turboCheckbox) {
                    turboCheckbox.checked = configData.turbo_mode;
                }
            } catch (cfgError) {
                console.error('Error fetching config:', cfgError);
            }
            configLoaded = true;
            
            // Set initial defaults
            updateModelDefaults();
        } catch (error) {
            console.error('Error fetching models:', error);
        }
    }

    async function saveConfig() {
        // Avoid overwriting config.json with unchecked defaults before the saved config has been applied
        if (!configLoaded) return;
        const payload = {
            preload_model: modelSelect.value,
            performance_mode: performanceSelect ? performanceSelect.value : "max_speed",
            torch_compile: compileCheckbox ? compileCheckbox.checked : false,
            turbo_mode: turboCheckbox ? turboCheckbox.checked : false
        };
        try {
            await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
        } catch (e) {
            console.error('Error saving config:', e);
        }
    }

    function updateModelDefaults() {
        const selectedKey = modelSelect.value;
        const model = modelsData[selectedKey];
        if (model) {
            modelDesc.textContent = model.description;
            stepsInput.value = model.default_steps;
            stepsVal.textContent = model.default_steps;
            cfgInput.value = model.default_cfg;
            cfgVal.textContent = parseFloat(model.default_cfg).toFixed(1);
        }
        applyTurboDefaults();
        checkQwenModeVisibility();
    }

    function isTurboApplicable() {
        if (!turboCheckbox || !turboCheckbox.checked) return false;
        const key = (modelSelect.value || '').toLowerCase();
        return !['flux', 'z_image', 'qwen'].some(k => key.includes(k));
    }

    function applyTurboDefaults() {
        if (!isTurboApplicable()) return;
        stepsInput.value = 4;
        stepsVal.textContent = '4';
        cfgInput.value = 1.0;
        cfgVal.textContent = '1.0';
        if (samplerSelect && samplerSelect.querySelector('option[value="lcm"]')) {
            samplerSelect.value = 'lcm';
        }
    }

    if (turboCheckbox) {
        turboCheckbox.addEventListener('change', () => {
            saveConfig();
            updateModelDefaults();
            alert(turboCheckbox.checked
                ? 'Turbo LoRA enabled. The model reloads on your next generation (the DMD2 LoRA downloads once). Use 4-8 steps, CFG 1.0, LCM sampler.'
                : 'Turbo LoRA disabled. The model reloads on your next generation.');
        });
    }

    if (cfgCutoffInput && cfgCutoffVal) {
        const renderCutoff = () => {
            const v = parseFloat(cfgCutoffInput.value);
            cfgCutoffVal.textContent = v >= 1.0 ? 'OFF' : `${Math.round(v * 100)}%`;
            localStorage.setItem('cfg_cutoff', cfgCutoffInput.value);
        };
        const storedCutoff = localStorage.getItem('cfg_cutoff');
        if (storedCutoff) cfgCutoffInput.value = storedCutoff;
        cfgCutoffInput.addEventListener('input', renderCutoff);
        renderCutoff();
    }

    if (previewCheckbox) {
        const storedPreview = localStorage.getItem('live_preview');
        if (storedPreview !== null) previewCheckbox.checked = storedPreview === 'true';
        previewCheckbox.addEventListener('change', () => {
            localStorage.setItem('live_preview', String(previewCheckbox.checked));
        });
    }

    // Live progress polling (step counter + TAESD preview)
    function startProgressPolling() {
        stopProgressPolling();
        lastPreviewId = -1;
        previewShown = false;
        progressInterval = setInterval(async () => {
            try {
                const res = await fetch('/api/progress');
                const p = await res.json();
                if (!p.active) return;
                if (badgeTitle && p.total) badgeTitle.textContent = `STEP ${p.step}/${p.total}`;
                if (p.preview && p.preview_id !== lastPreviewId) {
                    lastPreviewId = p.preview_id;
                    placeholderView.classList.add('hidden');
                    if (imageStage) imageStage.classList.remove('hidden');
                    resultImg.classList.remove('hidden');
                    resultImg.src = p.preview;
                    previewShown = true;
                }
            } catch (e) {
                // Ignore transient polling errors while the server is busy
            }
        }, 300);
    }

    function stopProgressPolling() {
        if (progressInterval) {
            clearInterval(progressInterval);
            progressInterval = null;
        }
    }

    if (cancelBtn) {
        cancelBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            try {
                await fetch('/api/cancel', { method: 'POST' });
                if (badgeTitle) badgeTitle.textContent = 'CANCELLING...';
            } catch (err) {
                console.error('Cancel failed:', err);
            }
        });
    }

    modelSelect.addEventListener('change', () => {
        updateModelDefaults();
        saveConfig();
    });

    if (performanceSelect) {
        performanceSelect.addEventListener('change', () => {
            saveConfig();
            alert('Performance Mode updated. Settings will be applied and the model will reload on your next generation.');
        });
    }

    if (compileCheckbox) {
        compileCheckbox.addEventListener('change', () => {
            saveConfig();
            if (compileCheckbox.checked) {
                alert('Model compilation enabled. Warmup of ~60s is required on the very first generation (subsequent generations will be extremely fast).');
            } else {
                alert('Model compilation disabled.');
            }
        });
    }

    // Non-blocking Generating Badge Helpers
    function showGeneratingBadge(title = 'GENERATING...') {
        if (!generatingBadge) return;
        generatingBadge.classList.remove('hidden', 'completed');
        if (badgeTitle) badgeTitle.textContent = title;
        if (badgeTimer) badgeTimer.textContent = '0.0s';
        generateStartTime = performance.now();

        if (generateTimerInterval) clearInterval(generateTimerInterval);
        generateTimerInterval = setInterval(() => {
            const elapsed = (performance.now() - generateStartTime) / 1000;
            if (badgeTimer) badgeTimer.textContent = `${elapsed.toFixed(1)}s`;
        }, 100);
    }

    function hideGeneratingBadge(stats = null) {
        if (!generatingBadge) return;
        if (generateTimerInterval) {
            clearInterval(generateTimerInterval);
            generateTimerInterval = null;
        }

        if (stats && (stats.time_taken || stats.speed)) {
            generatingBadge.classList.add('completed');
            if (badgeTitle) badgeTitle.textContent = 'SYNTHESIS COMPLETE';
            const info = [stats.time_taken, stats.speed].filter(Boolean).join(' • ');
            if (badgeTimer) badgeTimer.textContent = info || 'Done';
            setTimeout(() => {
                generatingBadge.classList.add('hidden');
                setTimeout(() => {
                    generatingBadge.classList.remove('completed');
                }, 300);
            }, 2400);
        } else {
            generatingBadge.classList.add('hidden');
        }
    }

    // Dual-Buffer Smooth Slide Image Transition
    function transitionImage(currentEl, incomingEl, newUrl, direction = 'left') {
        if (!currentEl) return;
        if (!currentEl.src || currentEl.classList.contains('hidden') || currentEl.src.includes('data:image/gif')) {
            currentEl.src = newUrl;
            currentEl.classList.remove('hidden');
            if (incomingEl) incomingEl.classList.add('hidden');
            return;
        }
        if (currentEl.src === newUrl) return;

        if (!incomingEl) {
            currentEl.src = newUrl;
            return;
        }

        const outClass = direction === 'left' ? 'slide-out-left' : 'slide-out-right';
        const inClass = direction === 'left' ? 'slide-in-right' : 'slide-in-left';

        incomingEl.src = newUrl;
        incomingEl.classList.remove('hidden', 'slide-in-right', 'slide-in-left', 'slide-out-left', 'slide-out-right');
        currentEl.classList.remove('slide-in-right', 'slide-in-left', 'slide-out-left', 'slide-out-right');

        incomingEl.classList.add(inClass);
        currentEl.classList.add(outClass);

        const onAnimEnd = () => {
            currentEl.removeEventListener('animationend', onAnimEnd);
            currentEl.src = newUrl;
            currentEl.classList.remove(outClass);
            incomingEl.classList.remove(inClass);
            incomingEl.classList.add('hidden');
            incomingEl.src = '';
        };
        currentEl.addEventListener('animationend', onAnimEnd, { once: true });
    }

    // Update Fullscreen Lightbox HUD (Thin Left Column)
    function updateZoomHud(url, meta) {
        if (zoomDownloadBtn) zoomDownloadBtn.href = url;
        if (meta) {
            if (zoomMetaPrompt) zoomMetaPrompt.textContent = meta.prompt || 'No prompt directives';
            if (zoomNegSection && zoomMetaNeg) {
                if (meta.negative_prompt && meta.negative_prompt.trim()) {
                    zoomMetaNeg.textContent = meta.negative_prompt;
                    zoomNegSection.classList.remove('hidden');
                } else {
                    zoomNegSection.classList.add('hidden');
                }
            }
            if (zoomMetaDetails) {
                zoomMetaDetails.innerHTML = '';
                const chips = [];
                if (meta.steps) chips.push({ lbl: 'STEPS', val: meta.steps });
                if (meta.cfg_scale) chips.push({ lbl: 'CFG', val: parseFloat(meta.cfg_scale).toFixed(1) });
                if (meta.seed !== undefined && meta.seed !== -1) chips.push({ lbl: 'SEED', val: meta.seed });
                if (meta.sampler_key) chips.push({ lbl: 'SMP', val: meta.sampler_key });
                if (meta.time_taken) chips.push({ lbl: 'TIME', val: meta.time_taken, highlight: true });
                if (meta.speed) chips.push({ lbl: 'SPD', val: meta.speed, highlight: true });

                chips.forEach(c => {
                    const chip = document.createElement('div');
                    chip.className = 'zoom-spec-chip' + (c.highlight ? ' highlight' : '');
                    chip.innerHTML = `<span class="chip-lbl">${c.lbl}:</span> ${c.val}`;
                    zoomMetaDetails.appendChild(chip);
                });
            }
        }
    }

    // Fetch generation history
    async function loadHistory() {
        try {
            const response = await fetch('/api/history');
            const history = await response.json();
            allHistoryItems = history;
            
            historyGallery.innerHTML = '';
            if (history.length === 0) {
                historyGallery.innerHTML = '<div class="no-history">No generations yet.</div>';
                return;
            }

            history.forEach((item, idx) => {
                const div = document.createElement('div');
                div.className = 'history-item';
                
                const img = document.createElement('img');
                img.src = item.image_url;
                img.alt = item.meta.prompt || 'Generated Image';
                
                div.appendChild(img);
                
                div.addEventListener('click', () => {
                    const direction = idx > currentHistoryIndex ? 'right' : 'left';
                    currentHistoryIndex = idx;
                    displayImage(item.image_url, item.meta, direction);
                });
                
                historyGallery.appendChild(div);
            });
        } catch (error) {
            console.error('Error loading history:', error);
        }
    }

    function displayImage(url, meta, direction = 'left') {
        placeholderView.classList.add('hidden');
        if (imageStage) imageStage.classList.remove('hidden');
        resultImg.classList.remove('hidden');
        imageActions.classList.remove('hidden');
        downloadLink.href = url;
        
        transitionImage(resultImg, incomingImg, url, direction);

        // If fullscreen viewer is open, also slide the new image into view right in fullscreen!
        if (zoomBackdrop && !zoomBackdrop.classList.contains('hidden')) {
            transitionImage(zoomImg, zoomIncomingImg, url, direction);
            updateZoomHud(url, meta);
        }

        if (meta) {
            currentImageSeed = meta.seed || -1;
            // Only populate prompt inputs if user is not actively typing in them
            const activeEl = document.activeElement;
            if (meta.prompt && activeEl !== promptInput) promptInput.value = meta.prompt;
            if (meta.negative_prompt && activeEl !== negPromptInput) negPromptInput.value = meta.negative_prompt;
            if (meta.steps && activeEl !== stepsInput) {
                stepsInput.value = meta.steps;
                stepsVal.textContent = meta.steps;
            }
            if (meta.cfg_scale && activeEl !== cfgInput) {
                cfgInput.value = meta.cfg_scale;
                cfgVal.textContent = parseFloat(meta.cfg_scale).toFixed(1);
            }
            if (meta.sampler_key && samplerSelect && activeEl !== samplerSelect) {
                samplerSelect.value = meta.sampler_key;
            }
            if (meta.loras && meta.loras.length > 0) {
                activeLoras = meta.loras;
                saveAndRenderLoras();
            }
            
            // Display performance specs if available
            if (meta.time_taken || meta.speed) {
                statTime.textContent = meta.time_taken;
                statSpeed.textContent = meta.speed;
                if (statBreakdown) statBreakdown.textContent = meta.breakdown || '';
                perfStats.classList.remove('hidden');
            } else {
                perfStats.classList.add('hidden');
            }
        } else {
            perfStats.classList.add('hidden');
        }
    }

    // Sampler Operations
    async function loadSamplers() {
        try {
            const response = await fetch('/api/samplers');
            const samplers = await response.json();
            samplerSelect.innerHTML = '';
            for (const [key, name] of Object.entries(samplers)) {
                const opt = document.createElement('option');
                opt.value = key;
                opt.textContent = name;
                samplerSelect.appendChild(opt);
            }
            // Set default sampler
            samplerSelect.value = 'euler_a';
            applyTurboDefaults();
        } catch (e) {
            console.error('Error fetching samplers:', e);
        }
    }

    // LoRA Operations
    async function scanLoras() {
        try {
            const response = await fetch('/api/loras');
            const loras = await response.json();
            loraSelect.innerHTML = '';
            
            if (loras.length === 0) {
                loraSelect.innerHTML = '<option value="">No local LoRAs detected in /loras</option>';
                return;
            }
            
            loras.forEach(lora => {
                const opt = document.createElement('option');
                opt.value = lora.name;
                opt.textContent = lora.name;
                loraSelect.appendChild(opt);
            });
        } catch (e) {
            console.error('Error scanning LoRAs:', e);
            loraSelect.innerHTML = '<option value="">Failed to scan /loras folder</option>';
        }
    }

    function saveAndRenderLoras() {
        localStorage.setItem('active_loras', JSON.stringify(activeLoras));
        activeLorasList.innerHTML = '';
        
        if (activeLoras.length === 0) {
            activeLorasList.innerHTML = '<div class="no-history">No active LoRA adapters.</div>';
            return;
        }

        activeLoras.forEach((lora, idx) => {
            const item = document.createElement('div');
            item.className = 'active-lora-item';
            
            // Header row: name + remove button
            const header = document.createElement('div');
            header.className = 'active-lora-header';
            
            const nameSpan = document.createElement('span');
            nameSpan.className = 'active-lora-name';
            nameSpan.textContent = lora.name;
            nameSpan.title = lora.name;
            
            const removeBtn = document.createElement('button');
            removeBtn.className = 'btn-remove';
            removeBtn.textContent = 'SUBTRACT';
            removeBtn.addEventListener('click', () => {
                activeLoras.splice(idx, 1);
                saveAndRenderLoras();
            });
            
            header.appendChild(nameSpan);
            header.appendChild(removeBtn);
            
            // Control row: toggle checkbox + scale slider
            const controls = document.createElement('div');
            controls.className = 'lora-weight-row';
            
            const toggleLabel = document.createElement('label');
            toggleLabel.className = 'lora-toggle-label';
            
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = lora.enabled;
            checkbox.addEventListener('change', () => {
                lora.enabled = checkbox.checked;
                localStorage.setItem('active_loras', JSON.stringify(activeLoras));
            });
            
            toggleLabel.appendChild(checkbox);
            toggleLabel.appendChild(document.createTextNode(' ENABLE'));
            
            const slider = document.createElement('input');
            slider.type = 'range';
            slider.min = '0.0';
            slider.max = '1.5';
            slider.step = '0.1';
            slider.value = lora.scale;
            
            const numSpan = document.createElement('span');
            numSpan.className = 'lora-weight-num';
            numSpan.textContent = parseFloat(lora.scale).toFixed(1);
            
            slider.addEventListener('input', () => {
                lora.scale = parseFloat(slider.value);
                numSpan.textContent = lora.scale.toFixed(1);
                localStorage.setItem('active_loras', JSON.stringify(activeLoras));
            });
            
            controls.appendChild(toggleLabel);
            controls.appendChild(slider);
            controls.appendChild(numSpan);
            
            item.appendChild(header);
            item.appendChild(controls);
            
            activeLorasList.appendChild(item);
        });
    }

    addLoraBtn.addEventListener('click', () => {
        let name = customLoraInput.value.trim();
        if (!name) {
            name = loraSelect.value;
        }
        
        if (!name) {
            alert('Please select a local LoRA or type a custom Hugging Face Repo ID!');
            return;
        }
        
        // Prevent duplicate add
        if (activeLoras.some(l => l.name === name)) {
            alert('LoRA is already added to active list!');
            return;
        }
        
        const scale = parseFloat(loraWeightInput.value);
        activeLoras.push({
            name: name,
            scale: scale,
            enabled: true
        });
        
        customLoraInput.value = '';
        saveAndRenderLoras();
    });

    scanLorasBtn.addEventListener('click', scanLoras);

    // Generate Image Action
    generateBtn.addEventListener('click', async () => {
        const payload = {
            prompt: promptInput.value.trim(),
            negative_prompt: negPromptInput.value.trim(),
            steps: parseInt(stepsInput.value),
            cfg_scale: parseFloat(cfgInput.value),
            width: parseInt(widthSelect.value),
            height: parseInt(heightSelect.value),
            seed: parseInt(seedInput.value),
            model_key: modelSelect.value,
            sampler_key: samplerSelect.value,
            loras: activeLoras,
            init_image: currentInitImage,
            strength: strengthInput ? parseFloat(strengthInput.value) : 0.75,
            edit_mode: editModeSelect ? editModeSelect.value : "edit",
            cfg_cutoff: cfgCutoffInput ? parseFloat(cfgCutoffInput.value) : 1.0,
            live_preview: previewCheckbox ? previewCheckbox.checked : true
        };

        if (!payload.prompt) {
            alert('Please enter a generation prompt!');
            return;
        }

        // Show non-blocking generating badge
        showGeneratingBadge('GENERATING...');
        generateBtn.disabled = true;
        startProgressPolling();

        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            stopProgressPolling();

            if (response.status === 409) {
                hideGeneratingBadge();
                return;
            }
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Failed to generate image');
            }

            const data = await response.json();
            
            // Build temporary meta object for display
            const meta = {
                prompt: payload.prompt,
                negative_prompt: payload.negative_prompt,
                steps: payload.steps,
                cfg_scale: payload.cfg_scale,
                seed: data.seed,
                time_taken: data.time_taken,
                speed: data.speed,
                breakdown: data.breakdown,
                sampler_key: payload.sampler_key
            };
            
            // Swap the final image in place of the live preview instead of sliding
            if (previewShown) resultImg.removeAttribute('src');
            currentHistoryIndex = 0;
            displayImage(data.image_url, meta, 'left');
            await loadHistory();
            hideGeneratingBadge(meta);
        } catch (error) {
            hideGeneratingBadge();
            alert('Error generating image: ' + error.message);
        } finally {
            stopProgressPolling();
            generateBtn.disabled = false;
        }
    });

    // Animate Image to Video with SVD
    const animateSvdBtn = document.getElementById('animate-svd-btn');
    if (animateSvdBtn) {
        animateSvdBtn.addEventListener('click', async () => {
            if (!resultImg.src || resultImg.classList.contains('hidden')) {
                alert('Please generate or select an image first!');
                return;
            }
            
            // Extract filename from URL
            const filename = resultImg.src.split('/').pop();
            
            showGeneratingBadge('RENDERING SVD VIDEO...');
            animateSvdBtn.disabled = true;

            try {
                const response = await fetch('/api/animate-video', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_filename: filename,
                        num_frames: 25,
                        fps: 12,
                        motion_bucket_id: 127
                    })
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Failed to render video');
                }

                const data = await response.json();
                hideGeneratingBadge({ time_taken: data.time_taken, speed: 'SVD' });
                alert(`Video rendered successfully in ${data.time_taken}! Opening video...`);
                window.open(data.video_url, '_blank');
            } catch (err) {
                console.error('Video generation failed:', err);
                hideGeneratingBadge();
                alert(`Video animation error: ${err.message}`);
            } finally {
                animateSvdBtn.disabled = false;
            }
        });
    }

    // Reuse seed action
    reuseSeedBtn.addEventListener('click', () => {
        if (currentImageSeed !== -1) {
            seedInput.value = currentImageSeed;
        }
    });

    // Fullscreen View Interactions
    function openFullscreenViewer(url, meta) {
        if (!url || url.includes('data:image/gif')) return;
        zoomImg.src = url;
        zoomBackdrop.classList.remove('hidden');
        zoomBackdrop.focus();
        updateZoomHud(url, meta);
    }

    // Click result image to zoom / enter ultra-viewport fullscreen
    resultImg.addEventListener('click', () => {
        if (resultImg.src && !resultImg.classList.contains('hidden')) {
            const activeMeta = (allHistoryItems[currentHistoryIndex] && allHistoryItems[currentHistoryIndex].meta) || {
                prompt: promptInput ? promptInput.value : '',
                steps: stepsInput ? stepsInput.value : '',
                cfg_scale: cfgInput ? cfgInput.value : '',
                seed: currentImageSeed,
                sampler_key: samplerSelect ? samplerSelect.value : '',
                time_taken: statTime ? statTime.textContent : '',
                speed: statSpeed ? statSpeed.textContent : ''
            };
            openFullscreenViewer(resultImg.src, activeMeta);
        }
    });

    if (zoomCloseBtn) {
        zoomCloseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            zoomBackdrop.classList.add('hidden');
            zoomImg.src = '';
        });
    }

    if (zoomSideToggle && zoomSideCol) {
        zoomSideToggle.addEventListener('click', (e) => {
            e.stopPropagation();
            const isCollapsed = zoomSideCol.classList.toggle('collapsed');
            zoomSideToggle.textContent = isCollapsed ? '▶' : '◀';
            zoomSideToggle.title = isCollapsed ? 'Expand panel' : 'Collapse panel';
            if (zoomBackdrop) {
                zoomBackdrop.classList.toggle('panel-collapsed', isCollapsed);
            }
        });
    }

    if (zoomMetaPrompt) {
        zoomMetaPrompt.addEventListener('click', (e) => {
            e.stopPropagation();
            const text = zoomMetaPrompt.textContent;
            if (text && text !== 'No prompt directives' && !text.includes('COPIED')) {
                navigator.clipboard.writeText(text).then(() => {
                    const original = zoomMetaPrompt.textContent;
                    zoomMetaPrompt.textContent = '✓ COPIED TO CLIPBOARD!';
                    setTimeout(() => {
                        zoomMetaPrompt.textContent = original;
                    }, 1200);
                }).catch(() => {});
            }
        });
    }

    if (zoomBackdrop) {
        zoomBackdrop.addEventListener('click', (e) => {
            if (zoomSideCol && zoomSideCol.contains(e.target)) return;
            if (e.target === zoomBackdrop || e.target.classList.contains('zoom-stage')) {
                zoomBackdrop.classList.add('hidden');
                zoomImg.src = '';
            }
        });
    }

    if (zoomRegenBtn) {
        zoomRegenBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            generateBtn.click();
        });
    }

    if (zoomUseImg2imgBtn) {
        zoomUseImg2imgBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (zoomImg.src) {
                setInitImage(zoomImg.src);
                zoomBackdrop.classList.add('hidden');
            }
        });
    }

    // Backtick key trigger & Fullscreen keyboard navigation
    window.addEventListener('keydown', (e) => {
        if (e.key === '`') {
            const activeElem = document.activeElement;
            if (activeElem && (activeElem.tagName === 'INPUT' || activeElem.tagName === 'TEXTAREA')) {
                e.preventDefault();
            }
            generateBtn.click();
            return;
        }

        // Inside fullscreen viewer shortcuts
        if (zoomBackdrop && !zoomBackdrop.classList.contains('hidden')) {
            if (e.key === 'Escape') {
                zoomBackdrop.classList.add('hidden');
                zoomImg.src = '';
            } else if (e.key === 'ArrowLeft') {
                // Navigate to older image in history
                if (allHistoryItems.length > 0 && currentHistoryIndex < allHistoryItems.length - 1) {
                    currentHistoryIndex++;
                    const item = allHistoryItems[currentHistoryIndex];
                    displayImage(item.image_url, item.meta, 'right');
                }
            } else if (e.key === 'ArrowRight') {
                // Navigate to newer image in history
                if (allHistoryItems.length > 0 && currentHistoryIndex > 0) {
                    currentHistoryIndex--;
                    const item = allHistoryItems[currentHistoryIndex];
                    displayImage(item.image_url, item.meta, 'left');
                }
            } else if (e.key.toLowerCase() === 'd' && !e.ctrlKey && !e.metaKey) {
                if (zoomDownloadBtn && zoomDownloadBtn.href) {
                    zoomDownloadBtn.click();
                }
            } else if (e.key.toLowerCase() === 'i' && !e.ctrlKey && !e.metaKey) {
                if (zoomUseImg2imgBtn) {
                    zoomUseImg2imgBtn.click();
                }
            }
        }
    });

    // Aspect Ratio Presets
    const aspectBtns = document.querySelectorAll('.aspect-btn');
    function syncAspectButtons(w, h) {
        let matched = false;
        aspectBtns.forEach(btn => {
            if (btn.dataset.w === String(w) && btn.dataset.h === String(h)) {
                btn.classList.add('active');
                matched = true;
            } else {
                btn.classList.remove('active');
            }
        });
        if (!matched) {
            aspectBtns.forEach(btn => btn.classList.remove('active'));
        }
    }

    aspectBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            aspectBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            if (widthSelect && btn.dataset.w) widthSelect.value = btn.dataset.w;
            if (heightSelect && btn.dataset.h) heightSelect.value = btn.dataset.h;
        });
    });

    if (widthSelect) {
        widthSelect.addEventListener('change', () => {
            syncAspectButtons(widthSelect.value, heightSelect ? heightSelect.value : 1024);
        });
    }
    if (heightSelect) {
        heightSelect.addEventListener('change', () => {
            syncAspectButtons(widthSelect ? widthSelect.value : 1024, heightSelect.value);
        });
    }
    // Initialize aspect preset active state
    if (widthSelect && heightSelect) {
        syncAspectButtons(widthSelect.value, heightSelect.value);
    }

    // Init
    loadModels();
    loadSamplers();
    scanLoras();
    saveAndRenderLoras();
    loadHistory();

    // Benchmark Test Matrix Logic
    const openTestModalBtn = document.getElementById('open-test-matrix-btn');
    const closeTestModalBtn = document.getElementById('close-test-modal-btn');
    const testMatrixModal = document.getElementById('test-matrix-modal');
    const testModelSelect = document.getElementById('test-model-select');
    const testPromptInput = document.getElementById('test-prompt');
    const runTestSuiteBtn = document.getElementById('run-test-suite-btn');
    const testStatusMessage = document.getElementById('test-status-message');
    const testStatusText = document.getElementById('test-status-text');
    const testFolderSelect = document.getElementById('test-folder-select');
    const testMatrixGrid = document.getElementById('test-matrix-grid');

    let allTestRuns = {};

    openTestModalBtn.addEventListener('click', () => {
        // Populate model choices in test modal
        testModelSelect.innerHTML = '';
        for (const [key, model] of Object.entries(modelsData)) {
            const opt = document.createElement('option');
            opt.value = key;
            opt.textContent = model.name;
            testModelSelect.appendChild(opt);
        }
        if (modelSelect.value) testModelSelect.value = modelSelect.value;
        
        loadTestResults();
        testMatrixModal.classList.remove('hidden');
    });

    closeTestModalBtn.addEventListener('click', () => {
        testMatrixModal.classList.add('hidden');
    });

    async function loadTestResults() {
        try {
            const res = await fetch('/api/test-results');
            allTestRuns = await res.json();
            
            testFolderSelect.innerHTML = '<option value="">Select Test Checkpoint Output Folder...</option>';
            for (const folderName of Object.keys(allTestRuns)) {
                const opt = document.createElement('option');
                opt.value = folderName;
                opt.textContent = folderName;
                testFolderSelect.appendChild(opt);
            }

            // Auto-select latest folder if available
            const folders = Object.keys(allTestRuns);
            if (folders.length > 0) {
                testFolderSelect.value = folders[folders.length - 1];
                renderTestFolder(folders[folders.length - 1]);
            }
        } catch (err) {
            console.error('Error fetching test results:', err);
        }
    }

    function renderTestFolder(folderName) {
        testMatrixGrid.innerHTML = '';
        const items = allTestRuns[folderName];
        if (!items || items.length === 0) {
            testMatrixGrid.innerHTML = '<p style="color:#8892b0;">No test results in this folder.</p>';
            return;
        }

        items.forEach(item => {
            const card = document.createElement('div');
            card.className = 'test-card';

            const img = document.createElement('img');
            img.src = item.image_url;
            img.alt = item.sampler_name;
            img.addEventListener('click', () => {
                zoomImg.src = item.image_url;
                zoomBackdrop.classList.remove('hidden');
            });

            const title = document.createElement('div');
            title.className = 'test-card-title';
            title.textContent = item.sampler_name;

            const meta = document.createElement('div');
            meta.className = 'test-card-meta';
            meta.innerHTML = `<span>${item.meta.time_taken || 'N/A'}</span><span>${item.meta.speed || ''}</span>`;

            card.appendChild(img);
            card.appendChild(title);
            card.appendChild(meta);
            testMatrixGrid.appendChild(card);
        });
    }

    testFolderSelect.addEventListener('change', () => {
        if (testFolderSelect.value) {
            renderTestFolder(testFolderSelect.value);
        }
    });

    runTestSuiteBtn.addEventListener('click', async () => {
        const modelKey = testModelSelect.value;
        const prompt = testPromptInput.value.trim();
        if (!modelKey) return alert('Select a model key.');

        runTestSuiteBtn.disabled = true;
        testStatusMessage.classList.remove('hidden');
        testStatusText.textContent = `Running sampler benchmark suite across all registered schedulers... Please wait.`;

        try {
            const res = await fetch('/api/test-samplers', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_key: modelKey,
                    prompt: prompt,
                    steps: 25,
                    cfg_scale: 6.0,
                    width: 1024,
                    height: 1024
                })
            });

            if (!res.ok) throw new Error(await res.text());
            
            const data = await res.json();
            testStatusText.textContent = `Benchmark completed! Saved output to static/test_output/${data.folder_name}`;
            await loadTestResults();
            if (data.folder_name) {
                testFolderSelect.value = data.folder_name;
                renderTestFolder(data.folder_name);
            }
        } catch (err) {
            alert('Test suite error: ' + err.message);
        } finally {
            runTestSuiteBtn.disabled = false;
            setTimeout(() => testStatusMessage.classList.add('hidden'), 5000);
        }
    });
});
