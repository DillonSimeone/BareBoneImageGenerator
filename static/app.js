document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const modelSelect = document.getElementById('model-select');
    const modelDesc = document.getElementById('model-desc');
    const promptInput = document.getElementById('prompt');
    const negPromptInput = document.getElementById('neg-prompt');
    const stepsInput = document.getElementById('steps');
    const stepsVal = document.getElementById('steps-val');
    const cfgInput = document.getElementById('cfg');
    const cfgVal = document.getElementById('cfg-val');
    const widthSelect = document.getElementById('width');
    const heightSelect = document.getElementById('height');
    const seedInput = document.getElementById('seed');
    
    const generateBtn = document.getElementById('generate-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const placeholderView = document.getElementById('placeholder-view');
    const resultImg = document.getElementById('result-img');
    const imageActions = document.getElementById('image-actions');
    const downloadLink = document.getElementById('download-link');
    const reuseSeedBtn = document.getElementById('reuse-seed-btn');
    const historyGallery = document.getElementById('history-gallery');
    
    // Performance stats elements
    const perfStats = document.getElementById('perf-stats');
    const statTime = document.getElementById('stat-time');
    const statSpeed = document.getElementById('stat-speed');

    let modelsData = {};
    let currentImageSeed = -1;

    // Listeners for slider updates
    stepsInput.addEventListener('input', () => {
        stepsVal.textContent = stepsInput.value;
    });

    cfgInput.addEventListener('input', () => {
        cfgVal.textContent = parseFloat(cfgInput.value).toFixed(1);
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
            
            // Set initial defaults
            updateModelDefaults();
        } catch (error) {
            console.error('Error fetching models:', error);
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
    }

    modelSelect.addEventListener('change', updateModelDefaults);

    // Fetch generation history
    async function loadHistory() {
        try {
            const response = await fetch('/api/history');
            const history = await response.json();
            
            historyGallery.innerHTML = '';
            if (history.length === 0) {
                historyGallery.innerHTML = '<div class="no-history">No generations yet.</div>';
                return;
            }

            history.forEach(item => {
                const div = document.createElement('div');
                div.className = 'history-item';
                
                const img = document.createElement('img');
                img.src = item.image_url;
                img.alt = item.meta.prompt || 'Generated Image';
                
                div.appendChild(img);
                
                div.addEventListener('click', () => {
                    displayImage(item.image_url, item.meta);
                });
                
                historyGallery.appendChild(div);
            });
        } catch (error) {
            console.error('Error loading history:', error);
        }
    }

    function displayImage(url, meta) {
        placeholderView.classList.add('hidden');
        resultImg.src = url;
        resultImg.classList.remove('hidden');
        imageActions.classList.remove('hidden');
        downloadLink.href = url;
        
        if (meta) {
            currentImageSeed = meta.seed || -1;
            if (meta.prompt) promptInput.value = meta.prompt;
            if (meta.negative_prompt) negPromptInput.value = meta.negative_prompt;
            if (meta.steps) {
                stepsInput.value = meta.steps;
                stepsVal.textContent = meta.steps;
            }
            if (meta.cfg_scale) {
                cfgInput.value = meta.cfg_scale;
                cfgVal.textContent = parseFloat(meta.cfg_scale).toFixed(1);
            }
            
            // Display performance specs if available
            if (meta.time_taken || meta.speed) {
                statTime.textContent = meta.time_taken;
                statSpeed.textContent = meta.speed;
                perfStats.classList.remove('hidden');
            } else {
                perfStats.classList.add('hidden');
            }
        } else {
            perfStats.classList.add('hidden');
        }
    }

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
            model_key: modelSelect.value
        };

        if (!payload.prompt) {
            alert('Please enter a generation prompt!');
            return;
        }

        // Show loader
        loadingOverlay.classList.remove('hidden');
        generateBtn.disabled = true;

        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

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
                speed: data.speed
            };
            
            displayImage(data.image_url, meta);
            await loadHistory();
        } catch (error) {
            alert('Error generating image: ' + error.message);
        } finally {
            loadingOverlay.classList.add('hidden');
            generateBtn.disabled = false;
        }
    });

    reuseSeedBtn.addEventListener('click', () => {
        if (currentImageSeed !== -1) {
            seedInput.value = currentImageSeed;
        }
    });

    const zoomBackdrop = document.getElementById('zoom-backdrop');
    const zoomImg = document.getElementById('zoom-img');

    // Click result image to zoom
    resultImg.addEventListener('click', () => {
        if (resultImg.src && !resultImg.classList.contains('hidden')) {
            zoomImg.src = resultImg.src;
            zoomBackdrop.classList.remove('hidden');
        }
    });

    // Click backdrop to close zoom
    zoomBackdrop.addEventListener('click', () => {
        zoomBackdrop.classList.add('hidden');
        zoomImg.src = '';
    });

    // Backtick key trigger
    window.addEventListener('keydown', (e) => {
        if (e.key === '`') {
            const activeElem = document.activeElement;
            if (activeElem && (activeElem.tagName === 'INPUT' || activeElem.tagName === 'TEXTAREA')) {
                // Prevent insertion of backtick character in prompt textareas/fields
                e.preventDefault();
            }
            generateBtn.click();
        }
    });

    // Init
    loadModels();
    loadHistory();
});
