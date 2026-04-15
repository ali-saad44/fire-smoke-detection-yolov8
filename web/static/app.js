/* ============================================================
   BLAST DETECTION AI — Frontend JavaScript
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
    // --- Elements ---
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const uploadContent = document.getElementById('upload-content');
    const previewArea = document.getElementById('preview-area');
    const previewImage = document.getElementById('preview-image');
    const previewVideo = document.getElementById('preview-video');
    const previewName = document.getElementById('preview-name');
    const previewSize = document.getElementById('preview-size');
    const btnRemove = document.getElementById('btn-remove');
    const detectBtn = document.getElementById('detect-btn');
    const processing = document.getElementById('processing');
    const processingStatus = document.getElementById('processing-status');
    const resultsSection = document.getElementById('results-section');
    const statsGrid = document.getElementById('stats-grid');
    const resultDisplay = document.getElementById('result-display');
    const detectionsTableWrap = document.getElementById('detections-table-wrap');
    const detectionsTbody = document.getElementById('detections-tbody');
    const downloadContainer = document.getElementById('download-container');
    const downloadLink = document.getElementById('download-link');
    const btnNew = document.getElementById('btn-new');

    let selectedFile = null;

    // --- Animate metric values on scroll ---
    const animateMetrics = () => {
        const metricValues = document.querySelectorAll('.metric-value[data-value]');
        metricValues.forEach(el => {
            const target = parseFloat(el.dataset.value);
            const duration = 1500;
            const start = performance.now();

            const animate = (now) => {
                const elapsed = now - start;
                const progress = Math.min(elapsed / duration, 1);
                const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
                const current = (target * eased).toFixed(1);
                el.textContent = current;
                if (progress < 1) requestAnimationFrame(animate);
            };
            requestAnimationFrame(animate);
        });
    };

    // Trigger metric animation when section is visible
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateMetrics();
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.3 });

    const modelSection = document.getElementById('model-section');
    if (modelSection) observer.observe(modelSection);

    // --- File Upload ---
    uploadArea.addEventListener('click', (e) => {
        if (e.target === btnRemove || e.target.closest('.btn-remove')) return;
        if (!selectedFile) fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) handleFile(e.target.files[0]);
    });

    // Drag & Drop
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
        if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
    });

    function handleFile(file) {
        selectedFile = file;
        uploadContent.style.display = 'none';
        previewArea.style.display = 'block';

        previewName.textContent = file.name;
        previewSize.textContent = formatSize(file.size);

        const ext = file.name.split('.').pop().toLowerCase();
        const videoExts = ['mp4', 'avi', 'mov', 'mkv', 'wmv'];

        if (videoExts.includes(ext)) {
            previewImage.style.display = 'none';
            previewVideo.style.display = 'block';
            previewVideo.src = URL.createObjectURL(file);
        } else {
            previewVideo.style.display = 'none';
            previewImage.style.display = 'block';
            previewImage.src = URL.createObjectURL(file);
        }

        detectBtn.disabled = false;
    }

    btnRemove.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        uploadContent.style.display = 'block';
        previewArea.style.display = 'none';
        previewImage.style.display = 'none';
        previewVideo.style.display = 'none';
        previewImage.src = '';
        previewVideo.src = '';
        detectBtn.disabled = true;
    }

    function formatSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    // --- Run Detection ---
    detectBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Show processing
        detectBtn.style.display = 'none';
        uploadArea.style.display = 'none';
        processing.style.display = 'block';
        resultsSection.style.display = 'none';

        const ext = selectedFile.name.split('.').pop().toLowerCase();
        const isVideo = ['mp4', 'avi', 'mov', 'mkv', 'wmv'].includes(ext);

        if (isVideo) {
            processingStatus.textContent = 'Processing video frames with YOLOv8... This may take a few minutes.';
        } else {
            processingStatus.textContent = 'Running YOLOv8 inference on image...';
        }

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Detection failed');
            }

            showResults(data);

        } catch (err) {
            alert('Error: ' + err.message);
            processing.style.display = 'none';
            detectBtn.style.display = 'flex';
            uploadArea.style.display = 'block';
        }
    });

    // --- Show Results ---
    function showResults(data) {
        processing.style.display = 'none';
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });

        // Build stats cards
        statsGrid.innerHTML = '';

        if (data.type === 'image') {
            downloadContainer.style.display = 'none';
            addStatCard('Detections', data.total_detections);
            addStatCard('Time', data.processing_time + 's');
            addStatCard('Type', 'Image');
            addStatCard('Timestamp', data.timestamp.split(' ')[1]);

            // Show annotated image
            resultDisplay.innerHTML = `<img src="${data.result_url}" alt="Detection result" style="max-width:100%; border-radius:12px;">`;

            // Detections table
            if (data.detections && data.detections.length > 0) {
                detectionsTableWrap.style.display = 'block';
                detectionsTbody.innerHTML = '';

                data.detections.forEach((d, i) => {
                    const classType = d.class.toLowerCase().includes('fire') ? 'fireball' : 'smoke';
                    const confPct = (d.confidence * 100).toFixed(1);
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${i + 1}</td>
                        <td><span class="class-badge ${classType}">${d.class}</span></td>
                        <td>
                            <div class="conf-bar">
                                <span>${confPct}%</span>
                                <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:${confPct}%"></div></div>
                            </div>
                        </td>
                        <td style="font-size:0.8rem; color:var(--text-muted);">[${d.bbox.join(', ')}]</td>
                    `;
                    detectionsTbody.appendChild(row);
                });
            } else {
                detectionsTableWrap.style.display = 'block';
                detectionsTbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:var(--text-muted); padding:24px;">No detections found — the scene appears clear.</td></tr>';
            }

        } else if (data.type === 'video') {
            const s = data.stats;
            addStatCard('Frames', s.total_frames);
            addStatCard('Detections', s.total_detections);
            addStatCard('Detection Rate', s.detection_rate + '%');
            addStatCard('Time', data.processing_time + 's');

            if (s.class_stats) {
                Object.entries(s.class_stats).forEach(([cls, count]) => {
                    if (count > 0) addStatCard(cls, count);
                });
            }

            // Show annotated video
            resultDisplay.innerHTML = `
                <video controls autoplay style="max-width:100%; border-radius:12px;">
                    <source src="${data.result_url}" type="video/mp4">
                    Your browser does not support direct playback (Missing H.264 codec).
                </video>
                <p style="margin-top:12px; color:var(--text-secondary); font-size:0.85rem;">
                    ${s.resolution} • ${s.fps} FPS • ${s.total_frames} frames analyzed
                </p>
            `;

            // Prepare download button
            downloadLink.href = data.result_url;
            downloadLink.download = `detected_blast_${data.timestamp.replace(/[: ]/g, '_')}.mp4`;
            downloadContainer.style.display = 'block';

            detectionsTableWrap.style.display = 'none';
        }
    }

    function addStatCard(label, value) {
        const card = document.createElement('div');
        card.className = 'stat-card';
        card.innerHTML = `
            <div class="stat-value">${value}</div>
            <div class="stat-label">${label}</div>
        `;
        statsGrid.appendChild(card);
    }

    // --- New Analysis ---
    btnNew.addEventListener('click', () => {
        resultsSection.style.display = 'none';
        uploadArea.style.display = 'block';
        detectBtn.style.display = 'flex';
        detectionsTableWrap.style.display = 'none';
        resetUpload();

        document.getElementById('upload-section').scrollIntoView({ behavior: 'smooth' });
    });
});
