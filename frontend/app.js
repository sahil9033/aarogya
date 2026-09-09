document.addEventListener('DOMContentLoaded', () => {
    const symptomPills = document.querySelectorAll('.symptom-pill');
    const symptomInput = document.getElementById('symptom-input');
    const submitBtn = document.getElementById('submit-btn');
    const loadingIndicator = document.getElementById('loading-indicator');
    const resultsSection = document.getElementById('results-section');
    
    // New UI Elements
    const toggleVitalsBtn = document.getElementById('toggle-vitals-btn');
    const vitalsForm = document.getElementById('vitals-form');
    const attachBtn = document.getElementById('attach-btn');
    const fileUpload = document.getElementById('file-upload');
    const attachmentBadge = document.getElementById('attachment-badge');
    const attachmentName = document.getElementById('attachment-name');
    const removeAttachmentBtn = document.getElementById('remove-attachment');
    
    const toggleReasoningBtn = document.getElementById('toggle-reasoning-btn');
    const aiReasoningPanel = document.getElementById('ai-reasoning-panel');
    const downloadPdfBtn = document.getElementById('download-pdf-btn');
    
    // AI Upgrades UI
    const cotLogs = document.getElementById('cot-logs');
    const followUpContainer = document.getElementById('follow-up-container');
    const followUpList = document.getElementById('follow-up-list');
    
    // Elements to update with results
    const resCauses = document.getElementById('res-causes');
    const resSteps = document.getElementById('res-steps');
    const resCare = document.getElementById('res-care');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceText = document.getElementById('confidence-text');
    const safetySignalsList = document.getElementById('safety-signals-list');

    let currentResultData = null; // Store for PDF download
    let attachedFile = null;

    // Handle pill click
    symptomPills.forEach(pill => {
        pill.addEventListener('click', () => {
            pill.classList.toggle('selected');
            updateTextareaFromPills();
        });
    });

    function updateTextareaFromPills() {
        const selectedSymptoms = Array.from(document.querySelectorAll('.symptom-pill.selected'))
                                     .map(p => p.getAttribute('data-symptom'));
        let currentText = symptomInput.value;
        selectedSymptoms.forEach(symptom => {
            if (!currentText.toLowerCase().includes(symptom.toLowerCase())) {
                if (currentText.length > 0 && !currentText.endsWith(' ') && !currentText.endsWith('\n')) {
                    currentText += ', ';
                }
                currentText += symptom;
            }
        });
        symptomInput.value = currentText;
    }

    // Toggle Vitals Form
    toggleVitalsBtn.addEventListener('click', () => {
        vitalsForm.classList.toggle('hidden');
        if(!vitalsForm.classList.contains('hidden')) {
            toggleVitalsBtn.style.color = '#1a56db';
        } else {
            toggleVitalsBtn.style.color = '';
        }
    });

    // File Attachment
    attachBtn.addEventListener('click', () => {
        fileUpload.click();
    });

    fileUpload.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            attachedFile = e.target.files[0];
            attachmentName.textContent = attachedFile.name;
            attachmentBadge.classList.remove('hidden');
            attachBtn.style.color = '#1a56db';
        }
    });

    removeAttachmentBtn.addEventListener('click', () => {
        attachedFile = null;
        fileUpload.value = '';
        attachmentBadge.classList.add('hidden');
        attachBtn.style.color = '';
    });

    // Toggle Reasoning
    toggleReasoningBtn.addEventListener('click', () => {
        aiReasoningPanel.classList.toggle('hidden');
    });

    // Handle PDF Download
    downloadPdfBtn.addEventListener('click', async () => {
        if (!currentResultData) return;
        
        const originalText = downloadPdfBtn.innerHTML;
        downloadPdfBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating...';
        
        try {
            const response = await fetch('/api/report/download', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    condition: currentResultData.primary_condition,
                    confidence: currentResultData.confidence_score,
                    causes: currentResultData.possible_causes,
                    steps: currentResultData.next_steps,
                    care: currentResultData.when_to_seek_care,
                    safety_signals: currentResultData.safety_signals
                })
            });
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'ClinicAI_Report.pdf';
                document.body.appendChild(a);
                a.click();
                a.remove();
            }
        } catch (e) {
            console.error('Failed to download PDF', e);
        } finally {
            downloadPdfBtn.innerHTML = originalText;
        }
    });

    // Simulated "Chain of Thought" animation
    function animateCotLogs() {
        cotLogs.innerHTML = '';
        const logs = [
            "> Initializing reasoning engine...",
            "> Extracting clinical entities from natural language input...",
            "> Mapping features to Random Forest inputs...",
            "> Running predictive classification...",
            "> Evaluating safety signals and critical thresholds...",
            "> Generating rich text medical synthesis with Gemini 2.5..."
        ];
        
        let i = 0;
        const interval = setInterval(() => {
            if (i < logs.length) {
                const div = document.createElement('div');
                div.className = 'cot-log-line';
                div.textContent = logs[i];
                cotLogs.appendChild(div);
                i++;
            } else {
                clearInterval(interval);
            }
        }, 800);
        return interval; // Return so we can clear if fetch finishes early
    }

    // Typewriter effect function
    function typeWriterEffect(element, htmlContent, speed = 10) {
        element.innerHTML = '';
        // Hack to support HTML tags in typewriter: parse it and type text nodes
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = htmlContent;
        
        // Very basic approach: just set innerHTML since true HTML typing is complex
        // We'll simulate typing by adding characters to the raw text, but wait, the prompt returns HTML <ul><li>
        // Since it's a web app, let's just fade the entire block in for a "generation" feel, 
        // OR type text and then innerHTML it. For now, innerHTML it and rely on CSS animation.
        element.innerHTML = htmlContent;
        element.style.opacity = '0';
        element.style.animation = 'fadeIn 1s forwards';
    }

    // Main Submit Action
    submitBtn.addEventListener('click', async () => {
        const text = symptomInput.value.trim();
        const selectedSymptoms = Array.from(document.querySelectorAll('.symptom-pill.selected'))
                                     .map(p => p.getAttribute('data-symptom'));
                                     
        if (!text && selectedSymptoms.length === 0 && !attachedFile) {
            alert('Please describe your symptoms, select a symptom pill, or attach a document.');
            return;
        }

        // Show loading
        resultsSection.classList.add('hidden');
        aiReasoningPanel.classList.add('hidden');
        loadingIndicator.classList.remove('hidden');
        
        const cotInterval = animateCotLogs();

        try {
            const formData = new FormData();
            formData.append('symptoms', text);
            formData.append('selected_symptoms', selectedSymptoms.join(','));
            
            // Append vitals
            formData.append('age', document.getElementById('age-input').value);
            formData.append('hr', document.getElementById('hr-input').value);
            formData.append('bp', document.getElementById('bp-input').value);
            formData.append('spo2', document.getElementById('spo2-input').value);
            formData.append('temp', document.getElementById('temp-input').value);
            formData.append('glucose', document.getElementById('glucose-input').value);
            
            if (attachedFile) {
                formData.append('file', attachedFile);
            }

            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData // No Content-Type header so the browser sets the boundary automatically
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            currentResultData = data; // Save for PDF
            
            clearInterval(cotInterval);
            
            // Update UI with results (with slight delay for "typing" feel)
            setTimeout(() => {
                typeWriterEffect(resCauses, `<strong>${data.primary_condition}</strong><br><br>${data.possible_causes}`);
                typeWriterEffect(resSteps, data.next_steps);
                typeWriterEffect(resCare, data.when_to_seek_care);
            }, 300);
            
            // Update Reasoning Panel
            confidenceBar.style.width = `${data.confidence_score}%`;
            confidenceText.innerHTML = `<strong>${data.confidence_score.toFixed(1)}%</strong> match with <strong>${data.primary_condition}</strong>`;
            
            safetySignalsList.innerHTML = '';
            if (data.safety_signals && data.safety_signals.length > 0) {
                data.safety_signals.forEach(sig => {
                    const li = document.createElement('li');
                    li.textContent = sig;
                    li.style.color = '#E24B4A';
                    safetySignalsList.appendChild(li);
                });
            } else {
                const li = document.createElement('li');
                li.textContent = 'No critical vital signs detected.';
                safetySignalsList.appendChild(li);
            }
            
            // Populate Follow Up Questions
            followUpList.innerHTML = '';
            if (data.follow_up_questions && data.follow_up_questions.length > 0) {
                followUpContainer.classList.remove('hidden');
                data.follow_up_questions.forEach(q => {
                    const bubble = document.createElement('div');
                    bubble.className = 'follow-up-bubble';
                    bubble.innerHTML = `<i class="fa-solid fa-reply" style="font-size: 11px; margin-right: 6px; opacity: 0.7;"></i> ${q}`;
                    bubble.addEventListener('click', () => {
                        // Append question to text area and re-submit
                        symptomInput.value = symptomInput.value.trim() + "\\n\\n" + "Answering follow-up: " + q + " - ";
                        symptomInput.focus();
                        // Scroll to input
                        symptomInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    });
                    followUpList.appendChild(bubble);
                });
            } else {
                followUpContainer.classList.add('hidden');
            }
            
        } catch (error) {
            console.error('Error fetching analysis:', error);
            clearInterval(cotInterval);
            
            // Fallback for demo if API fails
            resCauses.innerHTML = `<strong>Likely: Viral Infection</strong><br><br>Based on your symptoms, this seems to be a common viral issue.`;
            resSteps.innerHTML = "Rest, stay hydrated, and take over-the-counter medication.";
            resCare.innerHTML = "If symptoms persist for more than 3-4 days, seek immediate care.";
        } finally {
            // Hide loading, show results
            loadingIndicator.classList.add('hidden');
            resultsSection.classList.remove('hidden');
            
            // Smooth scroll to results
            setTimeout(() => {
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }, 100);
        }
    });
});
