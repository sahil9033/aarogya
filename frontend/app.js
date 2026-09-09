document.addEventListener('DOMContentLoaded', () => {
    const symptomPills = document.querySelectorAll('.symptom-pill');
    const symptomInput = document.getElementById('symptom-input');
    const submitBtn = document.getElementById('submit-btn');
    const loadingIndicator = document.getElementById('loading-indicator');
    const resultsSection = document.getElementById('results-section');
    
    // Elements to update with results
    const resCauses = document.getElementById('res-causes');
    const resSteps = document.getElementById('res-steps');
    const resCare = document.getElementById('res-care');

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
        
        // Remove old auto-added text if it exists (simple implementation)
        // For a more robust implementation, we'd want to just append to what the user typed
        
        // This is a simple implementation: we just append selected pills if they aren't in the text
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

    // Handle submit
    submitBtn.addEventListener('click', async () => {
        const text = symptomInput.value.trim();
        const selectedSymptoms = Array.from(document.querySelectorAll('.symptom-pill.selected'))
                                     .map(p => p.getAttribute('data-symptom'));
                                     
        if (!text && selectedSymptoms.length === 0) {
            alert('Please describe your symptoms or select a symptom pill.');
            return;
        }

        // Show loading
        resultsSection.classList.add('hidden');
        loadingIndicator.classList.remove('hidden');

        try {
            // Replace with actual API call to FastAPI backend
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    symptoms: text,
                    selected_symptoms: selectedSymptoms
                })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            
            // Update UI with results
            resCauses.innerHTML = `<strong>${data.primary_condition}</strong><br><br>${data.possible_causes}`;
            resSteps.innerHTML = data.next_steps;
            resCare.innerHTML = data.when_to_seek_care;
            
        } catch (error) {
            console.error('Error fetching analysis:', error);
            // Fallback for demo if API fails
            resCauses.innerHTML = `<strong>Likely: Viral Infection</strong><br><br>Based on your symptoms, this seems to be a common viral issue.`;
            resSteps.innerHTML = "Rest, stay hydrated, and take over-the-counter medication to manage fever and pain.";
            resCare.innerHTML = "If symptoms persist for more than 3-4 days, or if you experience difficulty breathing, seek immediate care.";
        } finally {
            // Hide loading, show results
            loadingIndicator.classList.add('hidden');
            resultsSection.classList.remove('hidden');
            
            // Smooth scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    });
});
