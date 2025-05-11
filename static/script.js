document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const dropArea = document.getElementById('dropArea');
    const uploadContent = document.getElementById('uploadContent');
    const browseBtn = document.getElementById('browseBtn');
    const sampleBtn = document.getElementById('sampleBtn');
    const changeImageBtn = document.getElementById('changeImageBtn');
    const fileInput = document.getElementById('fileInput');
    const uploadedImageContainer = document.getElementById('uploadedImageContainer');
    const uploadedImage = document.getElementById('uploadedImage');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const resultsPlaceholder = document.getElementById('resultsPlaceholder');
    const predictionResults = document.getElementById('predictionResults');
    const topPredictionImg = document.getElementById('topPredictionImg');
    const topBreedName = document.getElementById('topBreedName');
    const topConfidence = document.getElementById('topConfidence');
    const topConfidenceBar = document.getElementById('topConfidenceBar');
    const otherPredictions = document.getElementById('otherPredictions');
    
    // Initialize animation for about cards
    initializeScrollAnimation();
    
    // Event Listeners
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });
    
    sampleBtn.addEventListener('click', handleSampleImageRequest);
    
    changeImageBtn.addEventListener('click', () => {
        resetUploadArea();
        fileInput.click();
    });
    
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.add('drag-over');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.remove('drag-over');
        }, false);
    });
    
    dropArea.addEventListener('drop', handleDrop, false);
    
    // Functions
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length) {
            fileInput.files = files;
            handleFileSelect();
        }
    }
    
    function handleFileSelect() {
        if (fileInput.files && fileInput.files[0]) {
            const file = fileInput.files[0];
            
            // Check if file is an image
            if (!file.type.match('image.*')) {
                showError("Please select an image file");
                return;
            }
            
            // Display uploaded image
            const reader = new FileReader();
            reader.onload = function(e) {
                uploadedImage.src = e.target.result;
                uploadedImageContainer.style.display = 'block';
                uploadContent.style.display = 'none';
                
                // Send to server for prediction
                predictDogBreed(file);
            }
            reader.readAsDataURL(file);
        }
    }
    
    function handleSampleImageRequest() {
        // Show loading state
        loadingOverlay.style.display = 'flex';
        resultsPlaceholder.style.display = 'none';
        predictionResults.style.display = 'none';
        
        // Request a sample image from the server
        fetch('/get_sample_image', {
            method: 'GET'
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                showError(data.error);
                loadingOverlay.style.display = 'none';
                return;
            }
            
            // Display the sample image
            uploadedImage.src = data.image_data;
            uploadedImageContainer.style.display = 'block';
            uploadContent.style.display = 'none';
            
            // Process the predictions
            displayResults(data);
            
            // Hide loading
            loadingOverlay.style.display = 'none';
        })
        .catch(error => {
            console.error('Error:', error);
            loadingOverlay.style.display = 'none';
            showError("Failed to load sample image");
        });
    }
    
    function predictDogBreed(file) {
        // Show loading
        loadingOverlay.style.display = 'flex';
        resultsPlaceholder.style.display = 'none';
        predictionResults.style.display = 'none';
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        // Send to Flask backend
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading
            loadingOverlay.style.display = 'none';
            
            if (data.error) {
                showError(data.error);
                return;
            }
            
            // Display results
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            loadingOverlay.style.display = 'none';
            showError("An error occurred during prediction");
        });
    }
    
    function displayResults(data) {
        if (!data.predictions || data.predictions.length === 0) {
            showError("No predictions available");
            return;
        }
        
        // Top prediction
        const topPrediction = data.predictions[0];
        topPredictionImg.src = topPrediction.sample_image || '';
        topBreedName.textContent = formatBreedName(topPrediction.breed);
        topConfidence.textContent = topPrediction.confidence;
        
        // Animate confidence bar
        const confidenceValue = parseFloat(topPrediction.confidence);
        setTimeout(() => {
            topConfidenceBar.style.width = `${confidenceValue}%`;
        }, 100);
        
        // Other predictions
        otherPredictions.innerHTML = '';
        if (data.predictions.length > 1) {
            for (let i = 1; i < data.predictions.length; i++) {
                const pred = data.predictions[i];
                const confidenceValue = parseFloat(pred.confidence);
                
                const predictionCard = document.createElement('div');
                predictionCard.className = 'prediction-card';
                predictionCard.innerHTML = `
                    <img src="${pred.sample_image || ''}" alt="${pred.breed}">
                    <div class="prediction-card-info">
                        <div class="prediction-card-breed">${formatBreedName(pred.breed)}</div>
                        <div class="prediction-card-confidence">Confidence: ${pred.confidence}</div>
                        <div class="prediction-card-bar-container">
                            <div class="prediction-card-bar"></div>
                        </div>
                    </div>
                `;
                
                otherPredictions.appendChild(predictionCard);
                
                // Animate confidence bar with delay
                setTimeout(() => {
                    const bar = predictionCard.querySelector('.prediction-card-bar');
                    bar.style.width = `${confidenceValue}%`;
                }, 300 + i * 200);
            }
        }
        
        // Show results
        predictionResults.style.display = 'flex';
        resultsPlaceholder.style.display = 'none';
    }
    
    function formatBreedName(breed) {
        // Capitalize each word and replace underscores with spaces
        return breed
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }
    
    function showError(message) {
        resultsPlaceholder.innerHTML = `
            <i class="fas fa-exclamation-circle" style="color: #ff6b6b; font-size: 3rem;"></i>
            <p>${message}</p>
        `;
        resultsPlaceholder.style.display = 'flex';
        predictionResults.style.display = 'none';
    }
    
    function resetUploadArea() {
        uploadedImageContainer.style.display = 'none';
        uploadContent.style.display = 'flex';
        fileInput.value = '';
        resultsPlaceholder.style.display = 'flex';
        predictionResults.style.display = 'none';
    }
    
    function initializeScrollAnimation() {
        // Add scroll animation for about cards
        const aboutCards = document.querySelectorAll('.about-card');
        
        // Check if IntersectionObserver is supported
        if ('IntersectionObserver' in window) {
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('fade-in');
                        observer.unobserve(entry.target);
                    }
                });
            }, { threshold: 0.1 });
            
            aboutCards.forEach(card => {
                observer.observe(card);
            });
        } else {
            // Fallback for browsers that don't support IntersectionObserver
            aboutCards.forEach(card => {
                card.classList.add('fade-in');
            });
        }
        
        // Add hover effect for prediction cards
        document.addEventListener('mouseover', function(e) {
            if (e.target.closest('.prediction-card')) {
                const cards = document.querySelectorAll('.prediction-card');
                cards.forEach(card => {
                    if (card !== e.target.closest('.prediction-card')) {
                        card.style.opacity = '0.6';
                    }
                });
            }
        });
        
        document.addEventListener('mouseout', function(e) {
            if (e.target.closest('.prediction-card')) {
                const cards = document.querySelectorAll('.prediction-card');
                cards.forEach(card => {
                    card.style.opacity = '1';
                });
            }
        });
    }
});