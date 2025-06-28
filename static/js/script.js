document.getElementById('url-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const urlInput = document.getElementById('url-input');
    const resultDiv = document.getElementById('result');
    const url = urlInput.value.trim();
    
    if (!url) {
        resultDiv.className = 'result phishing';
        resultDiv.innerHTML = 'Error: Please enter a URL';
        resultDiv.style.display = 'block';
        return;
    }

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });
        const data = await response.json();
        
        // Handle array response from /predict
        const result = Array.isArray(data) ? data[0] : data;
        
        if (result.error) {
            resultDiv.className = 'result phishing';
            resultDiv.innerHTML = `Error: ${result.error}`;
        } else {
            resultDiv.className = `result ${result.result.toLowerCase()}`;
            resultDiv.innerHTML = `
                <strong>URL:</strong> ${result.url}<br>
                <strong>Result:</strong> ${result.result}<br>
                <strong>Confidence:</strong> ${result.confidence}%
            `;
        }
        resultDiv.style.display = 'block';
    } catch (error) {
        resultDiv.className = 'result phishing';
        resultDiv.innerHTML = `Error: Unable to connect to server - ${error.message}`;
        resultDiv.style.display = 'block';
    }
});