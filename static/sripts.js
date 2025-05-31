const video = document.getElementById('video');
const resultDiv = document.getElementById('result');

// Access webcam
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => video.srcObject = stream)
  .catch(err => console.error("Camera error:", err));

// Capture and send frame every 3 seconds
setInterval(() => {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);

  canvas.toBlob(blob => {
    const formData = new FormData();
    formData.append('file', blob, 'frame.jpg');

    fetch('/predict', {
      method: 'POST',
      body: formData
    })
    .then(res => res.json())
    .then(data => {
      resultDiv.innerText = `Prediction: ${data.label} (Confidence: ${data.confidence.toFixed(2)})`;
      if (data.label === 'Violence') {
        resultDiv.style.color = 'red';
      } else {
        resultDiv.style.color = 'green';
      }
    })
    .catch(err => {
      console.error("Prediction error:", err);
      resultDiv.innerText = "Error during prediction.";
    });
  }, 'image/jpeg');
}, 2000);

// SOS Button
function sendSOS() {
  alert("🚨 SOS Triggered! (Implement action here)");
}
