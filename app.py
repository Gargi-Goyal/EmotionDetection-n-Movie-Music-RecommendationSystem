from flask import Flask, render_template, request, jsonify
from model.model import load_model
import torchvision.transforms as transforms
from PIL import Image
import io
import torch

app = Flask(__name__, static_url_path='/static')

# Load model from ROOT folder
model = load_model('emotion_model.pth')

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
recommendations = {
    'Happy': '🎶 Song: Happy by Pharrell Williams, 🎬 Movie: Inside Out',
    'Sad': '🎶 Song: Fix You by Coldplay, 🎬 Movie: The Pursuit of Happyness',
    'Angry': '🎶 Song: In The End by Linkin Park, 🎬 Movie: Gladiator',
    'Fear': '🎶 Song: Fear of the Dark by Iron Maiden, 🎬 Movie: A Quiet Place',
    'Surprise': '🎶 Song: Surprise Yourself by Jack Garratt, 🎬 Movie: Now You See Me',
    'Disgust': '🎶 Song: I Don’t Care by Ed Sheeran, 🎬 Movie: Contagion',
    'Neutral': '🎶 Song: Let it Be by The Beatles, 🎬 Movie: Forrest Gump'
}

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        predicted_class = torch.argmax(outputs, dim=1).item()
        predicted_emotion = emotions[predicted_class]

    return jsonify({
        'emotion': predicted_emotion,
        'recommendation': recommendations.get(predicted_emotion, "🎵🎬 No recommendation")
    })

if __name__ == '__main__':
    app.run(debug=True)
