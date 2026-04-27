# Media Art Audio Analysis Backend

This project is a Python-based backend for media art installations. It performs real-time audio detection, transcription, and emotion analysis.

## Features
- **Real-time VAD**: Voice Activity Detection to capture speech.
- **Whisper STT**: Local speech-to-text using OpenAI's Whisper.
- **Emotion Analysis**: Calculates Valence and Arousal scores.
- **OSC Protocol**: Transmits analysis results to media art software (e.g., TouchDesigner, Unity) via OSC.

## Setup
1. Install Python 3.11.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables in `.env` (use `.env.example` as a template).

## Usage
Run the backend:
```bash
python main.py
```
Or use the provided `start.bat` on Windows.
