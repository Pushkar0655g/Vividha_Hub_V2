# 🎬 Vividha Hub V2  
AI-Powered Speaker-Aware Video Dubbing Engine  

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Node.js](https://img.shields.io/badge/Node.js-18+-green)
![Platform](https://img.shields.io/badge/Platform-Windows-orange)
![GPU](https://img.shields.io/badge/GPU-CUDA%20Supported-yellow)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

---

## 🚀 Overview

Vividha Hub V2 is a desktop application that converts videos into multilingual dubbed versions while preserving speaker identity and background music.

It provides an end-to-end automated pipeline for video localization using modern AI tools.

---

## 🎯 Problem

Manual video dubbing is expensive, slow, and requires significant human effort.

---

## ✅ Solution

This project automates the complete pipeline:

* Speech → Text
* Text → Translation
* Translation → Speaker-aware audio
* Audio → Final dubbed video

---

## ⚙️ Key Features

* 🎙 Speaker-aware dubbing
* 🌍 Multi-language translation
* 🎵 Background music preservation
* 🧠 Automatic subtitle generation
* ⚡ Smart timing & sync correction
* 🖥 Desktop application (Electron + Python)

---

## 🧠 Architecture

Video
→ Audio Extraction
→ Demucs (vocals + background split)
→ Whisper (transcription)
→ Pyannote (speaker detection)
→ Segment merging & overlap correction
→ Translation
→ Gender detection (librosa)
→ TTS generation (Edge-TTS)
→ Speed adjustment (FFmpeg)
→ Audio merging
→ Background mixing
→ Final video rendering

---

## 🧰 Tech Stack

### Backend

* Python
* Whisper
* Pyannote
* Demucs
* Edge-TTS
* FFmpeg
* Librosa
* Deep Translator

### Frontend

* Electron
* HTML, CSS, JavaScript

### System

* PyTorch (GPU acceleration)

---

## 🔄 V1 vs V2

| Feature                       | V1 | V2 |
| ----------------------------- | -- | -- |
| Transcription                 | ✅  | ✅  |
| Translation                   | ✅  | ✅  |
| Subtitles                     | ✅  | ✅  |
| Speaker-aware dubbing         | ❌  | ✅  |
| Background music preservation | ❌  | ✅  |
| Audio sync optimization       | ❌  | ✅  |
| Desktop application           | ❌  | ✅  |

---

## 📁 Project Structure

```
Vividha-Hub/
├── backend/        # Python backend
├── frontend/       # Electron UI
├── README.md
├── requirements.txt
├── package.json
├── package-lock.json
└── .gitignore
```

---

## 🛠 Prerequisites

Make sure the following are installed:

* Python 3.10 or 3.11
* Node.js and npm
* FFmpeg
* Git
* Hugging Face Token (for Pyannote)

Recommended:

* Windows 10/11
* NVIDIA GPU (for faster processing)

---

## ⚡ Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/Vividha-Hub.git
cd Vividha-Hub

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

set HF_TOKEN=your_token_here

npm install
npm start
```

---

## 🔧 Backend Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
set HF_TOKEN=your_token_here
```

Make sure:

* FFmpeg is installed OR
* `ffmpeg.exe` and `ffprobe.exe` are inside backend folder

---

## 🎨 Frontend Setup

```bash
cd frontend
npm install
npm start
```

---

## ▶️ Run Backend Manually

```bash
python backend/backend.py --video sample.mp4 --input_lang english --audio_lang hindi --subtitle_lang english
```

---

## ⚠️ Known Issues

* Edge-TTS requires internet connection
* Voice output may sound robotic
* Some segments may be skipped
* Processing is slow without GPU

---

## 🧪 Tested On

* Windows 11
* Python 3.11
* NVIDIA RTX 4050
* Electron Desktop Environment

---

## 🚀 Future Scope

* Offline TTS model
* Better voice cloning
* Lip-sync integration
* Improved natural speech

---

## 📦 Download

Executable version can be added in GitHub Releases section (coming soon).

---

## 👨‍💻 Author

Pushkar Chirra
