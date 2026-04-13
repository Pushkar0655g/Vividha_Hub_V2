# 🎬 Vividha Hub V2
AI-Powered Speaker-Aware Video Dubbing Engine

---

## 🚀 Overview
Vividha Hub V2 is a desktop application that converts videos into multilingual dubbed versions while preserving speaker identity and background music.

---

## 🎯 Problem
Manual dubbing is expensive, slow, and requires human effort.

## ✅ Solution
This project automates:
- Speech → Text
- Text → Translation
- Translation → Speaker-aware audio
- Audio → Final dubbed video

---

## ⚙️ Key Features

- 🎙 Speaker-aware dubbing  
- 🌍 Multi-language translation  
- 🎵 Background music preservation  
- 🧠 Automatic subtitle generation  
- ⚡ Smart timing & sync correction  
- 🖥 Desktop application (Electron + Python)

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
- Python
- Whisper
- Pyannote
- Demucs
- Edge-TTS
- FFmpeg
- Librosa
- Deep Translator

### Frontend
- Electron
- HTML, CSS, JavaScript

### System
- PyTorch (GPU acceleration)

---

## 🔄 V1 vs V2

| Feature | V1 | V2 |
|--------|----|----|
| Transcription | ✅ | ✅ |
| Translation | ✅ | ✅ |
| Subtitles | ✅ | ✅ |
| Speaker-aware dubbing | ❌ | ✅ |
| Background music preservation | ❌ | ✅ |
| Audio sync optimization | ❌ | ✅ |
| Desktop app | ❌ | ✅ |

---

## 🛠 Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/Vividha-Hub.git
cd Vividha-Hub