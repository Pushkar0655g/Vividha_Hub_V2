import os
import sys
import shutil
import subprocess
import json
import argparse
import logging
import time
import random
import edge_tts
import asyncio
import torch
import whisper

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ffmpeg_path = os.path.join(BACKEND_DIR, "ffmpeg.exe")

if getattr(sys, "frozen", False):
    BACKEND_DIR = os.path.dirname(sys.executable)
else:
    BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

os.environ["PATH"] = BACKEND_DIR + os.pathsep + os.environ.get("PATH", "")
from pyannote.audio import Pipeline
from deep_translator import GoogleTranslator, MyMemoryTranslator

LOG_FILE = os.path.join(BACKEND_DIR, "conversion.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=LOG_FILE
)

SUPPORTED_EXTENSIONS = [".mp4", ".mkv", ".avi", ".mov", ".wmv"]

LANGUAGE_MAP = {
    "english": "en",
    "hindi": "hi",
    "telugu": "te",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "arabic": "ar",
    "chinese": "zh-cn",
    "japanese": "ja",
    "korean": "ko",
    "portuguese": "pt",
    "russian": "ru",
    "italian": "it",
    "turkish": "tr",
    "dutch": "nl",
    "polish": "pl",
}

VOICE_MAP = {
    "te": {
        "male": "te-IN-MohanNeural",
        "female": "te-IN-ShrutiNeural"
    },
    "hi": {
        "male": "hi-IN-MadhurNeural",
        "female": "hi-IN-SwaraNeural"
    },
    "en": {
        "male": "en-US-GuyNeural",
        "female": "en-US-JennyNeural"
    }
}

_GENDER_CLASSIFIER = None
_XTTS_MODEL = None


def get_gender_classifier():
    global _GENDER_CLASSIFIER
    if _GENDER_CLASSIFIER is None:
        from speechbrain.inference.classifiers import EncoderClassifier
        _GENDER_CLASSIFIER = EncoderClassifier.from_hparams(
            source="speechbrain/urbansound8k_ecapa",
            savedir=os.path.join(BACKEND_DIR, "pretrained_models", "gender")
        )
    return _GENDER_CLASSIFIER


def detect_speaker_genders(audio_path, speaker_segments):
    """
    For each unique speaker, extract their audio segments,
    run a simple pitch-based gender heuristic (F0 median).
    Pitch > 165 Hz → female, else → male.
    """
    import librosa
    import numpy as np

    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    speaker_genders = {}

    speakers = {}
    for seg in speaker_segments:
        spk = seg["speaker"]
        speakers.setdefault(spk, []).append(seg)

    for spk, segs in speakers.items():
        f0_values = []
        for seg in segs[:5]:
            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            chunk = audio[start_sample:end_sample]

            if len(chunk) < sr * 0.5:
                continue

            try:
                f0, voiced_flag, _ = librosa.pyin(
                    chunk,
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=sr
                )
                if f0 is not None and voiced_flag is not None:
                    voiced_f0 = f0[voiced_flag]
                    if len(voiced_f0) > 0:
                        f0_values.extend(voiced_f0[~np.isnan(voiced_f0)].tolist())
            except Exception as e:
                logging.warning(f"F0 extraction failed for {spk}: {e}")

        if f0_values:
            median_f0 = np.median(f0_values)
            speaker_genders[spk] = "female" if median_f0 > 165.0 else "male"
            logging.info(f"Speaker {spk}: median F0={median_f0:.1f}Hz → {speaker_genders[spk]}")
        else:
            speaker_genders[spk] = "male"
            logging.warning(f"Speaker {spk}: no F0 detected, defaulting to male")

    return speaker_genders


def get_voice_for_speaker(speaker_label, lang_code, speaker_genders=None):
    voices = VOICE_MAP.get(lang_code, VOICE_MAP["en"])
    if speaker_genders and speaker_label in speaker_genders:
        gender = speaker_genders[speaker_label]
    else:
        gender = "male"
    return voices.get(gender, voices["male"])

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

WHISPER_MODEL = None
DIARIZATION_PIPELINE = None


def resolve_ffmpeg():
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    bundled = os.path.join(BACKEND_DIR, "ffmpeg.exe")
    if os.path.exists(bundled):
        return bundled
    raise RuntimeError(
        "ffmpeg not found. Install ffmpeg and add it to PATH, or place ffmpeg.exe in the backend folder."
    )


def resolve_ffprobe():
    """Resolve ffprobe executable path"""
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        return ffprobe_path
    bundled = os.path.join(BACKEND_DIR, "ffprobe.exe")
    if os.path.exists(bundled):
        return bundled
    raise RuntimeError(
        "ffprobe not found. Install ffmpeg and add it to PATH, or place ffprobe.exe in the backend folder."
    )


def extract_audio(video_path, audio_path):
    ffmpeg = resolve_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr.strip()}")
    logging.info(f"Audio extracted to {audio_path}")
    return audio_path


def separate_audio_with_demucs(audio_path, output_dir):
    cmd = [
        sys.executable, "-m", "demucs.separate",
        "-o", output_dir,
        "-n", "htdemucs",
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs separation failed: {result.stderr.strip()}")

    base = os.path.splitext(os.path.basename(audio_path))[0]
    sep_dir = os.path.join(output_dir, "htdemucs", base)
    vocals = os.path.join(sep_dir, "vocals.wav")
    no_vocals = os.path.join(sep_dir, "no_vocals.wav")
    if not os.path.exists(vocals) or not os.path.exists(no_vocals):
        raise RuntimeError("Demucs did not produce expected vocals/no_vocals outputs")
    logging.info(f"Audio separated with Demucs: vocals={vocals}, no_vocals={no_vocals}")
    return vocals, no_vocals


def transcribe_audio(audio_path, input_lang):
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        logging.info(f"Loading Whisper model on {device}")
        WHISPER_MODEL = whisper.load_model("small", device=device)

    result = WHISPER_MODEL.transcribe(audio_path, language=input_lang)
    segments = result.get("segments", [])
    logging.info(f"Transcription complete: {len(segments)} segments")
    return segments


def detect_speakers(audio_path):
    global DIARIZATION_PIPELINE

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is missing. Set it before running the backend.")

    if DIARIZATION_PIPELINE is None:
        DIARIZATION_PIPELINE = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )

    import soundfile as sf

    waveform, sample_rate = sf.read(audio_path, dtype="float32")

    if waveform.ndim == 1:
        waveform = torch.from_numpy(waveform).unsqueeze(0)
    elif waveform.ndim == 2:
        waveform = torch.from_numpy(waveform).transpose(0, 1)
    else:
        raise ValueError(f"Unexpected audio shape from soundfile: {waveform.shape}")

    diarization = DIARIZATION_PIPELINE({
        "waveform": waveform,
        "sample_rate": sample_rate
    })

    speaker_segments = []

    if hasattr(diarization, "itertracks"):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker)
            })
    elif hasattr(diarization, "speaker_diarization") and hasattr(diarization.speaker_diarization, "itertracks"):
        ann = diarization.speaker_diarization
        for turn, _, speaker in ann.itertracks(yield_label=True):
            speaker_segments.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker)
            })
    elif hasattr(diarization, "segments"):
        for seg in diarization.segments:
            speaker_segments.append({
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "speaker": str(seg["speaker"])
            })
    else:
        raise RuntimeError(f"Unsupported diarization output type: {type(diarization)}")

    return speaker_segments


def assign_speakers_to_segments(segments, speaker_segments):
    for seg in segments:
        best_speaker = "unknown"
        best_overlap = 0.0

        for sp in speaker_segments:
            overlap_start = max(seg["start"], sp["start"])
            overlap_end = min(seg["end"], sp["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = sp["speaker"]

        seg["speaker"] = best_speaker

    return segments


def translate_segments(segments, source_lang, target_lang):
    if source_lang == target_lang:
        logging.info("Source and target language are the same. Skipping translation.")
        return segments
    logging.info(f"Translating from {source_lang} to {target_lang}")
    translated = []
    for seg in segments:
        time.sleep(0.1)
        seg = dict(seg)
        text = seg.get("text", "").strip()
        if not text:
            translated.append(seg)
            continue
        try:
            translated_text = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
            seg["text"] = translated_text if translated_text else text
        except Exception as e:
            logging.warning(f"GoogleTranslator failed: {e}. Trying MyMemory.")
            try:
                translated_text = MyMemoryTranslator(source=source_lang, target=target_lang).translate(text)
                seg["text"] = translated_text if translated_text else text
            except Exception as e2:
                logging.error(f"MyMemoryTranslator also failed: {e2}. Keeping original text.")
                seg["text"] = text
        translated.append(seg)
    logging.info("Translation complete.")
    return translated


def merge_small_segments(segments, gap_threshold=0.3, min_words=4):
    """
    Merge consecutive segments ONLY if BOTH conditions are true:
    - gap < gap_threshold AND text is too short
    Prevents aggressive merging.
    """
    if len(segments) <= 1:
        return segments
    
    merged = []
    current = dict(segments[0])
    
    for next_seg in segments[1:]:
        gap = next_seg["start"] - current["end"]
        current_words = len(current.get("text", "").strip().split())
        
        # FIXED: Use AND instead of OR (both conditions must be true)
        if gap < gap_threshold and current_words < min_words:
            current["end"] = next_seg["end"]
            current["text"] = (current["text"].strip() + " " + next_seg["text"].strip()).strip()
            logging.debug(f"Merged: '{current['text'][:40]}...'")
        else:
            merged.append(current)
            current = dict(next_seg)
    
    merged.append(current)
    logging.info(f"Segments: {len(segments)} → {len(merged)} (merged small ones only)")
    return merged


def prevent_overlap(segments, min_duration=0.4, gap_buffer=0.05):
    """
    Remove overlaps & enforce min duration.
    FIXED: Added safety check for zero/negative duration.
    """
    if len(segments) <= 1:
        return segments
    
    adjusted = []
    for i, seg in enumerate(segments):
        seg = dict(seg)
        start = seg["start"]
        end = seg["end"]
        
        # Check against next segment
        if i < len(segments) - 1:
            next_start = segments[i + 1]["start"]
            max_end = next_start - gap_buffer
            if end > max_end:
                end = max_end
        
        # Enforce minimum duration
        duration = end - start
        if duration < min_duration:
            if i < len(segments) - 1:
                max_possible = segments[i + 1]["start"] - gap_buffer
                end = min(start + min_duration, max_possible)
            else:
                end = start + min_duration
        
        # FIXED: Safety check for invalid duration
        if end <= start:
            logging.warning(f"Segment {i+1} has zero/negative duration, skipping")
            continue
        
        seg["end"] = end
        seg["original_duration"] = end - start
        adjusted.append(seg)
    
    logging.info(f"Overlap fixed: {len(segments)} → {len(adjusted)}")
    return adjusted


def format_srt_time(ms):
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    ms_rem = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms_rem:03d}"


def generate_srt(segments, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            text = seg.get("text", "").strip()
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(start_ms)} --> {format_srt_time(end_ms)}\n")
            f.write(f"{text}\n\n")
    logging.info(f"SRT file written to {output_path}")


def get_audio_duration(audio_path, ffmpeg_cmd):
    """Get audio duration in seconds using ffprobe."""
    try:
        ffprobe_cmd = resolve_ffprobe()
        
        cmd = [
            ffprobe_cmd,
            "-i", audio_path,
            "-show_entries", "format=duration",
            "-v", "quiet",
            "-of", "csv=p=0"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        logging.debug(f"Duration: {os.path.basename(audio_path)} = {duration:.2f}s")
        return duration
    except Exception as e:
        logging.warning(f"Failed to get duration: {e}")
        return None


def _build_atempo_filter(speed_factor):
    """Build atempo filter chain for speeds > 2.0 or < 0.5"""
    filters = []
    
    while speed_factor > 2.0:
        filters.append("atempo=2.0")
        speed_factor /= 2.0
    
    while speed_factor < 0.5:
        filters.append("atempo=0.5")
        speed_factor /= 0.5
    
    filters.append(f"atempo={speed_factor:.4f}")
    return ",".join(filters)


def adjust_audio_speed(input_audio, output_audio, target_duration, ffmpeg_cmd):
    """
    Speed up ONLY if TTS is longer than target.
    Never stretch/slow down (keeps natural feel).
    """
    current_duration = get_audio_duration(input_audio, ffmpeg_cmd)
    
    if current_duration is None or target_duration <= 0:
        return input_audio
    
    if target_duration < 0.3:
        return input_audio
    
    # Only speed up if too long
    if current_duration <= target_duration:
        logging.debug(f"Duration OK: {current_duration:.2f}s ≤ {target_duration:.2f}s")
        return input_audio
    
    speed_factor = current_duration / target_duration
    speed_factor = min(2.0, speed_factor)
    
    if speed_factor <= 1.1:
        return input_audio
    
    try:
        atempo_filter = _build_atempo_filter(speed_factor)
        codec = "pcm_s16le" if output_audio.lower().endswith(".wav") else "aac"
        cmd = [
            ffmpeg_cmd, "-y",
            "-i", input_audio,
            "-af", f"{atempo_filter},loudnorm=I=-16:TP=-1.5:LRA=11",
            "-c:a", codec, "-b:a", "192k",
            output_audio
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_audio):
            new_dur = get_audio_duration(output_audio, ffmpeg_cmd)
            logging.info(f"[SPEED] {current_duration:.2f}s → {new_dur:.2f}s (target: {target_duration:.2f}s)")
            return output_audio
        else:
            logging.warning(f"Speed adjust failed: {result.stderr.strip()}")
            return input_audio
    except Exception as e:
        logging.warning(f"Speed adjust error: {e}")
        return input_audio


def trim_audio_to_duration(input_audio, output_audio, max_duration, ffmpeg_cmd):
    """Hard trim audio if still exceeds target duration."""
    current_duration = get_audio_duration(input_audio, ffmpeg_cmd)
    if current_duration is None or current_duration <= max_duration:
        return input_audio
    
    try:
        codec = "pcm_s16le" if output_audio.lower().endswith(".wav") else "aac"
        cmd = [
            ffmpeg_cmd, "-y",
            "-i", input_audio,
            "-t", str(max_duration),
            "-c:a", codec, "-b:a", "192k",
            output_audio
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_audio):
            logging.info(f"[TRIM] {current_duration:.2f}s → {max_duration:.2f}s")
            return output_audio
        else:
            return input_audio
    except Exception as e:
        logging.warning(f"Trim error: {e}")
        return input_audio


def fade_audio(input_audio, output_audio, duration, ffmpeg_cmd):
    fade_length = min(0.03, max(0.01, duration * 0.1))
    fade_out_start = max(0.0, duration - fade_length)
    try:
        cmd = [
            ffmpeg_cmd, "-y",
            "-i", input_audio,
            "-af", f"afade=t=in:st=0:d={fade_length:.3f},afade=t=out:st={fade_out_start:.3f}:d={fade_length:.3f}",
            "-c:a", "aac", "-b:a", "192k",
            output_audio
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output_audio):
            return output_audio
    except Exception as e:
        logging.warning(f"Fade audio error: {e}")
    return input_audio


def generate_natural_tts(text, output_file, voice, duration):
    """Generate TTS with natural variation (rate + pitch)"""
    rate = random.choice(["-5%", "+0%", "+5%"])
    pitch = random.choice(["-2Hz", "+0Hz", "+2Hz"])

    try:
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        asyncio.run(communicate.save(output_file))
        if os.path.exists(output_file):
            logging.info(f"TTS: '{text[:30]}...' → {voice} (rate={rate}, pitch={pitch})")
            return output_file
        else:
            logging.error(f"TTS failed for: '{text[:30]}...'")
            return None
    except Exception as e:
        logging.error(f"TTS error: {e}")
        return None


def get_xtts_model():
    global _XTTS_MODEL
    if _XTTS_MODEL is None:
        from TTS.api import TTS as CoquiTTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _XTTS_MODEL = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
        _XTTS_MODEL.to(device)
        logging.info(f"XTTS2 loaded on {device}")
    return _XTTS_MODEL


def extract_speaker_sample(audio_path, speaker_segments, speaker_label, output_dir,
                            min_duration=6.0, max_duration=12.0):
    import librosa
    import numpy as np
    import soundfile as sf

    segs = [s for s in speaker_segments if s["speaker"] == speaker_label]
    if not segs:
        return None

    segs_sorted = sorted(segs, key=lambda x: x["end"] - x["start"], reverse=True)
    audio, sr = librosa.load(audio_path, sr=24000, mono=True)

    collected = []
    total_dur = 0.0
    for seg in segs_sorted:
        if total_dur >= max_duration:
            break
        dur = seg["end"] - seg["start"]
        if dur < 1.0:
            continue
        start_s = int(seg["start"] * sr)
        end_s = int(seg["end"] * sr)
        chunk = audio[start_s:end_s]

        rms = np.sqrt(np.mean(chunk**2))
        if rms < 0.005:
            continue

        collected.append(chunk)
        total_dur += dur
        if total_dur >= min_duration:
            break

    if not collected:
        return None

    sample_audio = np.concatenate(collected)
    sample_path = os.path.join(output_dir, f"speaker_sample_{speaker_label}.wav")
    sf.write(sample_path, sample_audio, sr)
    logging.info(f"Speaker sample for {speaker_label}: {total_dur:.1f}s → {sample_path}")
    return sample_path


XTTS_LANG_MAP = {
    "en": "en", "hi": "hi", "es": "es", "fr": "fr",
    "de": "de", "it": "it", "pt": "pt", "pl": "pl",
    "tr": "tr", "ru": "ru", "nl": "nl", "cs": "cs",
    "ar": "ar", "zh-cn": "zh-cn", "ja": "ja", "ko": "ko",
    "te": "en",
}

EDGE_TTS_FALLBACK_LANGS = {"te", "ta", "ml", "bn"}


def _edge_tts_generate(text, output_path, voice):
    try:
        communicate = edge_tts.Communicate(text, voice, rate="+0%", pitch="+0Hz")
        asyncio.run(communicate.save(output_path))
        if os.path.exists(output_path):
            return output_path
    except Exception as e:
        logging.error(f"Edge-TTS failed: {e}")
    return None


def generate_tts_for_segment(text, output_path, speaker_sample_path,
                               lang_code, voice_fallback, target_duration):
    xtts_lang = XTTS_LANG_MAP.get(lang_code)
    use_xtts = (
        xtts_lang is not None
        and lang_code not in EDGE_TTS_FALLBACK_LANGS
        and speaker_sample_path is not None
        and os.path.exists(speaker_sample_path)
    )

    if use_xtts:
        try:
            model = get_xtts_model()
            wav_path = output_path.replace(".mp3", ".wav")
            model.tts_to_file(
                text=text,
                speaker_wav=speaker_sample_path,
                language=xtts_lang,
                file_path=wav_path,
                split_sentences=False
            )
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
                logging.info(f"XTTS2 generated: {os.path.basename(wav_path)}")
                return wav_path
        except Exception as e:
            logging.warning(f"XTTS2 failed: {e}. Falling back to Edge-TTS.")

    return _edge_tts_generate(text, output_path, voice_fallback)


def compute_timing_strategy(segments, tolerance=0.25):
    adjusted = []
    for i, seg in enumerate(segments):
        seg = dict(seg)
        if i < len(segments) - 1:
            next_speech_start = segments[i + 1]["start"]
        else:
            next_speech_start = seg["end"] + 10.0

        available_duration = next_speech_start - seg["start"] - 0.1
        seg["available_duration"] = available_duration
        seg["original_slot"] = seg["end"] - seg["start"]
        adjusted.append(seg)
    return adjusted


def _speed_adjust_gentle(input_audio, output_audio, target_duration, ffmpeg_cmd):
    current = get_audio_duration(input_audio, ffmpeg_cmd)
    if not current or current <= target_duration:
        return input_audio

    speed = min(1.35, current / target_duration)
    try:
        codec = "pcm_s16le" if output_audio.lower().endswith(".wav") else "aac"
        cmd = [
            ffmpeg_cmd, "-y", "-i", input_audio,
            "-af", f"atempo={speed:.4f},loudnorm=I=-16:TP=-1.5:LRA=11",
            "-c:a", codec, "-b:a", "192k",
            output_audio
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output_audio):
            return output_audio
    except Exception as e:
        logging.warning(f"Gentle speed adjust failed: {e}")
    return input_audio


def generate_tts_segments_v2(segments, speaker_segments, audio_path,
                               output_dir, language_code, ffmpeg_cmd):
    os.makedirs(output_dir, exist_ok=True)
    speaker_genders = detect_speaker_genders(audio_path, speaker_segments)

    speakers = set(s.get("speaker", "SPEAKER_00") for s in segments)
    speaker_samples = {}
    for spk in speakers:
        speaker_samples[spk] = extract_speaker_sample(
            audio_path, speaker_segments, spk, output_dir
        )

    segments = merge_small_segments(segments, gap_threshold=0.5, min_words=5)
    segments = compute_timing_strategy(segments)

    audio_files = []
    for i, seg in enumerate(segments, 1):
        text = seg.get("text", "").strip()
        if not text or len(text.split()) < 2:
            continue

        speaker_label = seg.get("speaker", "SPEAKER_00")
        voice_fallback = get_voice_for_speaker(speaker_label, language_code, speaker_genders)
        speaker_sample = speaker_samples.get(speaker_label)

        raw_file = os.path.join(output_dir, f"seg_{i}_raw.wav")
        adj_file = os.path.join(output_dir, f"seg_{i}_adj.wav")

        tts_result = generate_tts_for_segment(
            text, raw_file, speaker_sample,
            language_code, voice_fallback,
            seg["original_slot"]
        )

        if not tts_result or not os.path.exists(tts_result):
            logging.warning(f"Segment {i}: TTS failed, skipping")
            continue

        tts_duration = get_audio_duration(tts_result, ffmpeg_cmd)
        if not tts_duration:
            continue

        available = seg["available_duration"]
        if tts_duration > available:
            speed_ratio = tts_duration / available
            if speed_ratio <= 1.35:
                adjusted = _speed_adjust_gentle(tts_result, adj_file, available, ffmpeg_cmd)
            else:
                adjusted = trim_audio_to_duration(tts_result, adj_file, available, ffmpeg_cmd)
            final_file = adjusted if os.path.exists(adjusted) else tts_result
        else:
            final_file = tts_result

        final_duration = get_audio_duration(final_file, ffmpeg_cmd) or tts_duration
        actual_end = seg["start"] + min(final_duration, available)

        audio_files.append({
            "file": final_file,
            "start": seg["start"],
            "end": actual_end,
            "target_duration": actual_end - seg["start"],
        })

    return audio_files


def generate_tts_segments(segments, output_dir, language_code, ffmpeg_cmd):
    """
    Generate TTS with:
    1. Smart merging (only tiny + short text)
    2. Overlap prevention
    3. Duration validation before TTS
    4. Speed adjust + trim fallback
    5. Filter out failed segments
    """
    # STEP 1: Merge tiny segments only
    segments = merge_small_segments(segments, gap_threshold=0.3, min_words=4)
    
    # STEP 2: Fix overlaps
    segments = prevent_overlap(segments, min_duration=0.4, gap_buffer=0.05)
    
    logging.info(f"Processing {len(segments)} segments for TTS")
    
    os.makedirs(output_dir, exist_ok=True)
    audio_files = []

    for i, seg in enumerate(segments, 1):
        text = seg.get("text", "").strip()
        
        # FIXED: Skip if text is empty or too short
        if not text or len(text.strip()) < 2:
            logging.debug(f"Segment {i}: skipped (empty/short text)")
            continue
        
        target_duration = seg["end"] - seg["start"]
        
        # FIXED: Skip if duration is too short
        if target_duration <= 0.3 or len(text.split()) < 2:
            logging.debug(f"Segment {i}: skipped (duration {target_duration:.2f}s or tiny text)")
            continue
        
        raw_file = os.path.join(output_dir, f"segment_{i}.mp3")
        adj_file = os.path.join(output_dir, f"segment_{i}_adj.mp3")
        trim_file = os.path.join(output_dir, f"segment_{i}_trim.mp3")
        fade_file = os.path.join(output_dir, f"segment_{i}_fade.mp3")

        speaker_label = seg.get("speaker", "SPEAKER_00")
        voice = get_voice_for_speaker(speaker_label, language_code)

        # Generate TTS
        if not generate_natural_tts(text, raw_file, voice, target_duration):
            logging.warning(f"Segment {i}: TTS generation failed")
            continue

        # Speed adjust
        adjusted_file = adjust_audio_speed(raw_file, adj_file, target_duration, ffmpeg_cmd)

        # Trim if needed
        if adjusted_file == adj_file:
            final_file = trim_audio_to_duration(adj_file, trim_file, target_duration, ffmpeg_cmd)
        else:
            final_file = adjusted_file

        # Fade to soften cuts
        if os.path.exists(final_file):
            final_file = fade_audio(final_file, fade_file, get_audio_duration(final_file, ffmpeg_cmd) or target_duration, ffmpeg_cmd)

        # Validate final file exists
        if not os.path.exists(final_file):
            logging.warning(f"Segment {i}: final file missing")
            continue

        audio_files.append({
            "file": final_file,
            "start": seg["start"],
            "end": seg["end"],
            "target_duration": target_duration,
            "raw_file": raw_file,
            "adj_file": adj_file,
            "trim_file": trim_file
        })

    audio_files = sorted(audio_files, key=lambda x: x["start"])
    logging.info(f"✓ Generated {len(audio_files)} TTS segments (merged+cleaned)")
    return audio_files


def combine_tts_audio(audio_files, final_audio_path, ffmpeg_cmd):
    os.makedirs(os.path.dirname(final_audio_path) or ".", exist_ok=True)
    if not audio_files:
        raise RuntimeError("No TTS audio files to combine")

    audio_files = sorted(audio_files, key=lambda x: x["start"])

    valid_files = []
    for item in audio_files:
        fpath = item.get("file")
        if not fpath or not os.path.exists(fpath):
            continue
        dur = get_audio_duration(fpath, ffmpeg_cmd)
        if dur and dur > 0.1 and item["end"] > item["start"]:
            item = dict(item)
            item["_real_duration"] = dur
            valid_files.append(item)

    if not valid_files:
        raise RuntimeError("No valid audio files to combine")

    input_args = []
    filter_parts = []
    current_end = 0.0

    for i, item in enumerate(valid_files):
        path = item["file"]
        slot_start = item["start"]
        slot_end = item["end"]
        real_dur = item["_real_duration"]

        safe_start = max(slot_start, current_end)
        max_allowed = slot_end - 0.02

        if safe_start >= max_allowed:
            continue

        if real_dur > (slot_end - safe_start) or safe_start + real_dur > max_allowed:
            trimmed_path = os.path.splitext(path)[0] + "_slottrim.aac"
            trimmed = trim_audio_to_duration(path, trimmed_path, max_allowed - safe_start, ffmpeg_cmd)
            if not os.path.exists(trimmed):
                continue
            path = trimmed
            real_dur = get_audio_duration(path, ffmpeg_cmd) or (max_allowed - safe_start)

        delay_ms = max(0, int(safe_start * 1000))
        input_args.extend(["-i", path])
        filter_parts.append(f"[{len(filter_parts)}:a]adelay={delay_ms}|{delay_ms},asetpts=PTS-STARTPTS[a{len(filter_parts)}]")

        current_end = safe_start + real_dur + 0.02

    if not filter_parts:
        raise RuntimeError("No valid non-overlapping audio segments to combine")

    mix_inputs = "".join(f"[a{i}]" for i in range(len(filter_parts)))
    filter_complex = ";".join(filter_parts) + f";{mix_inputs}amix=inputs={len(filter_parts)}:duration=longest:normalize=0[out]"

    cmd = [
        ffmpeg_cmd, "-y",
        *input_args,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:a", "aac", "-b:a", "192k",
        final_audio_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mix failed: {result.stderr.strip()}")

    return final_audio_path


def combine_video_with_audio(video_path, bg_audio_path, dub_audio_path, output_path, ffmpeg_cmd):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except:
            pass

    cmd = [
        ffmpeg_cmd, "-y",
        "-i", video_path,
        "-i", bg_audio_path,
        "-i", dub_audio_path,
        "-filter_complex",
        "[1:a]volume=0.9[bg];"
        "[2:a]volume=1.6[tts];"
        "[bg][tts]amix=inputs=2:duration=first:normalize=0[aout]",
        "-map", "0:v:0",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Video+background+dub mix failed: {result.stderr.strip()}")
        
    logging.info(f"✓ Video+audio merged: {output_path}")


def combine_video_with_audio_and_subtitles(video_path, bg_audio_path, dub_audio_path, srt_path, output_path, ffmpeg_cmd):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except:
            pass

    srt_abs = os.path.abspath(srt_path)
    srt_filter_path = srt_abs.replace("\\", "/")
    if len(srt_filter_path) > 1 and srt_filter_path[1] == ":":
        srt_filter_path = srt_filter_path[0] + "\\:" + srt_filter_path[2:]
    srt_filter_path = srt_filter_path.replace("'", r"\\'")

    cmd = [
        ffmpeg_cmd, "-y",
        "-i", video_path,
        "-i", bg_audio_path,
        "-i", dub_audio_path,
        "-filter_complex",
        "[1:a]volume=0.9[bg];"
        "[2:a]volume=1.6[tts];"
        "[bg][tts]amix=inputs=2:duration=first:normalize=0[aout]",
        "-map", "0:v:0",
        "-map", "[aout]",
        "-vf", f"subtitles='{srt_filter_path}'",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Video+audio+subtitles mix failed: {result.stderr.strip()}")
    
    logging.info(f"✓ Final dubbed+subtitled video: {output_path}")


def combine_video_with_audio_ducked(video_path, bg_audio_path, dub_audio_path,
                                     srt_path, output_path, ffmpeg_cmd):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except:
            pass

    srt_filter = None
    if srt_path:
        srt_abs = os.path.abspath(srt_path)
        srt_filter = srt_abs.replace("\\", "/")
        if len(srt_filter) > 1 and srt_filter[1] == ":":
            srt_filter = srt_filter[0] + "\\:" + srt_filter[2:]
        srt_filter = srt_filter.replace("'", r"\\'")

    filter_complex = (
        "[2:a]loudnorm=I=-14:TP=-1.5:LRA=11[tts_norm];"
        "[tts_norm]asplit=2[tts_out][tts_sc];"
        "[1:a][tts_sc]sidechaincompress=threshold=0.02:ratio=8:attack=50:release=300:makeup=1[bg_ducked];"
        "[bg_ducked]volume=0.25[bg_final];"
        "[bg_final][tts_out]amix=inputs=2:duration=first:normalize=0[aout]"
    )

    cmd = [
        ffmpeg_cmd, "-y",
        "-i", video_path,
        "-i", bg_audio_path,
        "-i", dub_audio_path,
        "-filter_complex", filter_complex,
        "-map", "0:v:0",
        "-map", "[aout]",
    ]

    if srt_filter:
        cmd += ["-vf", f"subtitles='{srt_filter}'", "-c:v", "libx264", "-preset", "fast", "-crf", "23"]
    else:
        cmd += ["-c:v", "copy"]

    cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest", output_path]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.warning(f"Ducking mix failed, falling back: {result.stderr[-300:]}")
        _simple_mix_fallback(video_path, bg_audio_path, dub_audio_path, srt_path, output_path, ffmpeg_cmd)
    else:
        logging.info(f"✓ Ducked mix complete: {output_path}")


def _simple_mix_fallback(video_path, bg_audio_path, dub_audio_path,
                          srt_path, output_path, ffmpeg_cmd):
    srt_filter = None
    if srt_path:
        srt_abs = os.path.abspath(srt_path)
        srt_filter = srt_abs.replace("\\", "/")
        if len(srt_filter) > 1 and srt_filter[1] == ":":
            srt_filter = srt_filter[0] + "\\:" + srt_filter[2:]
        srt_filter = srt_filter.replace("'", r"\\'")

    filter_complex = (
        "[2:a]loudnorm=I=-14:TP=-1.5:LRA=11[tts_norm];"
        "[1:a]volume=0.18[bg];"
        "[bg][tts_norm]amix=inputs=2:duration=first:normalize=0[aout]"
    )
    cmd = [
        ffmpeg_cmd, "-y", "-i", video_path, "-i", bg_audio_path, "-i", dub_audio_path,
        "-filter_complex", filter_complex,
        "-map", "0:v:0", "-map", "[aout]",
    ]
    if srt_filter:
        cmd += ["-vf", f"subtitles='{srt_filter}'", "-c:v", "libx264", "-preset", "fast", "-crf", "23"]
    else:
        cmd += ["-c:v", "copy"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest", output_path]
    subprocess.run(cmd, capture_output=True, text=True)


def cleanup_temp_files(paths):
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
                logging.info(f"Cleaned: {path}")
            except Exception as e:
                logging.warning(f"Cleanup failed: {path} ({e})")


def process_video(video_path, input_lang_code, audio_lang_code, subtitle_lang_code, 
                  output_path, enable_dubbing, progress_callback):
    temp_audio = None
    temp_no_vocals = None
    temp_srt = None
    temp_dubbed_audio = None
    tts_segment_files = set()
    output_dir = os.path.dirname(os.path.abspath(output_path))

    try:
        video_path = os.path.abspath(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        ext = os.path.splitext(video_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported format: {ext}")
        
        os.makedirs(output_dir, exist_ok=True)
        ffmpeg_cmd = resolve_ffmpeg()

        progress_callback(0, "Starting...")

        # Extract audio
        temp_audio = os.path.join(output_dir, "temp_audio.wav")
        extract_audio(video_path, temp_audio)
        progress_callback(20, "Audio extracted")

        temp_no_vocals = temp_audio
        if enable_dubbing:
            try:
                _, temp_no_vocals = separate_audio_with_demucs(temp_audio, output_dir)
                progress_callback(30, "Audio separated")
            except Exception as e:
                logging.warning(f"Demucs separation failed: {e}. Falling back to original audio")
                temp_no_vocals = temp_audio

        # Transcribe
        segments = transcribe_audio(temp_audio, input_lang_code)
        if not segments:
            raise RuntimeError("Transcription failed")
        
        # Detect speakers
        speaker_segments = detect_speakers(temp_audio)
        segments = assign_speakers_to_segments(segments, speaker_segments)
        progress_callback(50, "Transcribed + speaker detected")

        # Translate
        subtitle_segments = translate_segments(segments, input_lang_code, subtitle_lang_code)
        audio_segments = (
            translate_segments(segments, input_lang_code, audio_lang_code)
            if enable_dubbing and audio_lang_code != subtitle_lang_code
            else subtitle_segments
        )

        progress_callback(65, "Translated")

        # Generate subtitles
        temp_srt = os.path.join(output_dir, "temp_subtitles.srt")
        generate_srt(subtitle_segments, temp_srt)
        progress_callback(70, "Subtitles created")

        if enable_dubbing:
            logging.info("=== DUBBING PIPELINE ===")
            
            tts_dir = os.path.join(output_dir, "tts_segments")
            os.makedirs(tts_dir, exist_ok=True)
            
            tts_audio_files = generate_tts_segments_v2(
                audio_segments,
                speaker_segments,
                temp_audio,
                tts_dir,
                audio_lang_code,
                ffmpeg_cmd
            )
            
            if not tts_audio_files:
                raise RuntimeError("TTS pipeline produced NO audio files")
            
            logging.info(f"✓ Generated {len(tts_audio_files)} TTS segments")
            progress_callback(80, f"TTS ready ({len(tts_audio_files)} segments)")

            for item in tts_audio_files:
                p = item.get("file")
                if p and os.path.exists(p):
                    tts_segment_files.add(p)

            temp_dubbed_audio = os.path.join(output_dir, "dubbed_audio.aac")
            combine_tts_audio(tts_audio_files, temp_dubbed_audio, ffmpeg_cmd)
            progress_callback(90, "Audio combined")

            combine_video_with_audio_ducked(
                video_path,
                temp_no_vocals,
                temp_dubbed_audio,
                temp_srt if subtitle_lang_code else None,
                output_path,
                ffmpeg_cmd
            )
            progress_callback(100, "Complete")
        else:
            logging.info("Subtitle-only mode (no dubbing)")
            # Simple subtitle burn
            cmd = [
                ffmpeg_cmd, "-y", "-i", video_path,
                "-vf", f"subtitles='{temp_srt}'",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "192k",
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError("Subtitle burn failed")
            progress_callback(100, "Complete")

        logging.info("✓ Processing complete!")
        
        result_path = os.path.join(output_dir, "result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"status": "complete", "output": output_path}, f, ensure_ascii=False, indent=2)

    except Exception as e:
        logging.error(f"ERROR: {e}")
        progress_callback(-1, f"Failed: {str(e)}")
        
        result_path = os.path.join(output_dir, "result.json")
        try:
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump({"status": "error", "message": str(e)}, f, ensure_ascii=False, indent=2)
        except:
            pass
        raise
    
    finally:
        cleanup_temp_files([temp_audio, temp_no_vocals, temp_srt, temp_dubbed_audio] + list(tts_segment_files))
        tts_dir = os.path.join(output_dir, "tts_segments")
        if os.path.exists(tts_dir):
            try:
                if not os.listdir(tts_dir):
                    os.rmdir(tts_dir)
            except:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vividha Hub - Video Dubbing Pipeline")
    parser.add_argument("--video", type=str, help="Input video path")
    parser.add_argument("--input_lang", type=str, default="english", help="Input language")
    parser.add_argument("--audio_lang", type=str, default="english", help="Dubbing language")
    parser.add_argument("--subtitle_lang", type=str, default="english", help="Subtitle language")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output path")
    parser.add_argument("--enable_dubbing", action="store_true", default=False, help="Enable dubbing")
    args = parser.parse_args()

    if args.video:
        video_path = args.video
        input_lang = args.input_lang
        audio_lang = args.audio_lang
        subtitle_lang = args.subtitle_lang
        output_path = args.output
        enable_dubbing = args.enable_dubbing
    else:
        input_json = os.path.join(BACKEND_DIR, "input.json")
        if not os.path.exists(input_json):
            raise FileNotFoundError("No --video arg and no input.json")
        
        with open(input_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        video_path = data.get("file")
        input_lang = data.get("input_lang", "english")
        audio_lang = data.get("audio_lang", "english")
        subtitle_lang = data.get("subtitle_lang", "english")
        output_path = data.get("output", "output.mp4")
        enable_dubbing = data.get("enable_dubbing", False)

    if not video_path:
        raise ValueError("Video path missing")

    input_lang_code = LANGUAGE_MAP.get(str(input_lang).lower())
    if not input_lang_code:
        raise ValueError(f"Unsupported input language: {input_lang}")

    subtitle_lang_code = LANGUAGE_MAP.get(str(subtitle_lang).lower())
    if not subtitle_lang_code:
        raise ValueError(f"Unsupported subtitle language: {subtitle_lang}")

    audio_lang_code = LANGUAGE_MAP.get(str(audio_lang).lower())
    if not audio_lang_code:
        audio_lang_code = subtitle_lang_code
        logging.warning(f"Audio language unsupported, using: {subtitle_lang}")

    progress_json = os.path.join(os.path.dirname(os.path.abspath(output_path)), "progress.json")

    def progress_callback(percent, message):
        logging.info(f"Progress {percent}%: {message}")
        try:
            os.makedirs(os.path.dirname(progress_json) or ".", exist_ok=True)
            with open(progress_json, "w", encoding="utf-8") as f:
                json.dump({"percent": percent, "message": message}, f, ensure_ascii=False, indent=2)
        except:
            pass

    process_video(
        video_path,
        input_lang_code,
        audio_lang_code,
        subtitle_lang_code,
        os.path.abspath(output_path),
        enable_dubbing,
        progress_callback
    )