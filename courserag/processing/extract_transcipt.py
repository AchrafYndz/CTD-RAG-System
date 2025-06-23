import os
import openai
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment
import math

openai.api_key = os.getenv("OPENAI_API_KEY")

RAW_DATA_DIR = "data/raw"
VIDEOS_DIR = os.path.join(RAW_DATA_DIR, "videos")
AUDIO_DIR = os.path.join(RAW_DATA_DIR, "audio")

CLEAN_DATA_DIR = "data/clean"
TRANSCRIPTS_DIR = os.path.join(CLEAN_DATA_DIR, "transcripts")
TEMP_AUDIO_DIR = os.path.join(CLEAN_DATA_DIR, "temp_audio")

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

MAX_AUDIO_SIZE_MB = 25
MAX_AUDIO_SIZE_BYTES = MAX_AUDIO_SIZE_MB * 1024 * 1024
CHUNK_LENGTH_MS = 10 * 60 * 1000

def extract_audio_from_video(video_path, audio_output_path):
    print(f"Extracting audio from: {video_path}")
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(audio_output_path, codec="mp3")
        audio.close()
        video.close()
        print(f"Audio extracted to: {audio_output_path}")
        return True
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return False

def extract_all_audio_from_videos():
    video_files = [f for f in os.listdir(VIDEOS_DIR) if f.endswith(".mp4")]
    print(f"Found {len(video_files)} video files to extract audio from.")
    
    extracted_count = 0
    for video_file in sorted(video_files):
        video_path = os.path.join(VIDEOS_DIR, video_file)
        base_name = os.path.splitext(video_file)[0]
        audio_output_path = os.path.join(AUDIO_DIR, f"{base_name}.mp3")
        
        if os.path.exists(audio_output_path):
            print(f"Audio for {video_file} already exists. Skipping extraction.")
            continue
        
        print(f"\n--- Extracting audio from {video_file} ---")
        if extract_audio_from_video(video_path, audio_output_path):
            extracted_count += 1
    
    print(f"\nAudio extraction complete. Successfully extracted {extracted_count} audio files.")

def transcribe_audio_chunk(audio_file_path):
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        print(f"Error transcribing {audio_file_path}: {e}")
        return None

def transcribe_audio_file(audio_filename):
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    base_name = os.path.splitext(audio_filename)[0]
    transcript_output_path = os.path.join(TRANSCRIPTS_DIR, f"{base_name}.txt")

    if os.path.exists(transcript_output_path):
        print(f"Transcript for {audio_filename} already exists. Skipping.")
        return

    audio_size_bytes = os.path.getsize(audio_path)
    full_transcript_text = ""

    if audio_size_bytes > MAX_AUDIO_SIZE_BYTES:
        print(f"Audio file {audio_filename} is too large ({audio_size_bytes / (1024*1024):.2f} MB). Splitting...")
        try:
            audio = AudioSegment.from_mp3(audio_path)
            total_length_ms = len(audio)
            num_chunks = math.ceil(total_length_ms / CHUNK_LENGTH_MS)

            for i in range(num_chunks):
                start_ms = i * CHUNK_LENGTH_MS
                end_ms = min((i + 1) * CHUNK_LENGTH_MS, total_length_ms)
                chunk = audio[start_ms:end_ms]

                chunk_filename = os.path.join(TEMP_AUDIO_DIR, f"{base_name}_chunk_{i+1}.mp3")
                chunk.export(chunk_filename, format="mp3")
                print(f"Transcribing chunk {i+1}/{num_chunks} ({chunk_filename})...")
                chunk_transcript = transcribe_audio_chunk(chunk_filename)
                if chunk_transcript:
                    full_transcript_text += chunk_transcript + " "
                os.remove(chunk_filename)
            print(f"Finished transcribing all chunks for {audio_filename}.")

        except Exception as e:
            print(f"Error splitting or transcribing large audio file {audio_filename}: {e}")
            full_transcript_text = None
    else:
        print(f"Transcribing {audio_filename} (size: {audio_size_bytes / (1024*1024):.2f} MB)...")
        full_transcript_text = transcribe_audio_chunk(audio_path)

    if full_transcript_text:
        with open(transcript_output_path, "w", encoding="utf-8") as f:
            f.write(full_transcript_text.strip())
        print(f"Transcript saved to: {transcript_output_path}")
    else:
        print(f"Failed to generate transcript for {audio_filename}.")

def transcribe_all_audio_files():
    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".mp3")]
    print(f"Found {len(audio_files)} audio files to transcribe.")
    
    transcribed_count = 0
    for audio_file in sorted(audio_files):
        print(f"\n--- Processing {audio_file} ---")
        try:
            transcribe_audio_file(audio_file)
            transcribed_count += 1
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
    
    print(f"\nTranscription complete. Successfully processed {transcribed_count} audio files.")

if __name__ == "__main__":
    print("=== PHASE 1: EXTRACTING AUDIO FROM VIDEOS ===")
    extract_all_audio_from_videos()
    
    print("\n=== PHASE 2: TRANSCRIBING AUDIO FILES ===")
    transcribe_all_audio_files()
