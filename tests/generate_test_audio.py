import wave
import struct
import math
import os

def create_synthetic_wav(filename: str = "test_audio.wav", duration_sec: float = 2.0, sample_rate: int = 16000):
    """Generates a clean synthetic 16kHz mono WAV file for testing audio endpoints."""
    num_samples = int(duration_sec * sample_rate)
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)      # Mono
        wav_file.setsampwidth(2)      # 16-bit
        wav_file.setframerate(sample_rate)
        
        # Generate 440Hz tone modulated with slight envelope
        for i in range(num_samples):
            t = i / sample_rate
            # 440 Hz tone with fade in / out
            env = min(1.0, t * 5, (duration_sec - t) * 5)
            sample_val = int(env * 0.5 * 32767.0 * math.sin(2.0 * math.pi * 440.0 * t))
            data = struct.pack('<h', sample_val)
            wav_file.writeframes(data)
            
    print(f"Created synthetic test audio: {filename} ({duration_sec}s, {sample_rate}Hz)")
    return filename

if __name__ == "__main__":
    create_synthetic_wav()
