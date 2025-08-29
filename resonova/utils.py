"""
Utility functions for ResoNova audio processing.
"""

import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns


def load_audio(file_path: str, target_sr: Optional[int] = None, 
               mono: bool = False) -> Tuple[np.ndarray, int]:
    """
    Load audio file with optional resampling and conversion to mono.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (None to keep original)
        mono: Convert to mono if True
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        # Load audio file
        audio, sample_rate = sf.read(file_path)
        
        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Convert to mono if requested
        if mono and audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if requested
        if target_sr is not None and target_sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr
        
        # Normalize to [-1, 1] range
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
        
        return audio, sample_rate
        
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {file_path}: {e}")


def save_audio(file_path: str, audio: np.ndarray, sample_rate: int, 
               bit_depth: int = 24, format: str = 'wav') -> None:
    """
    Save audio data to file.
    
    Args:
        file_path: Output file path
        audio: Audio data as numpy array
        sample_rate: Audio sample rate
        bit_depth: Output bit depth (16, 24, or 32)
        format: Output format ('wav', 'flac', 'ogg')
    """
    try:
        # Ensure output directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert bit depth
        if bit_depth == 16:
            dtype = np.int16
            audio = np.clip(audio * 32767, -32768, 32767).astype(dtype)
        elif bit_depth == 24:
            dtype = np.int32
            audio = np.clip(audio * 8388607, -8388608, 8388607).astype(dtype)
        elif bit_depth == 32:
            dtype = np.float32
            audio = audio.astype(dtype)
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")
        
        # Save audio
        sf.write(file_path, audio, sample_rate, format=format)
        
    except Exception as e:
        raise RuntimeError(f"Failed to save audio file {file_path}: {e}")


def analyze_loudness(audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """
    Analyze audio loudness using various metrics.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        
    Returns:
        Dictionary containing loudness metrics
    """
    # Calculate RMS
    rms = np.sqrt(np.mean(audio**2))
    rms_db = 20 * np.log10(rms) if rms > 0 else -120
    
    # Calculate peak
    peak = np.max(np.abs(audio))
    peak_db = 20 * np.log10(peak) if peak > 0 else -120
    
    # Calculate crest factor
    crest_factor = peak / rms if rms > 0 else 1
    
    # Calculate LUFS (simplified)
    # In production, use proper LUFS calculation library
    lufs = rms_db + 3.0  # Rough approximation
    
    # Calculate dynamic range
    dynamic_range = peak_db - rms_db
    
    # Calculate spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate).mean()
    
    return {
        'rms': rms,
        'rms_db': rms_db,
        'peak': peak,
        'peak_db': peak_db,
        'crest_factor': crest_factor,
        'integrated_lufs': lufs,
        'dynamic_range': dynamic_range,
        'spectral_centroid': spectral_centroid
    }


def analyze_spectrum(audio: np.ndarray, sample_rate: int, 
                    n_fft: int = 2048) -> Dict[str, np.ndarray]:
    """
    Analyze audio spectrum using FFT.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        n_fft: FFT window size
        
    Returns:
        Dictionary containing spectral analysis results
    """
    # Calculate FFT
    fft = np.fft.rfft(audio, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1/sample_rate)
    
    # Calculate magnitude spectrum
    magnitude = np.abs(fft)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    
    # Calculate power spectrum
    power = magnitude**2
    power_db = 10 * np.log10(power + 1e-10)
    
    # Calculate phase spectrum
    phase = np.angle(fft)
    
    # Calculate mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_mels=128, n_fft=n_fft
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return {
        'frequencies': freqs,
        'magnitude': magnitude,
        'magnitude_db': magnitude_db,
        'power': power,
        'power_db': power_db,
        'phase': phase,
        'mel_spectrogram': mel_spec,
        'mel_spectrogram_db': mel_spec_db
    }


def create_audio_visualization(audio: np.ndarray, sample_rate: int, 
                              output_path: Optional[str] = None,
                              title: str = "Audio Analysis") -> None:
    """
    Create comprehensive audio visualization.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        output_path: Path to save visualization (None to display)
        title: Plot title
    """
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    # 1. Waveform
    time = np.arange(len(audio)) / sample_rate
    axes[0, 0].plot(time, audio)
    axes[0, 0].set_title('Waveform')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True)
    
    # 2. Spectrum
    spectrum_data = analyze_spectrum(audio, sample_rate)
    axes[0, 1].semilogx(spectrum_data['frequencies'], spectrum_data['magnitude_db'])
    axes[0, 1].set_title('Frequency Spectrum')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude (dB)')
    axes[0, 1].grid(True)
    axes[0, 1].set_xlim(20, sample_rate/2)
    
    # 3. Mel-spectrogram
    im = axes[1, 0].imshow(spectrum_data['mel_spectrogram_db'], 
                           aspect='auto', origin='lower',
                           extent=[0, len(audio)/sample_rate, 0, 128])
    axes[1, 0].set_title('Mel-Spectrogram')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Mel Frequency Bins')
    plt.colorbar(im, ax=axes[1, 0], label='Power (dB)')
    
    # 4. Loudness over time
    window_size = int(0.1 * sample_rate)  # 100ms windows
    loudness_time = []
    time_windows = []
    
    for i in range(0, len(audio), window_size):
        window = audio[i:i+window_size]
        if len(window) > 0:
            rms = np.sqrt(np.mean(window**2))
            loudness_time.append(20 * np.log10(rms + 1e-10))
            time_windows.append(i / sample_rate)
    
    axes[1, 1].plot(time_windows, loudness_time)
    axes[1, 1].set_title('Loudness Over Time')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('RMS (dB)')
    axes[1, 1].grid(True)
    
    # 5. Phase spectrum
    axes[2, 0].plot(spectrum_data['frequencies'], spectrum_data['phase'])
    axes[2, 0].set_title('Phase Spectrum')
    axes[2, 0].set_xlabel('Frequency (Hz)')
    axes[2, 0].set_ylabel('Phase (radians)')
    axes[2, 0].grid(True)
    axes[2, 0].set_xlim(20, sample_rate/2)
    
    # 6. Statistics
    loudness_stats = analyze_loudness(audio, sample_rate)
    stats_text = f"""
    RMS: {loudness_stats['rms_db']:.1f} dB
    Peak: {loudness_stats['peak_db']:.1f} dB
    Crest Factor: {loudness_stats['crest_factor']:.2f}
    LUFS: {loudness_stats['integrated_lufs']:.1f}
    Dynamic Range: {loudness_stats['dynamic_range']:.1f} dB
    Duration: {len(audio)/sample_rate:.2f} s
    """
    
    axes[2, 1].text(0.1, 0.5, stats_text, transform=axes[2, 1].transAxes,
                     fontsize=12, verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='lightgray'))
    axes[2, 1].set_title('Audio Statistics')
    axes[2, 1].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_test_tone(frequency: float, duration: float, sample_rate: int = 44100,
                      amplitude: float = 0.5, waveform: str = 'sine') -> np.ndarray:
    """
    Generate test tone for testing and calibration.
    
    Args:
        frequency: Tone frequency in Hz
        duration: Tone duration in seconds
        sample_rate: Audio sample rate
        amplitude: Tone amplitude (0.0 to 1.0)
        waveform: Waveform type ('sine', 'square', 'sawtooth', 'triangle')
        
    Returns:
        Generated test tone
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    if waveform == 'sine':
        tone = np.sin(2 * np.pi * frequency * t)
    elif waveform == 'square':
        tone = np.sign(np.sin(2 * np.pi * frequency * t))
    elif waveform == 'sawtooth':
        tone = 2 * (t * frequency - np.floor(t * frequency + 0.5))
    elif waveform == 'triangle':
        tone = 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1
    else:
        raise ValueError(f"Unsupported waveform: {waveform}")
    
    return amplitude * tone.astype(np.float32)


def apply_fade(audio: np.ndarray, sample_rate: int, fade_in: float = 0.0,
               fade_out: float = 0.0) -> np.ndarray:
    """
    Apply fade in/out to audio.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        fade_in: Fade in duration in seconds
        fade_out: Fade out duration in seconds
        
    Returns:
        Audio with fades applied
    """
    result = audio.copy()
    
    # Fade in
    if fade_in > 0:
        fade_samples = int(fade_in * sample_rate)
        fade_samples = min(fade_samples, len(audio))
        fade_curve = np.linspace(0, 1, fade_samples)
        result[:fade_samples] *= fade_curve
    
    # Fade out
    if fade_out > 0:
        fade_samples = int(fade_out * sample_rate)
        fade_samples = min(fade_samples, len(audio))
        fade_curve = np.linspace(1, 0, fade_samples)
        result[-fade_samples:] *= fade_curve
    
    return result


def normalize_audio(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """
    Normalize audio to target RMS level.
    
    Args:
        audio: Input audio data
        target_rms: Target RMS level
        
    Returns:
        Normalized audio
    """
    current_rms = np.sqrt(np.mean(audio**2))
    if current_rms > 0:
        gain = target_rms / current_rms
        return audio * gain
    return audio


def detect_silence(audio: np.ndarray, sample_rate: int, threshold: float = 0.01,
                   min_duration: float = 0.1) -> list:
    """
    Detect silent regions in audio.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        threshold: Silence threshold
        min_duration: Minimum silence duration in seconds
        
    Returns:
        List of (start_time, end_time) tuples for silent regions
    """
    # Calculate RMS in short windows
    window_size = int(0.01 * sample_rate)  # 10ms windows
    rms_values = []
    
    for i in range(0, len(audio), window_size):
        window = audio[i:i+window_size]
        if len(window) > 0:
            rms = np.sqrt(np.mean(window**2))
            rms_values.append(rms)
    
    # Find silent regions
    silent_regions = []
    in_silence = False
    silence_start = 0
    
    for i, rms in enumerate(rms_values):
        time = i * window_size / sample_rate
        
        if rms < threshold and not in_silence:
            silence_start = time
            in_silence = True
        elif rms >= threshold and in_silence:
            silence_end = time
            if silence_end - silence_start >= min_duration:
                silent_regions.append((silence_start, silence_end))
            in_silence = False
    
    # Handle case where audio ends in silence
    if in_silence:
        silence_end = len(audio) / sample_rate
        if silence_end - silence_start >= min_duration:
            silent_regions.append((silence_start, silence_end))
    
    return silent_regions


def save_analysis_report(audio: np.ndarray, sample_rate: int, 
                        output_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save comprehensive audio analysis report.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        output_path: Path to save report
        metadata: Additional metadata to include
    """
    # Perform analysis
    loudness_stats = analyze_loudness(audio, sample_rate)
    spectrum_data = analyze_spectrum(audio, sample_rate)
    silent_regions = detect_silence(audio, sample_rate)
    
    # Create report
    report = {
        'file_info': {
            'sample_rate': sample_rate,
            'duration': len(audio) / sample_rate,
            'channels': audio.ndim,
            'samples': len(audio)
        },
        'loudness_analysis': loudness_stats,
        'spectral_analysis': {
            'frequency_range': {
                'min': float(spectrum_data['frequencies'].min()),
                'max': float(spectrum_data['frequencies'].max())
            },
            'spectral_centroid': float(loudness_stats['spectral_centroid']),
            'peak_frequency': float(spectrum_data['frequencies'][np.argmax(spectrum_data['magnitude'])])
        },
        'silence_detection': {
            'threshold': 0.01,
            'min_duration': 0.1,
            'silent_regions': silent_regions,
            'total_silence_time': sum(end - start for start, end in silent_regions)
        }
    }
    
    # Add metadata if provided
    if metadata:
        report['metadata'] = metadata
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)


def detect_beats(audio: np.ndarray, sample_rate: int, 
                 method: str = 'librosa') -> Dict[str, Any]:
    """
    Detect beats and tempo in audio.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        method: Detection method ('librosa', 'onset', 'spectral')
        
    Returns:
        Dictionary containing beat information
    """
    if method == 'librosa':
        # Use librosa's beat tracking
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sample_rate)
        beat_times = librosa.frames_to_time(beats, sr=sample_rate)
        
        return {
            'tempo': tempo,
            'beats': beats,
            'beat_times': beat_times,
            'method': method
        }
    
    elif method == 'onset':
        # Onset-based beat detection
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sample_rate)
        onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)
        
        # Estimate tempo from onset intervals
        if len(onset_times) > 1:
            intervals = np.diff(onset_times)
            tempo = 60.0 / np.median(intervals)
        else:
            tempo = 120.0
        
        return {
            'tempo': tempo,
            'onsets': onset_frames,
            'onset_times': onset_times,
            'method': method
        }
    
    elif method == 'spectral':
        # Spectral flux-based beat detection
        spectral_flux = librosa.onset.onset_strength(y=audio, sr=sample_rate)
        onset_frames = librosa.onset.onset_detect(onset_envelope=spectral_flux, sr=sample_rate)
        onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)
        
        # Estimate tempo
        if len(onset_times) > 1:
            intervals = np.diff(onset_times)
            tempo = 60.0 / np.median(intervals)
        else:
            tempo = 120.0
        
        return {
            'tempo': tempo,
            'spectral_flux': spectral_flux,
            'onsets': onset_frames,
            'onset_times': onset_times,
            'method': method
        }
    
    else:
        raise ValueError(f"Unknown beat detection method: {method}")


def extract_audio_features(audio: np.ndarray, sample_rate: int, 
                          feature_set: str = 'basic') -> Dict[str, np.ndarray]:
    """
    Extract comprehensive audio features for machine learning.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        feature_set: Feature set to extract ('basic', 'mfcc', 'chroma', 'full')
        
    Returns:
        Dictionary containing extracted features
    """
    features = {}
    
    if feature_set in ['basic', 'mfcc', 'chroma', 'full']:
        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        features['mfcc'] = mfcc
        features['mfcc_mean'] = np.mean(mfcc, axis=1)
        features['mfcc_std'] = np.std(mfcc, axis=1)
    
    if feature_set in ['chroma', 'full']:
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        features['chroma'] = chroma
        features['chroma_mean'] = np.mean(chroma, axis=1)
    
    if feature_set in ['full']:
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        
        features['spectral_centroid'] = spectral_centroid
        features['spectral_rolloff'] = spectral_rolloff
        features['spectral_bandwidth'] = spectral_bandwidth
        
        # Rhythm features
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
        features['tempo'] = tempo
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features['zero_crossing_rate'] = zcr
        
        # Root mean square energy
        rms = librosa.feature.rms(y=audio)
        features['rms_energy'] = rms
    
    return features


def remove_noise(audio: np.ndarray, sample_rate: int, 
                 method: str = 'spectral_gate', threshold: float = 0.1) -> np.ndarray:
    """
    Remove noise from audio using various methods.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        method: Noise removal method ('spectral_gate', 'wiener', 'spectral_subtraction')
        threshold: Noise threshold
        
    Returns:
        Noise-reduced audio
    """
    if method == 'spectral_gate':
        # Spectral gating - simple frequency domain noise gate
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        
        # Apply threshold
        noise_mask = magnitude > (threshold * np.max(magnitude))
        fft_filtered = fft * noise_mask
        
        return np.fft.irfft(fft_filtered, len(audio))
    
    elif method == 'wiener':
        # Wiener filtering (simplified)
        from scipy.signal import wiener
        return wiener(audio, mysize=min(100, len(audio)//10))
    
    elif method == 'spectral_subtraction':
        # Spectral subtraction
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        # Estimate noise from first 1000 samples
        noise_estimate = np.mean(magnitude[:1000])
        
        # Subtract noise estimate
        magnitude_clean = np.maximum(magnitude - noise_estimate * threshold, 0.1 * magnitude)
        
        # Reconstruct signal
        fft_clean = magnitude_clean * np.exp(1j * phase)
        return np.fft.irfft(fft_clean, len(audio))
    
    else:
        raise ValueError(f"Unknown noise removal method: {method}")


def enhance_audio(audio: np.ndarray, sample_rate: int, 
                  enhancement_type: str = 'brightness', amount: float = 0.5) -> np.ndarray:
    """
    Apply audio enhancement techniques.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        enhancement_type: Type of enhancement ('brightness', 'warmth', 'clarity', 'presence')
        amount: Enhancement amount (0.0 to 1.0)
        
    Returns:
        Enhanced audio
    """
    if enhancement_type == 'brightness':
        # High-frequency boost
        from scipy.signal import butter, filtfilt
        
        nyquist = sample_rate / 2
        high_freq = 8000 / nyquist
        b, a = butter(2, high_freq, btype='high')
        
        high_freq_audio = filtfilt(b, a, audio)
        return audio + amount * high_freq_audio
    
    elif enhancement_type == 'warmth':
        # Low-frequency boost
        from scipy.signal import butter, filtfilt
        
        nyquist = sample_rate / 2
        low_freq = 200 / nyquist
        b, a = butter(2, low_freq, btype='low')
        
        low_freq_audio = filtfilt(b, a, audio)
        return audio + amount * 0.5 * low_freq_audio
    
    elif enhancement_type == 'clarity':
        # Mid-frequency boost
        from scipy.signal import butter, filtfilt
        
        nyquist = sample_rate / 2
        mid_low = 1000 / nyquist
        mid_high = 4000 / nyquist
        b, a = butter(2, [mid_low, mid_high], btype='band')
        
        mid_freq_audio = filtfilt(b, a, audio)
        return audio + amount * mid_freq_audio
    
    elif enhancement_type == 'presence':
        # Presence boost (2-8kHz)
        from scipy.signal import butter, filtfilt
        
        nyquist = sample_rate / 2
        presence_low = 2000 / nyquist
        presence_high = 8000 / nyquist
        b, a = butter(2, [presence_low, presence_high], btype='band')
        
        presence_audio = filtfilt(b, a, audio)
        return audio + amount * presence_audio
    
    else:
        raise ValueError(f"Unknown enhancement type: {enhancement_type}")


def create_advanced_visualization(audio: np.ndarray, sample_rate: int,
                                 output_path: Optional[str] = None,
                                 title: str = "Advanced Audio Analysis") -> None:
    """
    Create advanced audio visualization with multiple analysis views.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        output_path: Path to save visualization (None to display)
        title: Plot title
    """
    # Create figure with more subplots
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle(title, fontsize=16)
    
    # 1. Waveform
    time = np.arange(len(audio)) / sample_rate
    axes[0, 0].plot(time, audio, alpha=0.7)
    axes[0, 0].set_title('Waveform')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Spectrum
    spectrum_data = analyze_spectrum(audio, sample_rate)
    axes[0, 1].semilogx(spectrum_data['frequencies'], spectrum_data['magnitude_db'])
    axes[0, 1].set_title('Frequency Spectrum')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude (dB)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(20, sample_rate/2)
    
    # 3. Mel-spectrogram
    im = axes[0, 2].imshow(spectrum_data['mel_spectrogram_db'], 
                           aspect='auto', origin='lower',
                           extent=[0, len(audio)/sample_rate, 0, 128])
    axes[0, 2].set_title('Mel-Spectrogram')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Mel Frequency Bins')
    plt.colorbar(im, ax=axes[0, 2], label='Power (dB)')
    
    # 4. Loudness over time
    window_size = int(0.1 * sample_rate)  # 100ms windows
    loudness_time = []
    time_windows = []
    
    for i in range(0, len(audio), window_size):
        window = audio[i:i+window_size]
        if len(window) > 0:
            rms = np.sqrt(np.mean(window**2))
            loudness_time.append(20 * np.log10(rms + 1e-10))
            time_windows.append(i / sample_rate)
    
    axes[1, 0].plot(time_windows, loudness_time, color='orange')
    axes[1, 0].set_title('Loudness Over Time')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('RMS (dB)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Phase spectrum
    axes[1, 1].plot(spectrum_data['frequencies'], spectrum_data['phase'])
    axes[1, 1].set_title('Phase Spectrum')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Phase (radians)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(20, sample_rate/2)
    
    # 6. Beat detection
    try:
        beat_data = detect_beats(audio, sample_rate)
        axes[1, 2].plot(time, audio, alpha=0.5)
        axes[1, 2].vlines(beat_data['beat_times'], -1, 1, color='red', alpha=0.7)
        axes[1, 2].set_title(f'Beat Detection (Tempo: {beat_data["tempo"]:.1f} BPM)')
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel('Amplitude')
        axes[1, 2].grid(True, alpha=0.3)
    except:
        axes[1, 2].text(0.5, 0.5, 'Beat detection failed', 
                        transform=axes[1, 2].transAxes, ha='center')
        axes[1, 2].set_title('Beat Detection')
    
    # 7. MFCC features
    try:
        features = extract_audio_features(audio, sample_rate, 'mfcc')
        axes[2, 0].imshow(features['mfcc'], aspect='auto', origin='lower')
        axes[2, 0].set_title('MFCC Features')
        axes[2, 0].set_xlabel('Time Frame')
        axes[2, 0].set_ylabel('MFCC Coefficient')
        plt.colorbar(axes[2, 0].images[0], ax=axes[2, 0])
    except:
        axes[2, 0].text(0.5, 0.5, 'MFCC extraction failed', 
                        transform=axes[2, 0].transAxes, ha='center')
        axes[2, 0].set_title('MFCC Features')
    
    # 8. Chroma features
    try:
        features = extract_audio_features(audio, sample_rate, 'chroma')
        axes[2, 1].imshow(features['chroma'], aspect='auto', origin='lower')
        axes[2, 1].set_title('Chroma Features')
        axes[2, 1].set_xlabel('Time Frame')
        axes[2, 1].set_ylabel('Pitch Class')
        plt.colorbar(axes[2, 1].images[0], ax=axes[2, 1])
    except:
        axes[2, 1].text(0.5, 0.5, 'Chroma extraction failed', 
                        transform=axes[2, 1].transAxes, ha='center')
        axes[2, 1].set_title('Chroma Features')
    
    # 9. Spectral centroid over time
    try:
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        centroid_times = librosa.frames_to_time(np.arange(len(spectral_centroid[0])), sr=sample_rate)
        axes[2, 2].plot(centroid_times, spectral_centroid[0], color='green')
        axes[2, 2].set_title('Spectral Centroid Over Time')
        axes[2, 2].set_xlabel('Time (s)')
        axes[2, 2].set_ylabel('Frequency (Hz)')
        axes[2, 2].grid(True, alpha=0.3)
    except:
        axes[2, 2].text(0.5, 0.5, 'Spectral analysis failed', 
                        transform=axes[2, 2].transAxes, ha='center')
        axes[2, 2].set_title('Spectral Centroid')
    
    # 10. Statistics
    loudness_stats = analyze_loudness(audio, sample_rate)
    stats_text = f"""
    RMS: {loudness_stats['rms_db']:.1f} dB
    Peak: {loudness_stats['peak_db']:.1f} dB
    Crest Factor: {loudness_stats['crest_factor']:.2f}
    LUFS: {loudness_stats['integrated_lufs']:.1f}
    Dynamic Range: {loudness_stats['dynamic_range']:.1f} dB
    Duration: {len(audio)/sample_rate:.2f} s
    Sample Rate: {sample_rate} Hz
    """
    
    axes[3, 0].text(0.1, 0.5, stats_text, transform=axes[3, 0].transAxes,
                     fontsize=10, verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='lightgray'))
    axes[3, 0].set_title('Audio Statistics')
    axes[3, 0].axis('off')
    
    # 11. Zero crossing rate
    try:
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_times = librosa.frames_to_time(np.arange(len(zcr[0])), sr=sample_rate)
        axes[3, 1].plot(zcr_times, zcr[0], color='purple')
        axes[3, 1].set_title('Zero Crossing Rate')
        axes[3, 1].set_xlabel('Time (s)')
        axes[3, 1].set_ylabel('Rate')
        axes[3, 1].grid(True, alpha=0.3)
    except:
        axes[3, 1].text(0.5, 0.5, 'ZCR analysis failed', 
                        transform=axes[3, 1].transAxes, ha='center')
        axes[3, 1].set_title('Zero Crossing Rate')
    
    # 12. RMS energy over time
    try:
        rms = librosa.feature.rms(y=audio)
        rms_times = librosa.frames_to_time(np.arange(len(rms[0])), sr=sample_rate)
        axes[3, 2].plot(rms_times, rms[0], color='brown')
        axes[3, 2].set_title('RMS Energy Over Time')
        axes[3, 2].set_xlabel('Time (s)')
        axes[3, 2].set_ylabel('Energy')
        axes[3, 2].grid(True, alpha=0.3)
    except:
        axes[3, 2].text(0.5, 0.5, 'RMS analysis failed', 
                        transform=axes[3, 2].transAxes, ha='center')
        axes[3, 2].set_title('RMS Energy')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_audio_comparison(audio1: np.ndarray, audio2: np.ndarray, 
                           sample_rate: int, labels: Tuple[str, str] = ('Original', 'Processed'),
                           output_path: Optional[str] = None) -> None:
    """
    Create side-by-side comparison of two audio signals.
    
    Args:
        audio1: First audio signal
        audio2: Second audio signal
        sample_rate: Audio sample rate
        labels: Labels for the two signals
        output_path: Path to save visualization (None to display)
    """
    # Ensure both signals are the same length
    max_length = max(len(audio1), len(audio2))
    audio1_padded = librosa.util.fix_length(audio1, size=max_length)
    audio2_padded = librosa.util.fix_length(audio2, size=max_length)
    
    # Create comparison plot
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Audio Comparison Analysis', fontsize=16)
    
    time = np.arange(max_length) / sample_rate
    
    # 1. Waveform comparison
    axes[0, 0].plot(time, audio1_padded, alpha=0.7, label=labels[0])
    axes[0, 0].set_title('Waveform Comparison')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(time, audio2_padded, alpha=0.7, label=labels[1], color='orange')
    axes[0, 1].set_title('Waveform Comparison')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 2. Spectrum comparison
    spectrum1 = analyze_spectrum(audio1_padded, sample_rate)
    spectrum2 = analyze_spectrum(audio2_padded, sample_rate)
    
    axes[1, 0].semilogx(spectrum1['frequencies'], spectrum1['magnitude_db'], 
                         alpha=0.7, label=labels[0])
    axes[1, 0].set_title('Frequency Spectrum Comparison')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude (dB)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(20, sample_rate/2)
    
    axes[1, 1].semilogx(spectrum2['frequencies'], spectrum2['magnitude_db'], 
                         alpha=0.7, label=labels[1], color='orange')
    axes[1, 1].set_title('Frequency Spectrum Comparison')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude (dB)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(20, sample_rate/2)
    
    # 3. Loudness comparison
    loudness1 = analyze_loudness(audio1_padded, sample_rate)
    loudness2 = analyze_loudness(audio2_padded, sample_rate)
    
    # Create bar chart comparison
    metrics = ['RMS (dB)', 'Peak (dB)', 'Crest Factor', 'LUFS']
    values1 = [loudness1['rms_db'], loudness1['peak_db'], 
               loudness1['crest_factor'], loudness1['integrated_lufs']]
    values2 = [loudness2['rms_db'], loudness2['peak_db'], 
               loudness2['crest_factor'], loudness2['integrated_lufs']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[2, 0].bar(x - width/2, values1, width, label=labels[0], alpha=0.7)
    axes[2, 0].bar(x + width/2, values2, width, label=labels[1], alpha=0.7, color='orange')
    axes[2, 0].set_title('Loudness Metrics Comparison')
    axes[2, 0].set_xlabel('Metrics')
    axes[2, 0].set_ylabel('Value')
    axes[2, 0].set_xticks(x)
    axes[2, 0].set_xticklabels(metrics)
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Difference plot
    differences = [v2 - v1 for v1, v2 in zip(values1, values2)]
    axes[2, 1].bar(metrics, differences, color='red', alpha=0.7)
    axes[2, 1].set_title('Difference (Processed - Original)')
    axes[2, 1].set_xlabel('Metrics')
    axes[2, 1].set_ylabel('Difference')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def analyze_audio_quality(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Analyze audio quality metrics.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        
    Returns:
        Dictionary containing quality metrics
    """
    # Basic metrics
    loudness_stats = analyze_loudness(audio, sample_rate)
    
    # Spectral analysis
    spectrum_data = analyze_spectrum(audio, sample_rate)
    
    # Calculate additional quality metrics
    # Signal-to-noise ratio approximation
    signal_power = np.mean(audio**2)
    noise_floor = np.percentile(audio**2, 5)  # 5th percentile as noise estimate
    snr = 10 * np.log10(signal_power / (noise_floor + 1e-10))
    
    # Spectral flatness
    spectral_flatness = np.exp(np.mean(np.log(spectrum_data['power'] + 1e-10)))
    
    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate).mean()
    
    # Spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate).mean()
    
    # Harmonic-percussive separation
    harmonic, percussive = librosa.effects.hpss(audio)
    harmonic_ratio = np.sum(harmonic**2) / (np.sum(harmonic**2) + np.sum(percussive**2) + 1e-10)
    
    return {
        'basic_metrics': loudness_stats,
        'signal_to_noise_ratio': snr,
        'spectral_flatness': spectral_flatness,
        'spectral_rolloff': spectral_rolloff,
        'spectral_contrast': spectral_contrast,
        'harmonic_ratio': harmonic_ratio,
        'dynamic_range': loudness_stats['dynamic_range'],
        'crest_factor': loudness_stats['crest_factor'],
        'quality_score': _calculate_quality_score(loudness_stats, snr, spectral_flatness, harmonic_ratio)
    }


def _calculate_quality_score(loudness_stats: Dict[str, float], snr: float, 
                           spectral_flatness: float, harmonic_ratio: float) -> float:
    """
    Calculate overall audio quality score (0-100).
    
    Args:
        loudness_stats: Loudness analysis results
        snr: Signal-to-noise ratio
        spectral_flatness: Spectral flatness measure
        harmonic_ratio: Harmonic to percussive ratio
        
    Returns:
        Quality score from 0 to 100
    """
    score = 0.0
    
    # Dynamic range scoring (0-25 points)
    dr = loudness_stats['dynamic_range']
    if dr > 20:
        score += 25
    elif dr > 15:
        score += 20
    elif dr > 10:
        score += 15
    elif dr > 5:
        score += 10
    else:
        score += 5
    
    # SNR scoring (0-25 points)
    if snr > 40:
        score += 25
    elif snr > 30:
        score += 20
    elif snr > 20:
        score += 15
    elif snr > 10:
        score += 10
    else:
        score += 5
    
    # Spectral flatness scoring (0-25 points)
    # Lower flatness is better for most music
    if spectral_flatness < 0.1:
        score += 25
    elif spectral_flatness < 0.3:
        score += 20
    elif spectral_flatness < 0.5:
        score += 15
    elif spectral_flatness < 0.7:
        score += 10
    else:
        score += 5
    
    # Harmonic content scoring (0-25 points)
    if harmonic_ratio > 0.7:
        score += 25
    elif harmonic_ratio > 0.5:
        score += 20
    elif harmonic_ratio > 0.3:
        score += 15
    elif harmonic_ratio > 0.1:
        score += 10
    else:
        score += 5
    
    return score
