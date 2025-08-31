"""
Digital Signal Processing (DSP) module for ResoNova.

Provides a plugin system for custom audio processing modules and built-in DSP functions.
"""

import numpy as np
from typing import Callable, Dict, Any, Optional
from functools import wraps
import inspect
import librosa


# Global registry for DSP modules
_DSP_REGISTRY: Dict[str, Callable] = {}


def register(name: str) -> Callable:
    """
    Decorator to register a DSP function in the global registry.
    
    Args:
        name: Name of the DSP module
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(audio: np.ndarray, sample_rate: int, **kwargs) -> np.ndarray:
            # Ensure audio is float32 and in valid range
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Clip audio to prevent overflow
            audio = np.clip(audio, -1.0, 1.0)
            
            # Call the original function
            result = func(audio, sample_rate, **kwargs)
            
            # Ensure output is valid
            if result is None:
                return audio
            
            # Ensure output is float32 and clipped
            if result.dtype != np.float32:
                result = result.astype(np.float32)
            
            result = np.clip(result, -1.0, 1.0)
            
            return result
        
        # Register the wrapped function
        _DSP_REGISTRY[name] = wrapper
        return wrapper
    
    return decorator


def get_dsp_module(name: str) -> Optional[Callable]:
    """
    Get a registered DSP module by name.
    
    Args:
        name: Name of the DSP module
        
    Returns:
        DSP function or None if not found
    """
    return _DSP_REGISTRY.get(name)


def list_dsp_modules() -> list:
    """
    List all registered DSP modules.
    
    Returns:
        List of module names
    """
    return list(_DSP_REGISTRY.keys())


def unregister_dsp_module(name: str) -> bool:
    """
    Unregister a DSP module.
    
    Args:
        name: Name of the DSP module to unregister
        
    Returns:
        True if module was unregistered, False if not found
    """
    if name in _DSP_REGISTRY:
        del _DSP_REGISTRY[name]
        return True
    return False


# Built-in DSP modules

@register('saturation')
def saturation(audio: np.ndarray, sample_rate: int, drive: float = 2.0) -> np.ndarray:
    """
    Apply soft saturation to audio.
    
    Args:
        audio: Input audio
        sample_rate: Audio sample rate
        drive: Saturation drive amount (1.0 = no effect, higher = more saturation)
        
    Returns:
        Saturated audio
    """
    return np.tanh(audio * drive)


@register('exciter')
def exciter(audio: np.ndarray, sample_rate: int, amount: float = 0.3, 
            frequency: float = 8000) -> np.ndarray:
    """
    Apply harmonic excitation to add brightness.
    
    Args:
        audio: Input audio
        sample_rate: Audio sample rate
        amount: Excitation amount (0.0 to 1.0)
        frequency: Excitation frequency in Hz
        
    Returns:
        Excited audio
    """
    # Create high-pass filter
    from scipy.signal import butter, filtfilt
    
    nyquist = sample_rate / 2
    high_pass_freq = frequency / nyquist
    b, a = butter(2, high_pass_freq, btype='high')
    
    # Apply high-pass filter
    high_freq = filtfilt(b, a, audio)
    
    # Apply saturation to high frequencies
    excited_high = np.tanh(high_freq * 3.0)
    
    # Mix with original
    return audio + amount * excited_high


@register('stereo_widener')
def stereo_widener(audio: np.ndarray, sample_rate: int, width: float = 0.5) -> np.ndarray:
    """
    Apply stereo widening effect.
    
    Args:
        audio: Input audio (stereo)
        sample_rate: Audio sample rate
        width: Stereo width (0.0 = mono, 1.0 = maximum width)
        
    Returns:
        Widened stereo audio
    """
    if audio.ndim != 2 or audio.shape[1] != 2:
        # If not stereo, return as-is
        return audio
    
    left, right = audio[:, 0], audio[:, 1]
    
    # Create mid and side signals
    mid = (left + right) / 2
    side = (left - right) / 2
    
    # Apply width to side signal
    side = side * (1 + width)
    
    # Reconstruct stereo
    new_left = mid + side
    new_right = mid - side
    
    return np.column_stack([new_left, new_right])


@register('multiband_compressor')
def multiband_compressor(audio: np.ndarray, sample_rate: int, 
                        low_threshold: float = 0.3, mid_threshold: float = 0.3,
                        high_threshold: float = 0.3, low_ratio: float = 4.0,
                        mid_ratio: float = 4.0, high_ratio: float = 4.0) -> np.ndarray:
    """
    Apply multi-band compression.
    
    Args:
        audio: Input audio
        sample_rate: Audio sample rate
        low_threshold: Low band threshold (0.0 to 1.0)
        mid_threshold: Mid band threshold (0.0 to 1.0)
        high_threshold: High band threshold (0.0 to 1.0)
        low_ratio: Low band compression ratio
        mid_ratio: Mid band compression ratio
        high_ratio: High band compression ratio
        
    Returns:
        Compressed audio
    """
    from scipy.signal import butter, filtfilt
    
    # Define crossover frequencies
    low_cross = 250  # Hz
    high_cross = 2000  # Hz
    
    # Create filters
    nyquist = sample_rate / 2
    
    # Low band filter
    low_b, low_a = butter(4, low_cross / nyquist, btype='low')
    
    # Mid band filter
    mid_b, mid_a = butter(4, [low_cross / nyquist, high_cross / nyquist], btype='band')
    
    # High band filter
    high_b, high_a = butter(4, high_cross / nyquist, btype='high')
    
    # Apply filters
    low_band = filtfilt(low_b, low_a, audio)
    mid_band = filtfilt(mid_b, mid_a, audio)
    high_band = filtfilt(high_b, high_a, audio)
    
    # Apply compression to each band
    def compress_band(band_audio, threshold, ratio):
        # Calculate RMS envelope
        window_size = int(0.01 * sample_rate)  # 10ms window
        rms = np.sqrt(np.convolve(band_audio**2, np.ones(window_size)/window_size, mode='same'))
        
        # Calculate gain reduction
        gain_reduction = np.ones_like(band_audio)
        for i in range(len(band_audio)):
            if rms[i] > threshold:
                compression_amount = (rms[i] - threshold) / threshold
                gain_reduction_db = compression_amount * (1 - 1/ratio) * 20
                gain_reduction[i] = 10 ** (gain_reduction_db / 20)
        
        return band_audio * gain_reduction
    
    # Compress each band
    low_compressed = compress_band(low_band, low_threshold, low_ratio)
    mid_compressed = compress_band(mid_band, mid_threshold, mid_ratio)
    high_compressed = compress_band(high_band, high_threshold, high_ratio)
    
    # Sum bands
    return low_compressed + mid_compressed + high_compressed


@register('reverb')
def reverb(audio: np.ndarray, sample_rate: int, room_size: float = 0.5,
           damping: float = 0.5, wet_level: float = 0.3) -> np.ndarray:
    """
    Apply simple reverb effect.
    
    Args:
        audio: Input audio
        sample_rate: Audio sample rate
        room_size: Room size (0.0 to 1.0)
        damping: High frequency damping (0.0 to 1.0)
        wet_level: Wet signal level (0.0 to 1.0)
        
    Returns:
        Audio with reverb
    """
    # Simple delay-based reverb
    delay_samples = int(room_size * 0.1 * sample_rate)  # 0 to 100ms
    
    # Create delay line
    delayed = np.zeros_like(audio)
    delayed[delay_samples:] = audio[:-delay_samples]
    
    # Apply damping (simple low-pass filter)
    from scipy.signal import butter, filtfilt
    nyquist = sample_rate / 2
    cutoff = (1 - damping) * nyquist * 0.5
    b, a = butter(2, cutoff / nyquist, btype='low')
    delayed = filtfilt(b, a, delayed)
    
    # Mix dry and wet
    dry_level = 1.0 - wet_level
    return dry_level * audio + wet_level * delayed


@register('chorus')
def chorus(audio: np.ndarray, sample_rate: int, rate: float = 1.5,
           depth: float = 0.002, mix: float = 0.5) -> np.ndarray:
    """
    Apply chorus effect.
    
    Args:
        audio: Input audio
        sample_rate: Audio sample rate
        rate: LFO rate in Hz
        depth: Modulation depth in seconds
        mix: Wet/dry mix (0.0 to 1.0)
        
    Returns:
        Audio with chorus effect
    """
    # Generate LFO
    t = np.arange(len(audio)) / sample_rate
    lfo = depth * np.sin(2 * np.pi * rate * t)
    
    # Create modulated delay
    modulated_delay = np.zeros_like(audio)
    
    for i in range(len(audio)):
        delay_samples = int(lfo[i] * sample_rate)
        if i + delay_samples < len(audio):
            modulated_delay[i] = audio[i + delay_samples]
    
    # Mix dry and wet
    dry_level = 1.0 - mix
    return dry_level * audio + mix * modulated_delay


@register('bit_crusher')
def bit_crusher(audio: np.ndarray, sample_rate: int, bit_depth: float = 8.0,
                sample_rate_reduction: float = 0.5) -> np.ndarray:
    """
    Apply bit crushing effect.
    
    Args:
        audio: Input audio
        sample_rate: Audio sample rate
        bit_depth: Target bit depth (1.0 to 16.0)
        sample_rate_reduction: Sample rate reduction factor (0.1 to 1.0)
        
    Returns:
        Bit-crushed audio
    """
    # Quantize to target bit depth
    max_value = 2 ** (bit_depth - 1) - 1
    quantized = np.round(audio * max_value) / max_value
    
    # Apply sample rate reduction
    if sample_rate_reduction < 1.0:
        target_sr = int(sample_rate * sample_rate_reduction)
        from scipy import signal
        
        # Resample
        resampled = signal.resample(quantized, int(len(quantized) * sample_rate_reduction))
        
        # Resample back to original length
        quantized = signal.resample(resampled, len(quantized))
    
    return quantized


@register('distortion')
def distortion(audio: np.ndarray, sample_rate: int, drive: float = 2.0,
              type: str = 'hard') -> np.ndarray:
    """
    Apply distortion effect.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        drive: Distortion drive amount
        type: Distortion type ('hard', 'soft', 'tube')
        
    Returns:
        Distorted audio
    """
    if type == 'hard':
        # Hard clipping
        return np.clip(audio * drive, -1.0, 1.0)
    elif type == 'soft':
        # Soft clipping with cubic curve
        return np.sign(audio) * (1 - np.exp(-np.abs(audio) * drive))
    elif type == 'tube':
        # Tube-like distortion
        return np.sign(audio) * (1 - np.exp(-np.abs(audio) * drive)) / drive
    else:
        return audio


@register('pitch_shift')
def pitch_shift(audio: np.ndarray, sample_rate: int, semitones: float = 0.0,
                quality: str = 'fast') -> np.ndarray:
    """
    Apply pitch shifting to audio.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        semitones: Pitch shift in semitones (positive = higher, negative = lower)
        quality: Quality mode ('fast', 'high', 'professional')
        
    Returns:
        Pitch-shifted audio
    """
    if semitones == 0:
        return audio
    
    # Calculate pitch ratio
    pitch_ratio = 2 ** (semitones / 12)
    
    if quality == 'fast':
        # Simple resampling approach
        target_length = int(len(audio) / pitch_ratio)
        from scipy import signal
        return signal.resample(audio, target_length)
    
    elif quality == 'high':
        # Phase vocoder approach using librosa
        return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=semitones)
    
    elif quality == 'professional':
        # Advanced phase vocoder with formant preservation
        # This would use a more sophisticated algorithm in production
        return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=semitones)
    
    return audio


@register('time_stretch')
def time_stretch(audio: np.ndarray, sample_rate: int, rate: float = 1.0,
                 quality: str = 'fast') -> np.ndarray:
    """
    Apply time stretching to audio.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        rate: Time stretch ratio (1.0 = normal, 2.0 = twice as fast, 0.5 = half speed)
        quality: Quality mode ('fast', 'high', 'professional')
        
    Returns:
        Time-stretched audio
    """
    if rate == 1.0:
        return audio
    
    if quality == 'fast':
        # Simple resampling approach
        target_length = int(len(audio) * rate)
        from scipy import signal
        return signal.resample(audio, target_length)
    
    elif quality in ['high', 'professional']:
        # Phase vocoder approach using librosa
        return librosa.effects.time_stretch(audio, rate=rate)
    
    return audio


@register('harmonic_enhancer')
def harmonic_enhancer(audio: np.ndarray, sample_rate: int, amount: float = 0.3,
                     even_harmonics: bool = True, odd_harmonics: bool = True) -> np.ndarray:
    """
    Enhance harmonics by adding synthesized overtones.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        amount: Enhancement amount (0.0 to 1.0)
        even_harmonics: Include even harmonics
        odd_harmonics: Include odd harmonics
        
    Returns:
        Harmonic-enhanced audio
    """
    enhanced = audio.copy()
    
    # Generate harmonics
    if even_harmonics:
        # Even harmonics (2x, 4x, 6x frequency)
        for harmonic in [2, 4, 6]:
            harmonic_audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=12*np.log2(harmonic))
            enhanced += amount * 0.3 * harmonic_audio
    
    if odd_harmonics:
        # Odd harmonics (3x, 5x frequency)
        for harmonic in [3, 5]:
            harmonic_audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=12*np.log2(harmonic))
            enhanced += amount * 0.2 * harmonic_audio
    
    # Normalize to prevent clipping
    return enhanced / (1 + amount)


@register('sidechain_compressor')
def sidechain_compressor(audio: np.ndarray, sample_rate: int, 
                        sidechain_audio: np.ndarray, threshold: float = 0.3,
                        ratio: float = 4.0, attack: float = 0.01, 
                        release: float = 0.1, depth: float = 0.8) -> np.ndarray:
    """
    Apply sidechain compression using external audio as trigger.
    
    Args:
        audio: Input audio to be compressed
        sample_rate: Audio sample rate
        sidechain_audio: Audio signal to trigger compression
        threshold: Compression threshold (0.0 to 1.0)
        ratio: Compression ratio
        attack: Attack time in seconds
        release: Release time in seconds
        depth: Maximum compression depth (0.0 to 1.0)
        
    Returns:
        Sidechain-compressed audio
    """
    # Ensure sidechain audio is same length
    if len(sidechain_audio) != len(audio):
        sidechain_audio = librosa.util.fix_length(sidechain_audio, size=len(audio))
    
    # Calculate RMS envelope of sidechain
    window_size = int(0.01 * sample_rate)  # 10ms window
    sidechain_rms = np.sqrt(
        np.convolve(sidechain_audio**2, np.ones(window_size)/window_size, mode='same')
    )
    
    # Calculate gain reduction
    gain_reduction = np.ones_like(audio)
    
    for i in range(1, len(audio)):
        if sidechain_rms[i] > threshold:
            # Calculate compression amount
            compression_amount = (sidechain_rms[i] - threshold) / threshold
            gain_reduction_db = compression_amount * (1 - 1/ratio) * 20 * depth
            gain_reduction_linear = 10 ** (gain_reduction_db / 20)
            
            # Apply attack/release
            if gain_reduction_linear < gain_reduction[i-1]:
                # Attack phase
                gain_reduction[i] = gain_reduction[i-1] + (
                    gain_reduction_linear - gain_reduction[i-1]
                ) * (1 - np.exp(-1 / (attack * sample_rate)))
            else:
                # Release phase
                gain_reduction[i] = gain_reduction[i-1] + (
                    gain_reduction_linear - gain_reduction[i-1]
                ) * (1 - np.exp(-1 / (release * sample_rate)))
        else:
            gain_reduction[i] = gain_reduction[i-1]
    
    return audio * gain_reduction


@register('frequency_shifter')
def frequency_shifter(audio: np.ndarray, sample_rate: int, shift: float = 100.0,
                     mix: float = 0.5) -> np.ndarray:
    """
    Apply frequency shifting effect.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        shift: Frequency shift in Hz
        mix: Wet/dry mix (0.0 to 1.0)
        
    Returns:
        Frequency-shifted audio
    """
    # Convert to frequency domain
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
    
    # Shift frequencies
    shifted_freqs = freqs + shift
    
    # Create new FFT with shifted frequencies
    shifted_fft = np.zeros_like(fft, dtype=complex)
    
    for i, new_freq in enumerate(shifted_freqs):
        if 0 <= new_freq < sample_rate / 2:
            # Find closest frequency bin
            bin_idx = np.argmin(np.abs(freqs - new_freq))
            shifted_fft[bin_idx] += fft[i]
    
    # Convert back to time domain
    shifted_audio = np.fft.irfft(shifted_fft, len(audio))
    
    # Mix with original
    dry_level = 1.0 - mix
    return dry_level * audio + mix * shifted_audio


@register('ring_modulator')
def ring_modulator(audio: np.ndarray, sample_rate: int, frequency: float = 100.0,
                  mix: float = 0.5) -> np.ndarray:
    """
    Apply ring modulation effect.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        frequency: Modulation frequency in Hz
        mix: Wet/dry mix (0.0 to 1.0)
        
    Returns:
        Ring-modulated audio
    """
    # Generate carrier signal
    t = np.arange(len(audio)) / sample_rate
    carrier = np.sin(2 * np.pi * frequency * t)
    
    # Apply ring modulation
    modulated = audio * carrier
    
    # Mix with original
    dry_level = 1.0 - mix
    return dry_level * audio + mix * modulated


@register('granular_synthesizer')
def granular_synthesizer(audio: np.ndarray, sample_rate: int, grain_size: float = 0.1,
                        grain_density: float = 10.0, pitch_shift: float = 0.0,
                        time_stretch: float = 1.0) -> np.ndarray:
    """
    Apply granular synthesis effect.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        grain_size: Grain size in seconds
        grain_density: Grains per second
        pitch_shift: Pitch shift in semitones
        time_stretch: Time stretch ratio
        
    Returns:
        Granularly synthesized audio
    """
    grain_samples = int(grain_size * sample_rate)
    grains_per_second = grain_density
    
    # Calculate number of grains
    total_grains = int(len(audio) / sample_rate * grains_per_second)
    
    # Initialize output
    output_length = int(len(audio) * time_stretch)
    output = np.zeros(output_length)
    
    for i in range(total_grains):
        # Random grain start position
        start_pos = np.random.randint(0, len(audio) - grain_samples)
        
        # Extract grain
        grain = audio[start_pos:start_pos + grain_samples]
        
        # Apply pitch shift if needed
        if pitch_shift != 0:
            grain = librosa.effects.pitch_shift(grain, sr=sample_rate, n_steps=pitch_shift)
        
        # Apply window function
        window = np.hanning(len(grain))
        grain = grain * window
        
        # Random output position
        output_pos = np.random.randint(0, output_length - len(grain))
        
        # Add grain to output
        output[output_pos:output_pos + len(grain)] += grain
    
    # Normalize
    return output / np.max(np.abs(output))


@register('spectral_filter')
def spectral_filter(audio: np.ndarray, sample_rate: int, filter_type: str = 'bandpass',
                   low_freq: float = 100.0, high_freq: float = 1000.0,
                   resonance: float = 1.0) -> np.ndarray:
    """
    Apply spectral filtering with resonance control.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        filter_type: Filter type ('lowpass', 'highpass', 'bandpass', 'notch')
        low_freq: Low frequency cutoff in Hz
        high_freq: High frequency cutoff in Hz
        resonance: Filter resonance (Q factor)
        
    Returns:
        Filtered audio
    """
    from scipy.signal import butter, filtfilt, iirnotch
    
    nyquist = sample_rate / 2
    
    if filter_type == 'lowpass':
        b, a = butter(4, high_freq / nyquist, btype='low')
    elif filter_type == 'highpass':
        b, a = butter(4, low_freq / nyquist, btype='high')
    elif filter_type == 'bandpass':
        b, a = butter(4, [low_freq / nyquist, high_freq / nyquist], btype='band')
    elif filter_type == 'notch':
        center_freq = (low_freq + high_freq) / 2
        b, a = iirnotch(center_freq, resonance, sample_rate)
    else:
        return audio
    
    return filtfilt(b, a, audio)


@register('dynamic_eq')
def dynamic_eq(audio: np.ndarray, sample_rate: int, frequency: float = 1000.0,
               threshold: float = 0.3, ratio: float = 2.0, q: float = 1.0,
               gain: float = 6.0) -> np.ndarray:
    """
    Apply dynamic EQ that responds to audio level.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        frequency: Center frequency in Hz
        threshold: Threshold for dynamic response
        ratio: Dynamic response ratio
        q: Filter Q factor
        gain: Maximum gain in dB
        
    Returns:
        Dynamically EQ'd audio
    """
    from scipy.signal import butter, filtfilt
    
    # Create bandpass filter around target frequency
    nyquist = sample_rate / 2
    low_freq = frequency / (2 ** (q / 2))
    high_freq = frequency * (2 ** (q / 2))
    
    b, a = butter(4, [low_freq / nyquist, high_freq / nyquist], btype='band')
    
    # Extract frequency band
    band_audio = filtfilt(b, a, audio)
    
    # Calculate RMS envelope
    window_size = int(0.01 * sample_rate)  # 10ms window
    rms_envelope = np.sqrt(
        np.convolve(band_audio**2, np.ones(window_size)/window_size, mode='same')
    )
    
    # Calculate dynamic gain
    dynamic_gain = np.ones_like(audio)
    
    for i in range(len(audio)):
        if rms_envelope[i] > threshold:
            # Calculate gain reduction
            excess = (rms_envelope[i] - threshold) / threshold
            gain_reduction = excess * (1 - 1/ratio)
            dynamic_gain[i] = 1 - gain_reduction
        else:
            # Apply boost when below threshold
            boost = (threshold - rms_envelope[i]) / threshold
            dynamic_gain[i] = 1 + boost * (gain / 20)
    
    # Apply dynamic gain to the frequency band
    processed_band = band_audio * dynamic_gain
    
    # Reconstruct audio
    # Remove original band
    audio_without_band = audio - band_audio
    
    # Add processed band
    return audio_without_band + processed_band


@register('stereo_enhancer')
def stereo_enhancer(audio: np.ndarray, sample_rate: int, width: float = 0.5,
                    phase_shift: float = 0.0, frequency_split: float = 1000.0) -> np.ndarray:
    """
    Enhanced stereo widening with frequency-dependent processing.
    
    Args:
        audio: Input audio data (stereo)
        sample_rate: Audio sample rate
        width: Stereo width (0.0 to 1.0)
        phase_shift: Phase shift in degrees
        frequency_split: Frequency above which to apply enhanced widening
        
    Returns:
        Enhanced stereo audio
    """
    if audio.ndim != 2 or audio.shape[1] != 2:
        return audio
    
    left, right = audio[:, 0], audio[:, 1]
    
    # Create mid and side signals
    mid = (left + right) / 2
    side = (left - right) / 2
    
    # Apply frequency-dependent widening
    from scipy.signal import butter, filtfilt
    
    nyquist = sample_rate / 2
    low_b, low_a = butter(4, frequency_split / nyquist, btype='low')
    high_b, high_a = butter(4, frequency_split / nyquist, btype='high')
    
    # Process low frequencies with standard widening
    low_mid = filtfilt(low_b, low_a, mid)
    low_side = filtfilt(low_b, low_a, side) * (1 + width * 0.5)
    
    # Process high frequencies with enhanced widening
    high_mid = filtfilt(high_b, high_a, mid)
    high_side = filtfilt(high_b, high_a, side) * (1 + width)
    
    # Apply phase shift to high frequencies
    if phase_shift != 0:
        phase_rad = np.radians(phase_shift)
        high_side = high_side * np.cos(phase_rad) + np.roll(high_side, 1) * np.sin(phase_rad)
    
    # Reconstruct stereo
    new_left = low_mid + low_side + high_mid + high_side
    new_right = low_mid - low_side + high_mid - high_side
    
    return np.column_stack([new_left, new_right])


@register('transient_shaper')
def transient_shaper(audio: np.ndarray, sample_rate: int, attack: float = 0.5,
                     sustain: float = 0.5, sensitivity: float = 0.5) -> np.ndarray:
    """
    Shape transients in audio.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        attack: Attack enhancement (0.0 to 1.0)
        sustain: Sustain enhancement (0.0 to 1.0)
        sensitivity: Transient detection sensitivity
        
    Returns:
        Transient-shaped audio
    """
    # Detect transients using envelope follower
    window_size = int(0.005 * sample_rate)  # 5ms window
    envelope = np.sqrt(
        np.convolve(audio**2, np.ones(window_size)/window_size, mode='same')
    )
    
    # Calculate envelope derivative (transient detection)
    envelope_diff = np.diff(envelope, prepend=envelope[0])
    
    # Detect transients
    threshold = sensitivity * np.std(envelope_diff)
    transients = np.abs(envelope_diff) > threshold
    
    # Create transient mask
    transient_mask = np.zeros_like(audio)
    
    for i in range(len(audio)):
        if transients[i]:
            # Apply attack enhancement
            if envelope_diff[i] > 0:  # Attack
                transient_mask[i] = attack
            else:  # Release
                transient_mask[i] = sustain
    
    # Apply transient shaping
    shaped_audio = audio * (1 + transient_mask)
    
    # Normalize to prevent clipping
    return shaped_audio / np.max(np.abs(shaped_audio))


# Utility functions

def get_dsp_module_info(name: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a DSP module.
    
    Args:
        name: Name of the DSP module
        
    Returns:
        Dictionary with module information or None if not found
    """
    if name not in _DSP_REGISTRY:
        return None
    
    func = _DSP_REGISTRY[name]
    sig = inspect.signature(func)
    
    return {
        'name': name,
        'function': func,
        'parameters': list(sig.parameters.keys()),
        'docstring': func.__doc__,
        'source': inspect.getsource(func)
    }


def validate_dsp_module(name: str, audio: np.ndarray, sample_rate: int) -> bool:
    """
    Validate that a DSP module works correctly.
    
    Args:
        name: Name of the DSP module
        audio: Test audio
        sample_rate: Test sample rate
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        module = get_dsp_module(name)
        if module is None:
            return False
        
        # Test with default parameters
        result = module(audio, sample_rate)
        
        # Check output validity
        if result is None:
            return False
        
        if not isinstance(result, np.ndarray):
            return False
        
        if result.shape != audio.shape:
            return False
        
        if result.dtype != np.float32:
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            return False
        
        return True
        
    except Exception:
        return False
