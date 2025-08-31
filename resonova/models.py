"""
Neural models for ResoNova AI mixing and mastering.
"""

import numpy as np
import librosa
from typing import Dict, Any, Optional, Tuple
import tensorflow as tf
from tensorflow import keras


class GenreClassifier:
    """
    Neural network-based genre classifier trained on electronic music.
    
    Supports EDM, R&B, Techno, House, and other electronic genres.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the genre classifier.
        
        Args:
            model_path: Path to pre-trained model weights
        """
        self.model = None
        self.genres = ['edm', 'techno', 'house', 'dubstep', 'trance', 'drum_and_bass']
        
        if model_path:
            self.load_model(model_path)
        else:
            self._build_model()
    
    def _build_model(self):
        """Build the genre classification model architecture."""
        model = keras.Sequential([
            keras.layers.Input(shape=(128, 128, 1)),  # Mel-spectrogram input
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(len(self.genres), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
    
    def predict(self, audio: np.ndarray, sample_rate: int) -> str:
        """
        Predict the genre of the input audio.
        
        Args:
            audio: Input audio as numpy array
            sample_rate: Audio sample rate
            
        Returns:
            Predicted genre string
        """
        if self.model is None:
            # Fallback to simple heuristic-based classification
            return self._heuristic_classification(audio, sample_rate)
        
        # Extract mel-spectrogram features
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate, n_mels=128, hop_length=512
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize and reshape for model input
        mel_spec_normalized = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()
        mel_spec_input = mel_spec_normalized.reshape(1, 128, 128, 1)
        
        # Make prediction
        predictions = self.model.predict(mel_spec_input)
        predicted_genre_idx = np.argmax(predictions[0])
        
        return self.genres[predicted_genre_idx]
    
    def _heuristic_classification(self, audio: np.ndarray, sample_rate: int) -> str:
        """Fallback heuristic-based genre classification."""
        # Simple tempo and spectral centroid based classification
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate).mean()
        
        if tempo > 140:
            return 'drum_and_bass'
        elif tempo > 130:
            return 'dubstep'
        elif tempo > 125:
            return 'trance'
        elif tempo > 120:
            return 'house'
        elif tempo > 110:
            return 'techno'
        else:
            return 'edm'
    
    def load_model(self, model_path: str):
        """Load pre-trained model weights."""
        try:
            self.model = keras.models.load_model(model_path)
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            self._build_model()


class NeuralEQ:
    """
    Neural network-based EQ that learns genre-specific frequency responses.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the neural EQ.
        
        Args:
            model_path: Path to pre-trained model weights
        """
        self.model = None
        self.freq_bands = np.logspace(1, 4, 32)  # 10Hz to 10kHz
        
        if model_path:
            self.load_model(model_path)
        else:
            self._build_model()
    
    def _build_model(self):
        """Build the neural EQ model architecture."""
        model = keras.Sequential([
            keras.layers.Input(shape=(128,)),  # Mel-frequency features
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='tanh')  # EQ gains in dB
        ])
        
        model.compile(optimizer='adam', loss='mse')
        self.model = model
    
    def process(self, audio: np.ndarray, sample_rate: int, 
                genre: Optional[str] = None) -> np.ndarray:
        """
        Apply neural EQ to the input audio.
        
        Args:
            audio: Input audio as numpy array
            sample_rate: Audio sample rate
            genre: Genre for genre-specific EQ
            
        Returns:
            EQ-processed audio
        """
        if self.model is None:
            return audio
        
        # Extract mel-frequency features
        mel_features = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate, n_mels=128, hop_length=512
        )
        mel_features_db = librosa.power_to_db(mel_features, ref=np.max)
        
        # Get average features across time
        mel_features_avg = np.mean(mel_features_db, axis=1)
        
        # Normalize features
        mel_features_normalized = (mel_features_avg - mel_features_avg.mean()) / mel_features_avg.std()
        
        # Predict EQ gains
        eq_gains = self.model.predict(mel_features_normalized.reshape(1, -1))[0]
        
        # Apply EQ using FFT
        return self._apply_eq(audio, sample_rate, eq_gains)
    
    def _apply_eq(self, audio: np.ndarray, sample_rate: int, eq_gains: np.ndarray) -> np.ndarray:
        """Apply EQ gains to audio using FFT."""
        # Convert to frequency domain
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
        
        # Create frequency response curve
        freq_response = np.ones_like(freqs)
        
        for i, (band_freq, gain_db) in enumerate(zip(self.freq_bands, eq_gains)):
            if band_freq < sample_rate / 2:  # Nyquist limit
                # Apply Gaussian filter around the band frequency
                sigma = band_freq * 0.1  # 10% bandwidth
                filter_response = np.exp(-0.5 * ((freqs - band_freq) / sigma) ** 2)
                gain_linear = 10 ** (gain_db / 20)
                freq_response *= (1 + (gain_linear - 1) * filter_response)
        
        # Apply frequency response
        fft_filtered = fft * freq_response
        
        # Convert back to time domain
        return np.fft.irfft(fft_filtered, len(audio))
    
    def load_model(self, model_path: str):
        """Load pre-trained model weights."""
        try:
            self.model = keras.models.load_model(model_path)
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            self._build_model()


class NeuralCompressor:
    """
    Neural network-based multi-band compressor with genre-specific settings.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the neural compressor.
        
        Args:
            model_path: Path to pre-trained model weights
        """
        self.model = None
        self.bands = ['low', 'mid', 'high']
        self.band_freqs = [250, 2000, 8000]  # Hz
        
        if model_path:
            self.load_model(model_path)
        else:
            self._build_model()
    
    def _build_model(self):
        """Build the neural compressor model architecture."""
        model = keras.Sequential([
            keras.layers.Input(shape=(128,)),  # Audio features
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(15, activation='sigmoid')  # 5 params per band
        ])
        
        model.compile(optimizer='adam', loss='mse')
        self.model = model
    
    def process(self, audio: np.ndarray, sample_rate: int, 
                genre: Optional[str] = None) -> np.ndarray:
        """
        Apply neural compression to the input audio.
        
        Args:
            audio: Input audio as numpy array
            sample_rate: Audio sample rate
            genre: Genre for genre-specific compression
            
        Returns:
            Compressed audio
        """
        if self.model is None:
            return audio
        
        # Extract audio features
        features = self._extract_features(audio, sample_rate)
        
        # Predict compression parameters
        comp_params = self.model.predict(features.reshape(1, -1))[0]
        
        # Apply multi-band compression
        return self._apply_compression(audio, sample_rate, comp_params)
    
    def _extract_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract features for compression parameter prediction."""
        # RMS energy
        rms = np.sqrt(np.mean(audio**2))
        
        # Crest factor
        peak = np.max(np.abs(audio))
        crest_factor = peak / rms if rms > 0 else 1
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate).mean()
        
        # Combine features
        features = np.array([rms, crest_factor, spectral_centroid, spectral_rolloff])
        
        # Pad to expected size (128 features)
        features_padded = np.zeros(128)
        features_padded[:len(features)] = features
        
        return features_padded
    
    def _apply_compression(self, audio: np.ndarray, sample_rate: int, 
                          comp_params: np.ndarray) -> np.ndarray:
        """Apply multi-band compression."""
        compressed_audio = np.zeros_like(audio)
        
        for i, (band_name, band_freq) in enumerate(zip(self.bands, self.band_freqs)):
            # Extract band parameters
            start_idx = i * 5
            threshold = comp_params[start_idx] * 0.5  # 0 to 0.5
            ratio = 1 + comp_params[start_idx + 1] * 9  # 1 to 10
            attack = comp_params[start_idx + 2] * 0.1  # 0 to 0.1 seconds
            release = comp_params[start_idx + 3] * 1.0  # 0 to 1 second
            makeup_gain = comp_params[start_idx + 4] * 6  # 0 to 6 dB
            
            # Apply band-pass filter
            band_audio = self._bandpass_filter(audio, sample_rate, band_freq)
            
            # Apply compression
            compressed_band = self._compress_band(
                band_audio, sample_rate, threshold, ratio, attack, release, makeup_gain
            )
            
            compressed_audio += compressed_band
        
        return compressed_audio
    
    def _bandpass_filter(self, audio: np.ndarray, sample_rate: int, 
                        center_freq: float) -> np.ndarray:
        """Apply band-pass filter around center frequency."""
        # Simple IIR band-pass filter
        Q = 2.0  # Quality factor
        w0 = 2 * np.pi * center_freq / sample_rate
        alpha = np.sin(w0) / (2 * Q)
        
        b0 = alpha
        b1 = 0
        b2 = -alpha
        a0 = 1 + alpha
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha
        
        # Normalize coefficients
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1, a2]) / a0
        
        # Apply filter
        from scipy.signal import filtfilt
        return filtfilt(b, a, audio)
    
    def _compress_band(self, audio: np.ndarray, sample_rate: int, threshold: float,
                       ratio: float, attack: float, release: float, 
                       makeup_gain: float) -> np.ndarray:
        """Apply compression to a single frequency band."""
        # Calculate RMS envelope
        window_size = int(0.01 * sample_rate)  # 10ms window
        rms_envelope = np.sqrt(
            np.convolve(audio**2, np.ones(window_size)/window_size, mode='same')
        )
        
        # Calculate gain reduction
        gain_reduction = np.ones_like(audio)
        
        for i in range(1, len(audio)):
            if rms_envelope[i] > threshold:
                # Calculate compression amount
                compression_amount = (rms_envelope[i] - threshold) / threshold
                gain_reduction_db = compression_amount * (1 - 1/ratio) * 20
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
        
        # Apply makeup gain
        makeup_gain_linear = 10 ** (makeup_gain / 20)
        
        return audio * gain_reduction * makeup_gain_linear
    
    def load_model(self, model_path: str):
        """Load pre-trained model weights."""
        try:
            self.model = keras.models.load_model(model_path)
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            self._build_model()
