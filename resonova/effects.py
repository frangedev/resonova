"""
Advanced audio effects and processing chains for ResoNova.

Provides pre-built effect combinations and processing chains for common
mastering and mixing scenarios.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from .dsp import get_dsp_module
from .utils import analyze_loudness, analyze_spectrum


class EffectChain:
    """
    Chain multiple DSP effects together for complex audio processing.
    """
    
    def __init__(self, name: str, effects: List[Dict[str, Any]]):
        """
        Initialize an effect chain.
        
        Args:
            name: Name of the effect chain
            effects: List of effect configurations
        """
        self.name = name
        self.effects = effects
    
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process audio through the entire effect chain.
        
        Args:
            audio: Input audio data
            sample_rate: Audio sample rate
            
        Returns:
            Processed audio
        """
        processed_audio = audio.copy()
        
        for effect_config in self.effects:
            effect_name = effect_config['effect']
            params = effect_config.get('params', {})
            
            # Get the effect function
            effect_func = get_dsp_module(effect_name)
            if effect_func is None:
                print(f"Warning: Effect '{effect_name}' not found, skipping...")
                continue
            
            # Apply the effect
            try:
                processed_audio = effect_func(processed_audio, sample_rate, **params)
            except Exception as e:
                print(f"Warning: Error applying effect '{effect_name}': {e}")
                continue
        
        return processed_audio
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the effect chain."""
        return {
            'name': self.name,
            'effects': self.effects,
            'total_effects': len(self.effects)
        }


class MasteringChain:
    """
    Pre-built mastering chain for electronic music.
    """
    
    def __init__(self, genre: str = 'electronic', target_lufs: float = -14.0):
        """
        Initialize mastering chain.
        
        Args:
            genre: Target genre for mastering
            target_lufs: Target LUFS level
        """
        self.genre = genre
        self.target_lufs = target_lufs
        self.chain = self._build_chain()
    
    def _build_chain(self) -> EffectChain:
        """Build the mastering effect chain."""
        if self.genre == 'electronic':
            effects = [
                {'effect': 'multiband_compressor', 'params': {
                    'low_threshold': 0.4, 'mid_threshold': 0.3, 'high_threshold': 0.2,
                    'low_ratio': 3.0, 'mid_ratio': 4.0, 'high_ratio': 5.0
                }},
                {'effect': 'exciter', 'params': {'amount': 0.3, 'frequency': 8000}},
                {'effect': 'stereo_widener', 'params': {'width': 0.4}},
                {'effect': 'saturation', 'params': {'drive': 1.5}},
                {'effect': 'transient_shaper', 'params': {'attack': 0.6, 'sustain': 0.4}}
            ]
        elif self.genre == 'techno':
            effects = [
                {'effect': 'multiband_compressor', 'params': {
                    'low_threshold': 0.5, 'mid_threshold': 0.4, 'high_threshold': 0.3,
                    'low_ratio': 4.0, 'mid_ratio': 5.0, 'high_ratio': 6.0
                }},
                {'effect': 'harmonic_enhancer', 'params': {'amount': 0.4, 'even_harmonics': True}},
                {'effect': 'stereo_enhancer', 'params': {'width': 0.3, 'frequency_split': 800}},
                {'effect': 'transient_shaper', 'params': {'attack': 0.7, 'sustain': 0.5}}
            ]
        elif self.genre == 'house':
            effects = [
                {'effect': 'multiband_compressor', 'params': {
                    'low_threshold': 0.3, 'mid_threshold': 0.2, 'high_threshold': 0.1,
                    'low_ratio': 2.5, 'mid_ratio': 3.5, 'high_ratio': 4.5
                }},
                {'effect': 'enhance_audio', 'params': {'enhancement_type': 'warmth', 'amount': 0.4}},
                {'effect': 'stereo_widener', 'params': {'width': 0.5}},
                {'effect': 'exciter', 'params': {'amount': 0.2, 'frequency': 6000}}
            ]
        else:
            # Default electronic chain
            effects = [
                {'effect': 'multiband_compressor', 'params': {
                    'low_threshold': 0.4, 'mid_threshold': 0.3, 'high_threshold': 0.2,
                    'low_ratio': 3.0, 'mid_ratio': 4.0, 'high_ratio': 5.0
                }},
                {'effect': 'stereo_widener', 'params': {'width': 0.4}},
                {'effect': 'saturation', 'params': {'drive': 1.3}}
            ]
        
        return EffectChain(f"{self.genre}_mastering", effects)
    
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process audio through the mastering chain.
        
        Args:
            audio: Input audio data
            sample_rate: Audio sample rate
            
        Returns:
            Mastered audio
        """
        # Apply effect chain
        processed_audio = self.chain.process(audio, sample_rate)
        
        # Normalize to target LUFS
        processed_audio = self._normalize_loudness(processed_audio, sample_rate)
        
        return processed_audio
    
    def _normalize_loudness(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Normalize audio to target LUFS level."""
        current_lufs = analyze_loudness(audio, sample_rate)['integrated_lufs']
        
        if current_lufs != self.target_lufs:
            gain_db = self.target_lufs - current_lufs
            gain_linear = 10 ** (gain_db / 20)
            audio = audio * gain_linear
        
        return audio


class VocalChain:
    """
    Pre-built vocal processing chain.
    """
    
    def __init__(self, style: str = 'pop'):
        """
        Initialize vocal chain.
        
        Args:
            style: Vocal style ('pop', 'rap', 'singer_songwriter', 'electronic')
        """
        self.style = style
        self.chain = self._build_chain()
    
    def _build_chain(self) -> EffectChain:
        """Build the vocal effect chain."""
        if self.style == 'pop':
            effects = [
                {'effect': 'spectral_filter', 'params': {
                    'filter_type': 'highpass', 'low_freq': 80, 'high_freq': 8000
                }},
                {'effect': 'enhance_audio', 'params': {'enhancement_type': 'presence', 'amount': 0.6}},
                {'effect': 'multiband_compressor', 'params': {
                    'low_threshold': 0.2, 'mid_threshold': 0.15, 'high_threshold': 0.1,
                    'low_ratio': 2.0, 'mid_ratio': 3.0, 'high_ratio': 4.0
                }},
                {'effect': 'exciter', 'params': {'amount': 0.2, 'frequency': 10000}},
                {'effect': 'reverb', 'params': {'room_size': 0.3, 'wet_level': 0.2}}
            ]
        elif self.style == 'rap':
            effects = [
                {'effect': 'spectral_filter', 'params': {
                    'filter_type': 'highpass', 'low_freq': 100, 'high_freq': 6000
                }},
                {'effect': 'enhance_audio', 'params': {'enhancement_type': 'clarity', 'amount': 0.5}},
                {'effect': 'multiband_compressor', 'params': {
                    'low_threshold': 0.3, 'mid_threshold': 0.2, 'high_threshold': 0.15,
                    'low_ratio': 3.0, 'mid_ratio': 4.0, 'high_ratio': 5.0
                }},
                {'effect': 'saturation', 'params': {'drive': 1.8, 'type': 'soft'}}
            ]
        elif self.style == 'electronic':
            effects = [
                {'effect': 'spectral_filter', 'params': {
                    'filter_type': 'highpass', 'low_freq': 120, 'high_freq': 10000
                }},
                {'effect': 'enhance_audio', 'params': {'enhancement_type': 'brightness', 'amount': 0.4}},
                {'effect': 'multiband_compressor', 'params': {
                    'low_threshold': 0.25, 'mid_threshold': 0.2, 'high_threshold': 0.15,
                    'low_ratio': 2.5, 'mid_ratio': 3.5, 'high_ratio': 4.5
                }},
                {'effect': 'exciter', 'params': {'amount': 0.3, 'frequency': 8000}},
                {'effect': 'chorus', 'params': {'rate': 1.2, 'depth': 0.001, 'mix': 0.3}}
            ]
        else:
            # Default pop chain
            effects = [
                {'effect': 'spectral_filter', 'params': {
                    'filter_type': 'highpass', 'low_freq': 80, 'high_freq': 8000
                }},
                {'effect': 'enhance_audio', 'params': {'enhancement_type': 'presence', 'amount': 0.5}},
                {'effect': 'multiband_compressor', 'params': {
                    'low_threshold': 0.2, 'mid_threshold': 0.15, 'high_threshold': 0.1,
                    'low_ratio': 2.0, 'mid_ratio': 3.0, 'high_ratio': 4.0
                }}
            ]
        
        return EffectChain(f"{self.style}_vocal", effects)
    
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process audio through the vocal chain.
        
        Args:
            audio: Input audio data
            sample_rate: Audio sample rate
            
        Returns:
            Processed vocal audio
        """
        return self.chain.process(audio, sample_rate)


class DrumChain:
    """
    Pre-built drum processing chain.
    """
    
    def __init__(self, style: str = 'electronic'):
        """
        Initialize drum chain.
        
        Args:
            style: Drum style ('electronic', 'acoustic', 'trap', 'house')
        """
        self.style = style
        self.chain = self._build_chain()
    
    def _build_chain(self) -> EffectChain:
        """Build the drum effect chain."""
        if self.style == 'electronic':
            effects = [
                {'effect': 'spectral_filter', 'params': {
                    'filter_type': 'highpass', 'low_freq': 40, 'high_freq': 12000
                }},
                {'effect': 'transient_shaper', 'params': {'attack': 0.8, 'sustain': 0.6}},
                {'effect': 'multiband_compressor', 'params': {
                    'low_threshold': 0.4, 'mid_threshold': 0.3, 'high_threshold': 0.2,
                    'low_ratio': 4.0, 'mid_ratio': 5.0, 'high_ratio': 6.0
                }},
                {'effect': 'enhance_audio', 'params': {'enhancement_type': 'brightness', 'amount': 0.3}},
                {'effect': 'saturation', 'params': {'drive': 1.6}}
            ]
        elif self.style == 'trap':
            effects = [
                {'effect': 'spectral_filter', 'params': {
                    'filter_type': 'highpass', 'low_freq': 30, 'high_freq': 15000
                }},
                {'effect': 'transient_shaper', 'params': {'attack': 0.9, 'sustain': 0.7}},
                {'effect': 'multiband_compressor', 'params': {
                    'low_threshold': 0.5, 'mid_threshold': 0.4, 'high_threshold': 0.3,
                    'low_ratio': 5.0, 'mid_ratio': 6.0, 'high_ratio': 7.0
                }},
                {'effect': 'enhance_audio', 'params': {'enhancement_type': 'clarity', 'amount': 0.4}},
                {'effect': 'saturation', 'params': {'drive': 2.0, 'type': 'hard'}}
            ]
        elif self.style == 'house':
            effects = [
                {'effect': 'spectral_filter', 'params': {
                    'filter_type': 'highpass', 'low_freq': 50, 'high_freq': 10000
                }},
                {'effect': 'transient_shaper', 'params': {'attack': 0.7, 'sustain': 0.5}},
                {'effect': 'multiband_compressor', 'params': {
                    'low_threshold': 0.3, 'mid_threshold': 0.25, 'high_threshold': 0.2,
                    'low_ratio': 3.0, 'mid_ratio': 4.0, 'high_ratio': 5.0
                }},
                {'effect': 'enhance_audio', 'params': {'enhancement_type': 'warmth', 'amount': 0.3}}
            ]
        else:
            # Default electronic chain
            effects = [
                {'effect': 'transient_shaper', 'params': {'attack': 0.8, 'sustain': 0.6}},
                {'effect': 'multiband_compressor', 'params': {
                    'low_threshold': 0.4, 'mid_threshold': 0.3, 'high_threshold': 0.2,
                    'low_ratio': 4.0, 'mid_ratio': 5.0, 'high_ratio': 6.0
                }},
                {'effect': 'saturation', 'params': {'drive': 1.5}}
            ]
        
        return EffectChain(f"{self.style}_drums", effects)
    
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process audio through the drum chain.
        
        Args:
            audio: Input audio data
            sample_rate: Audio sample rate
            
        Returns:
            Processed drum audio
        """
        return self.chain.process(audio, sample_rate)


class BassChain:
    """
    Pre-built bass processing chain.
    """
    
    def __init__(self, style: str = 'electronic'):
        """
        Initialize bass chain.
        
        Args:
            style: Bass style ('electronic', 'acoustic', 'sub', 'mid')
        """
        self.style = style
        self.chain = self._build_chain()
    
    def _build_chain(self) -> EffectChain:
        """Build the bass effect chain."""
        if self.style == 'electronic':
            effects = [
                {'effect': 'spectral_filter', 'params': {
                    'filter_type': 'lowpass', 'low_freq': 200, 'high_freq': 8000
                }},
                {'effect': 'enhance_audio', 'params': {'enhancement_type': 'warmth', 'amount': 0.5}},
                {'effect': 'multiband_compressor', 'params': {
                    'low_threshold': 0.3, 'mid_threshold': 0.25, 'high_threshold': 0.2,
                    'low_ratio': 3.0, 'mid_ratio': 4.0, 'high_ratio': 5.0
                }},
                {'effect': 'saturation', 'params': {'drive': 1.4, 'type': 'tube'}}
            ]
        elif self.style == 'sub':
            effects = [
                {'effect': 'spectral_filter', 'params': {
                    'filter_type': 'lowpass', 'low_freq': 150, 'high_freq': 5000
                }},
                {'effect': 'enhance_audio', 'params': {'enhancement_type': 'warmth', 'amount': 0.6}},
                {'effect': 'multiband_compressor', 'params': {
                    'low_threshold': 0.4, 'mid_threshold': 0.3, 'high_threshold': 0.2,
                    'low_ratio': 4.0, 'mid_ratio': 5.0, 'high_ratio': 6.0
                }},
                {'effect': 'saturation', 'params': {'drive': 1.8, 'type': 'soft'}}
            ]
        elif self.style == 'mid':
            effects = [
                {'effect': 'spectral_filter', 'params': {
                    'filter_type': 'bandpass', 'low_freq': 200, 'high_freq': 8000
                }},
                {'effect': 'enhance_audio', 'params': {'enhancement_type': 'clarity', 'amount': 0.4}},
                {'effect': 'multiband_compressor', 'params': {
                    'low_threshold': 0.25, 'mid_threshold': 0.2, 'high_threshold': 0.15,
                    'low_ratio': 2.5, 'mid_ratio': 3.5, 'high_ratio': 4.5
                }},
                {'effect': 'exciter', 'params': {'amount': 0.2, 'frequency': 6000}}
            ]
        else:
            # Default electronic chain
            effects = [
                {'effect': 'enhance_audio', 'params': {'enhancement_type': 'warmth', 'amount': 0.5}},
                {'effect': 'multiband_compressor', 'params': {
                    'low_threshold': 0.3, 'mid_threshold': 0.25, 'high_threshold': 0.2,
                    'low_ratio': 3.0, 'mid_ratio': 4.0, 'high_ratio': 5.0
                }},
                {'effect': 'saturation', 'params': {'drive': 1.4}}
            ]
        
        return EffectChain(f"{self.style}_bass", effects)
    
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process audio through the bass chain.
        
        Args:
            audio: Input audio data
            sample_rate: Audio sample rate
            
        Returns:
            Processed bass audio
        """
        return self.chain.process(audio, sample_rate)


def create_custom_chain(name: str, effects: List[Dict[str, Any]]) -> EffectChain:
    """
    Create a custom effect chain.
    
    Args:
        name: Name of the chain
        effects: List of effect configurations
        
    Returns:
        Custom effect chain
    """
    return EffectChain(name, effects)


def get_preset_chains() -> Dict[str, EffectChain]:
    """
    Get all available preset effect chains.
    
    Returns:
        Dictionary of preset chains
    """
    return {
        'electronic_mastering': MasteringChain('electronic').chain,
        'techno_mastering': MasteringChain('techno').chain,
        'house_mastering': MasteringChain('house').chain,
        'pop_vocal': VocalChain('pop').chain,
        'rap_vocal': VocalChain('rap').chain,
        'electronic_vocal': VocalChain('electronic').chain,
        'electronic_drums': DrumChain('electronic').chain,
        'trap_drums': DrumChain('trap').chain,
        'house_drums': DrumChain('house').chain,
        'electronic_bass': BassChain('electronic').chain,
        'sub_bass': BassChain('sub').chain,
        'mid_bass': BassChain('mid').chain
    }


def analyze_chain_effect(audio: np.ndarray, sample_rate: int, 
                        chain: EffectChain) -> Dict[str, Any]:
    """
    Analyze the effect of a processing chain on audio.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        chain: Effect chain to analyze
        
    Returns:
        Dictionary containing analysis results
    """
    # Analyze original audio
    original_loudness = analyze_loudness(audio, sample_rate)
    original_spectrum = analyze_spectrum(audio, sample_rate)
    
    # Process audio through chain
    processed_audio = chain.process(audio, sample_rate)
    
    # Analyze processed audio
    processed_loudness = analyze_loudness(processed_audio, sample_rate)
    processed_spectrum = analyze_spectrum(processed_audio, sample_rate)
    
    # Calculate differences
    loudness_diff = {
        'rms_db': processed_loudness['rms_db'] - original_loudness['rms_db'],
        'peak_db': processed_loudness['peak_db'] - original_loudness['peak_db'],
        'crest_factor': processed_loudness['crest_factor'] - original_loudness['crest_factor'],
        'integrated_lufs': processed_loudness['integrated_lufs'] - original_loudness['integrated_lufs']
    }
    
    return {
        'chain_name': chain.name,
        'original_audio': {
            'loudness': original_loudness,
            'spectrum': original_spectrum
        },
        'processed_audio': {
            'loudness': processed_loudness,
            'spectrum': processed_spectrum
        },
        'differences': {
            'loudness': loudness_diff
        },
        'chain_info': chain.get_info()
    }
