"""
Core ResoNova class that orchestrates the AI mixing and mastering pipeline.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import time

from .models import GenreClassifier, NeuralEQ, NeuralCompressor
from .dsp import get_dsp_module
from .utils import load_audio, save_audio, analyze_loudness


class ResoNova:
    """
    Main ResoNova class for AI-powered mixing and mastering.
    
    This class orchestrates the entire pipeline from input audio to mastered output,
    including genre classification, neural EQ, compression, and limiting.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ResoNova with configuration.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config or self._get_default_config()
        
        # Initialize models
        self.genre_classifier = GenreClassifier()
        self.neural_eq = NeuralEQ()
        self.neural_compressor = NeuralCompressor()
        
        # Processing history
        self.processing_history = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration parameters."""
        return {
            "target_lufs": -14.0,
            "true_peak_ceiling": -1.0,
            "oversample_factor": 4,
            "enable_genre_classification": True,
            "enable_neural_eq": True,
            "enable_compression": True,
            "enable_limiting": True,
            "post_processing": [],
            "output_format": "wav",
            "output_bit_depth": 24
        }
    
    def process_file(self, input_path: str, output_path: Optional[str] = None, 
                    genre: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an audio file through the complete ResoNova pipeline.
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output file (auto-generated if None)
            genre: Genre override (auto-detected if None)
            
        Returns:
            Dictionary containing processing results and metadata
        """
        start_time = time.time()
        
        # Load audio
        audio, sample_rate = load_audio(input_path)
        
        # Process audio
        processed_audio, processing_info = self.process_audio(audio, sample_rate, genre)
        
        # Generate output path if not provided
        if output_path is None:
            input_path_obj = Path(input_path)
            output_path = f"outputs/{input_path_obj.stem}_mastered.{self.config['output_format']}"
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save processed audio
        save_audio(output_path, processed_audio, sample_rate, 
                  bit_depth=self.config['output_bit_depth'])
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Compile results
        results = {
            "input_file": input_path,
            "output_file": output_path,
            "sample_rate": sample_rate,
            "duration": len(audio) / sample_rate,
            "processing_time": processing_time,
            "processing_info": processing_info,
            "config": self.config,
            "loudness_analysis": analyze_loudness(processed_audio, sample_rate)
        }
        
        # Save processing report
        report_path = output_path.replace(f".{self.config['output_format']}", "_report.json")
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def process_audio(self, audio: np.ndarray, sample_rate: int, 
                     genre: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process audio through the ResoNova pipeline.
        
        Args:
            audio: Input audio as numpy array
            sample_rate: Audio sample rate
            genre: Genre override (auto-detected if None)
            
        Returns:
            Tuple of (processed_audio, processing_info)
        """
        processing_info = {}
        processed_audio = audio.copy()
        
        # 1. Genre Classification
        if self.config['enable_genre_classification'] and genre is None:
            genre = self.genre_classifier.predict(processed_audio, sample_rate)
            processing_info['detected_genre'] = genre
        elif genre:
            processing_info['specified_genre'] = genre
        
        # 2. Neural EQ
        if self.config['enable_neural_eq']:
            processed_audio = self.neural_eq.process(
                processed_audio, sample_rate, genre=genre
            )
            processing_info['eq_applied'] = True
        
        # 3. Neural Compression
        if self.config['enable_compression']:
            processed_audio = self.neural_compressor.process(
                processed_audio, sample_rate, genre=genre
            )
            processing_info['compression_applied'] = True
        
        # 4. Limiting
        if self.config['enable_limiting']:
            processed_audio = self._apply_limiting(processed_audio, sample_rate)
            processing_info['limiting_applied'] = True
        
        # 5. Post-processing
        if self.config['post_processing']:
            processed_audio = self._apply_post_processing(processed_audio, sample_rate)
            processing_info['post_processing'] = self.config['post_processing']
        
        # 6. Final loudness normalization
        processed_audio = self._normalize_loudness(processed_audio, sample_rate)
        
        return processed_audio, processing_info
    
    def _apply_limiting(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply look-ahead limiting with LUFS-TruePeak guard."""
        # Simple brickwall limiter implementation
        # In production, this would use a more sophisticated algorithm
        threshold = 10 ** (self.config['true_peak_ceiling'] / 20)
        limited_audio = np.clip(audio, -threshold, threshold)
        return limited_audio
    
    def _apply_post_processing(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply custom post-processing modules."""
        processed_audio = audio.copy()
        
        for module_name in self.config['post_processing']:
            module = get_dsp_module(module_name)
            if module:
                processed_audio = module(processed_audio, sample_rate)
        
        return processed_audio
    
    def _normalize_loudness(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Normalize audio to target LUFS level."""
        current_lufs = analyze_loudness(audio, sample_rate)['integrated_lufs']
        target_lufs = self.config['target_lufs']
        
        if current_lufs != target_lufs:
            gain_db = target_lufs - current_lufs
            gain_linear = 10 ** (gain_db / 20)
            audio = audio * gain_linear
        
        return audio
    
    def get_processing_history(self) -> list:
        """Get the history of all processing operations."""
        return self.processing_history.copy()
    
    def reset_history(self):
        """Clear the processing history."""
        self.processing_history = []
