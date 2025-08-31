"""
ResoNova - Open-Source AI Mixing & Mastering for Electronic Music

A transparent, fully-customizable AI pipeline that balances, compresses and limits
electronic tracks to industry loudness specs—powered by LibROSA & TensorFlow.
"""

__version__ = "0.1.0"
__author__ = "ResoNova Team"
__email__ = "team@resonova.ai"

from .core import ResoNova
from .dsp import register, get_dsp_module, list_dsp_modules
from .models import GenreClassifier, NeuralEQ, NeuralCompressor
from .effects import (
    EffectChain, MasteringChain, VocalChain, DrumChain, BassChain,
    create_custom_chain, get_preset_chains, analyze_chain_effect
)
from .utils import (
    load_audio, save_audio, analyze_loudness, analyze_spectrum,
    detect_beats, extract_audio_features, remove_noise, enhance_audio,
    create_advanced_visualization, create_audio_comparison, analyze_audio_quality
)

__all__ = [
    "ResoNova",
    "register",
    "get_dsp_module",
    "list_dsp_modules",
    "GenreClassifier",
    "NeuralEQ",
    "NeuralCompressor",
    "EffectChain",
    "MasteringChain",
    "VocalChain",
    "DrumChain",
    "BassChain",
    "create_custom_chain",
    "get_preset_chains",
    "analyze_chain_effect",
    "load_audio",
    "save_audio",
    "analyze_loudness",
    "analyze_spectrum",
    "detect_beats",
    "extract_audio_features",
    "remove_noise",
    "enhance_audio",
    "create_advanced_visualization",
    "create_audio_comparison",
    "analyze_audio_quality",
]
