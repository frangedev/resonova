# ResoNova – Open-Source AI Mixing & Mastering for Electronic Music

![ResoNova Logo](resonova_logo.png)

**Professional-grade AI audio processing with 25+ DSP effects, pre-built chains, and comprehensive analysis tools**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-0.1.0-orange.svg)](https://github.com/frangedev/resonova)
[![Stars](https://img.shields.io/github/stars/frangedev/resonova?style=social)](https://github.com/frangedev/resonova)

---

## 🎛️ **One-liner**  
ResoNova is a transparent, fully-customizable AI pipeline that balances, compresses and limits electronic tracks to industry loudness specs—powered by LibROSA & TensorFlow. **Now with 25+ DSP effects, pre-built processing chains, and professional analysis tools.**

---

## 🆕 **What's New in v0.1.0**

### **🚀 Advanced DSP Effects**
- **Pitch & Time**: `pitch_shift`, `time_stretch` with quality options
- **Harmonic Processing**: `harmonic_enhancer`, `ring_modulator`
- **Advanced Compression**: `sidechain_compressor`, `dynamic_eq`
- **Creative Effects**: `frequency_shifter`, `granular_synthesizer`
- **Enhanced Processing**: `spectral_filter`, `stereo_enhancer`, `transient_shaper`

### **🔗 Pre-built Effect Chains**
- **Mastering**: Genre-specific chains (electronic, techno, house)
- **Vocals**: Style-specific processing (pop, rap, electronic)
- **Drums**: Genre optimization (electronic, trap, house)
- **Bass**: Style enhancement (electronic, sub, mid)

### **📊 Professional Analysis Tools**
- **Beat Detection**: Multiple algorithms and tempo analysis
- **Feature Extraction**: MFCC, chroma, spectral features for ML
- **Quality Metrics**: Comprehensive scoring system (0-100)
- **Noise Removal**: Multiple denoising methods
- **Audio Enhancement**: Brightness, warmth, clarity, presence

### **🎨 Advanced Visualization**
- **12-panel Analysis**: Comprehensive audio analysis views
- **Comparison Tools**: Side-by-side audio comparison
- **Beat Visualization**: Tempo and beat detection display
- **Feature Analysis**: MFCC, chroma, and spectral visualization

---

## 📌 **Core Features**  
| Stage | What it does | Knobs you can tweak |
|-------|--------------|---------------------|
| **Genre-Aware EQ** | Automatic tonal balancing trained on 20k EDM/R&B/Techno tracks | Target curve, band weights |
| **Smart Compression** | Multi-band RMS & crest-factor control | Threshold ratio, attack/release ranges |
| **Transparent Limiting** | Look-ahead brickwall with LUFS-TruePeak guard | Ceiling, oversample factor |
| **Open DSP** | Replace any block with your own Python DSP | Plug-in API |
| **Effect Chains** | Pre-built processing chains for common scenarios | Custom chain creation |
| **Advanced Analysis** | Professional audio quality metrics and visualization | Multiple analysis modes |

---

## 🚀 **Quick Start**  

```bash
git clone https://github.com/frangedev/resonova.git
cd resonova
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Basic mastering
python cli.py master --in my_track.wav --genre techno --lufs -14

# Apply preset effect chain
python cli.py process --in my_track.wav --chain techno_mastering

# Create advanced visualization
python cli.py visualize --in my_track.wav --type advanced --output analysis.png

# Analyze audio quality
python cli.py analyze --in my_track.wav --features full --output results.json
```

The mastered WAV appears in `outputs/` alongside a JSON report of all moves the AI made.

---

## 🧪 **Tech Stack**  
- **Analysis**  LibROSA (STFT, chroma, MFCC)  
- **ML Engine** TensorFlow 2.x (Keras)  
- **DSP**   Pure Python + PyFFTW + custom Cython extensions  
- **Effects**  25+ professional DSP effects and processing chains
- **Visualization** Matplotlib + Seaborn for comprehensive analysis
- **Export**  32-bit float WAV, 24-bit WAV, or MP3 preview  

---

## 🎚️ **Architecture Overview**  

```
┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
│   Input    │→ │  Genre     │→ │  Neural    │→ │   Brick-   │→ │   Effect   │
│   Track    │  │  Classifier│  │  EQ & DRC  │  │   wall     │  │   Chains   │
└────────────┘  └────────────┘  └────────────┘  └────────────┘  └────────────┘
```

All stages expose a clean `process(buffer: np.ndarray, sr:int) -> np.ndarray` interface for unit-testing and swapping.

---

## 🛠️ **Customization & Advanced Usage**  

### **1. Effect Chains**
Apply pre-built processing chains for common scenarios:
```bash
# Mastering chains
python cli.py process --in track.wav --chain electronic_mastering
python cli.py process --in track.wav --chain techno_mastering

# Vocal processing
python cli.py process --in vocals.wav --chain pop_vocal
python cli.py process --in vocals.wav --chain rap_vocal

# Drum enhancement
python cli.py process --in drums.wav --chain trap_drums
```

### **2. Custom Effect Chains**
Create your own processing chains with JSON configuration:
```json
{
  "name": "custom_edm_mastering",
  "effects": [
    {"effect": "spectral_filter", "params": {"filter_type": "highpass", "low_freq": 30}},
    {"effect": "multiband_compressor", "params": {"low_threshold": 0.5, "low_ratio": 5.0}},
    {"effect": "harmonic_enhancer", "params": {"amount": 0.5}}
  ]
}
```

### **3. Individual Effects**
Apply single DSP effects with custom parameters:
```bash
# Apply saturation
python cli.py apply-effect --in track.wav --effect saturation --params drive=2.0

# Apply multiband compression
python cli.py apply-effect --in track.wav --effect multiband_compressor \
  --params low_threshold=0.3 --params mid_threshold=0.2 --params high_threshold=0.1
```

### **4. Audio Analysis & Visualization**
Comprehensive audio analysis and visualization tools:
```bash
# Basic analysis
python cli.py analyze --in track.wav

# Advanced visualization
python cli.py visualize --in track.wav --type advanced --output analysis.png

# Compare two audio files
python cli.py visualize --in original.wav --type comparison --compare processed.wav
```

### **5. Audio Enhancement & Restoration**
Professional audio enhancement and noise removal:
```bash
# Enhance brightness
python cli.py enhance --in track.wav --type brightness --amount 0.6

# Remove noise
python cli.py denoise --in noisy_audio.wav --method spectral_gate --threshold 0.2
```

---

## 🔧 **Available DSP Effects**

### **Core Effects**
- `saturation` - Soft saturation with drive control
- `exciter` - Harmonic excitation for brightness
- `stereo_widener` - Basic stereo enhancement
- `multiband_compressor` - Multi-band compression
- `reverb` - Simple delay-based reverb
- `chorus` - Chorus effect with LFO modulation
- `bit_crusher` - Bit depth and sample rate reduction
- `distortion` - Multiple distortion types (hard, soft, tube)

### **Advanced Effects**
- `pitch_shift` - Pitch shifting with quality options
- `time_stretch` - Time stretching with phase vocoder
- `harmonic_enhancer` - Harmonic synthesis
- `sidechain_compressor` - External trigger compression
- `frequency_shifter` - Frequency domain shifting
- `ring_modulator` - Ring modulation effects
- `granular_synthesizer` - Granular synthesis
- `spectral_filter` - Advanced filtering
- `dynamic_eq` - Level-responsive EQ
- `stereo_enhancer` - Frequency-dependent stereo
- `transient_shaper` - Transient enhancement

---

## 🎯 **Pre-built Effect Chains**

### **Mastering Chains**
- `electronic_mastering` - General electronic music
- `techno_mastering` - Techno-specific processing
- `house_mastering` - House music optimization

### **Vocal Chains**
- `pop_vocal` - Pop vocal enhancement
- `rap_vocal` - Rap vocal processing
- `electronic_vocal` - Electronic vocal effects

### **Drum Chains**
- `electronic_drums` - Electronic drum enhancement
- `trap_drums` - Trap drum processing
- `house_drums` - House drum optimization

### **Bass Chains**
- `electronic_bass` - Electronic bass processing
- `sub_bass` - Sub-bass enhancement
- `mid_bass` - Mid-range bass processing

---

## 📊 **Benchmarks**  
| Dataset | Baseline LUFS | ResoNova LUFS | Δ TruePeak | CPU RT* |
|---------|---------------|---------------|------------|---------|
| EDM-1k  | -9.3          | -9.0          | +0.2 dBTP  | 0.35×   |
| Techno-500 | -10.1      | -9.8          | +0.1 dBTP  | 0.29×   |

\* Real-time factor on Ryzen 7 5800X, 48 kHz / 512-block.

---

## 🖥️ **Command Line Interface**

ResoNova now includes a comprehensive CLI for all operations:

```bash
# List all available effects
python cli.py list-effects

# List all available effect chains
python cli.py list-chains

# Master audio with AI pipeline
python cli.py master --in track.wav --lufs -14

# Process with effect chains
python cli.py process --in track.wav --chain techno_mastering

# Create visualizations
python cli.py visualize --in track.wav --type advanced

# Analyze audio quality
python cli.py analyze --in track.wav --features full

# Apply individual effects
python cli.py apply-effect --in track.wav --effect saturation --params drive=2.0
```

---

## 🤝 **Contributing**  
1. Fork & branch (`feature/my-idea`)  
2. `pytest tests/` must pass  
3. PR with clear diff + notebook demo → we'll review within 48 h

### **Adding New DSP Effects**
```python
from resonova.dsp import register

@register('my_effect')
def my_effect(audio: np.ndarray, sample_rate: int, **kwargs) -> np.ndarray:
    """Apply my custom effect."""
    # Your effect implementation here
    return processed_audio
```

### **Creating Custom Effect Chains**
```python
from resonova.effects import create_custom_chain

my_chain = create_custom_chain("my_chain", [
    {"effect": "saturation", "params": {"drive": 2.0}},
    {"effect": "stereo_widener", "params": {"width": 0.5}}
])
```

---

## 📄 **License**  
MIT – see [LICENSE](LICENSE).  
Trained weights are CC-BY-SA 4.0. Commercial use allowed; attribution required.

---

## 🙋 **FAQ**  

**Q:** *Can I run it on GPU?*  
**A:** Training benefits from CUDA; inference is CPU-only for portability.

**Q:** *Does it work on vocals?*  
**A:** Yes! Now includes dedicated vocal processing chains for pop, rap, and electronic styles.

**Q:** *How many effects are available?*  
**A:** 25+ professional DSP effects including pitch shifting, granular synthesis, and advanced compression.

**Q:** *Can I create custom processing chains?*  
**A:** Absolutely! Use JSON configuration files to create custom effect chains.

**Q:** *What visualization tools are available?*  
**A:** 12-panel analysis views, comparison tools, and comprehensive audio quality metrics.

---

## 🧑‍💻 **Maintainers**  
- @makalin – lead DSP & effects architecture
- @frangedev – ML & model ops  

---

## 📚 **Documentation & Examples**

- **[Examples Directory](examples/)** - Configuration files and usage examples
- **[CLI Reference](cli.py)** - Complete command-line interface documentation
- **[Effects Module](resonova/effects.py)** - Pre-built effect chains and customization
- **[DSP Module](resonova/dsp.py)** - All available DSP effects and plugin system

---

---

*Star ⭐ if ResoNova tightens your low-end and keeps your highs crispy.*

**Now with professional-grade effects, analysis tools, and processing chains!**
