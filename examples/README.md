# ResoNova Examples

This directory contains examples and configurations for the enhanced ResoNova AI mixing and mastering system.

## 🆕 New Features Added

### 1. **Advanced DSP Effects**
- **Pitch & Time Manipulation**: `pitch_shift`, `time_stretch`
- **Harmonic Enhancement**: `harmonic_enhancer` with even/odd harmonic control
- **Sidechain Compression**: `sidechain_compressor` for ducking effects
- **Frequency Shifting**: `frequency_shifter` for creative effects
- **Ring Modulation**: `ring_modulator` for metallic sounds
- **Granular Synthesis**: `granular_synthesizer` for texture creation
- **Advanced Filtering**: `spectral_filter` with resonance control
- **Dynamic EQ**: `dynamic_eq` that responds to audio level
- **Enhanced Stereo**: `stereo_enhancer` with frequency-dependent processing
- **Transient Shaping**: `transient_shaper` for drum enhancement

### 2. **Pre-built Effect Chains**
- **Mastering Chains**: Genre-specific mastering for electronic, techno, house
- **Vocal Chains**: Style-specific vocal processing (pop, rap, electronic)
- **Drum Chains**: Genre-specific drum enhancement (electronic, trap, house)
- **Bass Chains**: Style-specific bass processing (electronic, sub, mid)

### 3. **Enhanced Audio Analysis**
- **Beat Detection**: Multiple algorithms (librosa, onset, spectral)
- **Feature Extraction**: MFCC, chroma, spectral features for ML
- **Noise Removal**: Multiple denoising methods
- **Audio Enhancement**: Brightness, warmth, clarity, presence
- **Quality Scoring**: Comprehensive audio quality metrics (0-100)

### 4. **Advanced Visualization**
- **12-panel Analysis**: Comprehensive audio analysis views
- **Comparison Tools**: Side-by-side audio comparison
- **Beat Visualization**: Tempo and beat detection display
- **Feature Visualization**: MFCC, chroma, and spectral analysis

## 🚀 Quick Start Examples

### Basic Mastering
```bash
# Master a track with automatic genre detection
python cli.py master -i my_track.wav -l -14

# Master with specific genre and post-processing
python cli.py master -i my_track.wav -g techno -l -12 -p saturation -p exciter
```

### Using Effect Chains
```bash
# Apply preset mastering chain
python cli.py process -i my_track.wav -c techno_mastering

# Apply vocal processing chain
python cli.py process -i vocals.wav -c pop_vocal

# Apply custom effect chain
python cli.py process -i my_track.wav -c custom_edm_mastering --custom --config custom_chain_config.json
```

### Audio Analysis
```bash
# Basic analysis
python cli.py analyze -i my_track.wav

# Full feature extraction
python cli.py analyze -i my_track.wav -f full -o analysis.json

# Create visualizations
python cli.py visualize -i my_track.wav -t advanced -o analysis.png

# Compare two audio files
python cli.py visualize -i original.wav -t comparison --compare processed.wav
```

### Individual Effects
```bash
# Apply single effect
python cli.py apply-effect -i my_track.wav -e saturation -p drive=2.0

# Apply with multiple parameters
python cli.py apply-effect -i my_track.wav -e multiband_compressor \
  -p low_threshold=0.3 -p mid_threshold=0.2 -p high_threshold=0.1
```

### Audio Enhancement
```bash
# Enhance brightness
python cli.py enhance -i my_track.wav -t brightness -a 0.6

# Remove noise
python cli.py denoise -i noisy_audio.wav -m spectral_gate -t 0.2
```

## 📁 Configuration Files

### Custom Effect Chain
The `custom_chain_config.json` shows how to create custom effect chains:

```json
{
  "name": "custom_edm_mastering",
  "description": "Custom EDM mastering chain",
  "effects": [
    {
      "effect": "spectral_filter",
      "params": {
        "filter_type": "highpass",
        "low_freq": 30,
        "high_freq": 18000
      }
    },
    {
      "effect": "multiband_compressor",
      "params": {
        "low_threshold": 0.5,
        "mid_threshold": 0.4,
        "high_threshold": 0.3
      }
    }
  ]
}
```

## 🔧 Available Effects

### Core DSP Effects
- `saturation` - Soft saturation with drive control
- `exciter` - Harmonic excitation for brightness
- `stereo_widener` - Basic stereo enhancement
- `multiband_compressor` - Multi-band compression
- `reverb` - Simple delay-based reverb
- `chorus` - Chorus effect with LFO modulation
- `bit_crusher` - Bit depth and sample rate reduction
- `distortion` - Multiple distortion types (hard, soft, tube)

### Advanced Effects
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

## 🎯 Effect Chains

### Mastering Chains
- `electronic_mastering` - General electronic music
- `techno_mastering` - Techno-specific processing
- `house_mastering` - House music optimization

### Vocal Chains
- `pop_vocal` - Pop vocal enhancement
- `rap_vocal` - Rap vocal processing
- `electronic_vocal` - Electronic vocal effects

### Drum Chains
- `electronic_drums` - Electronic drum enhancement
- `trap_drums` - Trap drum processing
- `house_drums` - House drum optimization

### Bass Chains
- `electronic_bass` - Electronic bass processing
- `sub_bass` - Sub-bass enhancement
- `mid_bass` - Mid-range bass processing

## 📊 Analysis Features

### Beat Detection Methods
- `librosa` - LibROSA beat tracking
- `onset` - Onset-based detection
- `spectral` - Spectral flux analysis

### Feature Extraction Levels
- `basic` - MFCC features only
- `mfcc` - MFCC + chroma features
- `chroma` - Chroma + spectral features
- `full` - Complete feature set

### Quality Metrics
- Dynamic range scoring
- Signal-to-noise ratio
- Spectral characteristics
- Harmonic content analysis
- Overall quality score (0-100)

## 🎨 Visualization Types

### Basic Visualization
- Waveform display
- Frequency spectrum
- Mel-spectrogram
- Loudness over time
- Phase spectrum
- Audio statistics

### Advanced Visualization
- All basic features plus:
- Beat detection display
- MFCC feature visualization
- Chroma feature analysis
- Spectral centroid tracking
- Zero crossing rate
- RMS energy over time

### Comparison Visualization
- Side-by-side waveform comparison
- Frequency spectrum comparison
- Loudness metrics comparison
- Difference analysis

## 💡 Tips and Best Practices

1. **Start with Presets**: Use preset chains before creating custom ones
2. **Parameter Ranges**: Most parameters are normalized (0.0-1.0)
3. **Quality vs Speed**: Higher quality effects may be slower
4. **Chain Order**: Effects are applied in sequence - order matters
5. **Monitoring**: Use visualization tools to monitor processing results
6. **Backup**: Always keep original files when experimenting

## 🔍 Troubleshooting

### Common Issues
- **Effect not found**: Use `python cli.py list-effects` to see available effects
- **Chain not found**: Use `python cli.py list-chains` to see available chains
- **Parameter errors**: Check parameter ranges and types
- **Memory issues**: Large files may require more RAM

### Performance Tips
- Use lower quality settings for real-time processing
- Process shorter audio segments for testing
- Close other applications to free up resources
- Use SSD storage for faster I/O

## 📚 Further Reading

- Check the main README.md for project overview
- Review the source code for implementation details
- Explore the test suite for usage examples
- Join the community for support and feedback
