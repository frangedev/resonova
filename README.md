**ResoNova – Open-Source AI Mixing & Mastering for Electronic Music**

---

### 🎛️ One-liner  
ResoNova is a transparent, fully-customizable AI pipeline that balances, compresses and limits electronic tracks to industry loudness specs—powered by LibROSA & TensorFlow.

---

### 📌 Features  
| Stage | What it does | Knobs you can tweak |
|-------|--------------|---------------------|
| **Genre-Aware EQ** | Automatic tonal balancing trained on 20 k EDM/R&B/Techno tracks | Target curve, band weights |
| **Smart Compression** | Multi-band RMS & crest-factor control | Threshold ratio, attack/release ranges |
| **Transparent Limiting** | Look-ahead brickwall with LUFS-TruePeak guard | Ceiling, oversample factor |
| **Open DSP** | Replace any block with your own Python DSP | Plug-in API |

---

### 🚀 Quick Start  

```bash
git clone https://github.com/frangedev/resonova.git
cd resonova
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python cli.py --in my_track.wav --genre techno --lufs -14
```

The mastered WAV appears in `outputs/` alongside a JSON report of all moves the AI made.

---

### 🧪 Tech Stack  
- **Analysis**  LibROSA (STFT, chroma, MFCC)  
- **ML Engine** TensorFlow 2.x (Keras)  
- **DSP**   Pure Python + PyFFTW + custom Cython extensions  
- **Export**  32-bit float WAV, 24-bit WAV, or MP3 preview  

---

### 🎚️ Architecture Overview  

```
┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
│   Input    │→ │  Genre     │→ │  Neural    │→ │   Brick-   │
│   Track    │  │  Classifier│  │  EQ & DRC  │  │   wall     │
└────────────┘  └────────────┘  └────────────┘  └────────────┘
```

All stages expose a clean `process(buffer: np.ndarray, sr:int) -> np.ndarray` interface for unit-testing and swapping.

---

### 🛠️ Customization  

1. **Retrain**  
   Place new labelled stems in `data/raw/` and run:  
   `python train.py --task eq --epochs 50`

2. **Add a DSP module**  
   ```python
   from resonova.dsp import register
   @register('my_saturation')
   def my_sat(buffer, drive=2.0):
       return np.tanh(buffer * drive)
   ```

3. **CLI Flag**  
   `python cli.py ... --post my_saturation --drive 2.5`

---

### 📊 Benchmarks  
| Dataset | Baseline LUFS | ResoNova LUFS | Δ TruePeak | CPU RT* |
|---------|---------------|---------------|------------|---------|
| EDM-1k  | -9.3          | -9.0          | +0.2 dBTP  | 0.35×   |
| Techno-500 | -10.1      | -9.8          | +0.1 dBTP  | 0.29×   |

\* Real-time factor on Ryzen 7 5800X, 48 kHz / 512-block.

---

### 🤝 Contributing  
1. Fork & branch (`feature/my-idea`)  
2. `pytest tests/` must pass  
3. PR with clear diff + notebook demo → we’ll review within 48 h

---

### 📄 License  
MIT – see [LICENSE](LICENSE).  
Trained weights are CC-BY-SA 4.0. Commercial use allowed; attribution required.

---

### 🙋 FAQ  

**Q:** *Can I run it on GPU?*  
**A:** Training benefits from CUDA; inference is CPU-only for portability.

**Q:** *Does it work on vocals?*  
**A:** Currently optimized for full-mix electronic material; vocal stems coming in v0.5.

---

### 🧑‍💻 Maintainers  
- @makalin – lead DSP  
- @frangedev – ML & model ops  

---

*Star ⭐ if ResoNova tightens your low-end and keeps your highs crispy.*
