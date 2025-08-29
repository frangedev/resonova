#!/usr/bin/env python3
"""
ResoNova Command Line Interface

A comprehensive CLI for the ResoNova AI mixing and mastering system.
"""

import click
import numpy as np
from pathlib import Path
import json
import sys
from typing import Optional

from resonova import (
    ResoNova, EffectChain, MasteringChain, VocalChain, DrumChain, BassChain,
    create_custom_chain, get_preset_chains, analyze_chain_effect,
    load_audio, save_audio, analyze_loudness, analyze_spectrum,
    detect_beats, extract_audio_features, remove_noise, enhance_audio,
    create_advanced_visualization, create_audio_comparison, analyze_audio_quality,
    list_dsp_modules, get_dsp_module
)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """ResoNova - AI Mixing & Mastering for Electronic Music"""
    pass


@cli.command()
@click.option('--input', '-i', required=True, help='Input audio file path')
@click.option('--output', '-o', help='Output file path (auto-generated if not specified)')
@click.option('--genre', '-g', help='Target genre (auto-detected if not specified)')
@click.option('--lufs', '-l', default=-14.0, help='Target LUFS level')
@click.option('--post', '-p', multiple=True, help='Post-processing DSP modules')
@click.option('--config', '-c', help='Configuration file path')
def master(input, output, genre, lufs, post, config):
    """Master an audio file using the ResoNova AI pipeline."""
    try:
        # Load configuration
        config_dict = {}
        if config:
            with open(config, 'r') as f:
                config_dict = json.load(f)
        
        # Update config with CLI options
        config_dict.update({
            'target_lufs': lufs,
            'post_processing': list(post)
        })
        
        # Initialize ResoNova
        resonova = ResoNova(config_dict)
        
        # Process file
        click.echo(f"Processing {input}...")
        results = resonova.process_file(input, output, genre)
        
        click.echo(f"✅ Mastered audio saved to: {results['output_file']}")
        click.echo(f"📊 Processing time: {results['processing_time']:.2f}s")
        click.echo(f"🎵 Detected genre: {results['processing_info'].get('detected_genre', 'Unknown')}")
        
        # Display loudness analysis
        loudness = results['loudness_analysis']
        click.echo(f"🔊 Final LUFS: {loudness['integrated_lufs']:.1f}")
        click.echo(f"📈 Peak: {loudness['peak_db']:.1f} dB")
        click.echo(f"📊 Crest Factor: {loudness['crest_factor']:.2f}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--input', '-i', required=True, help='Input audio file path')
@click.option('--output', '-o', help='Output file path')
@click.option('--chain', '-c', required=True, help='Effect chain to apply')
@click.option('--custom', is_flag=True, help='Use custom chain configuration')
@click.option('--config', help='Custom chain configuration file')
def process(input, output, chain, custom, config):
    """Process audio using pre-built or custom effect chains."""
    try:
        # Load audio
        audio, sample_rate = load_audio(input)
        click.echo(f"📁 Loaded audio: {len(audio)/sample_rate:.2f}s, {sample_rate}Hz")
        
        # Get effect chain
        if custom:
            if not config:
                click.echo("❌ Custom chain requires --config file", err=True)
                sys.exit(1)
            
            with open(config, 'r') as f:
                chain_config = json.load(f)
            
            effect_chain = create_custom_chain(
                chain_config['name'], 
                chain_config['effects']
            )
            click.echo(f"🔧 Using custom chain: {chain_config['name']}")
        else:
            preset_chains = get_preset_chains()
            if chain not in preset_chains:
                click.echo(f"❌ Unknown preset chain: {chain}", err=True)
                click.echo(f"Available chains: {', '.join(preset_chains.keys())}")
                sys.exit(1)
            
            effect_chain = preset_chains[chain]
            click.echo(f"🔧 Using preset chain: {chain}")
        
        # Process audio
        click.echo("⚡ Processing audio...")
        processed_audio = effect_chain.process(audio, sample_rate)
        
        # Save output
        if not output:
            input_path = Path(input)
            output = f"{input_path.stem}_{chain}.wav"
        
        save_audio(output, processed_audio, sample_rate)
        click.echo(f"✅ Processed audio saved to: {output}")
        
        # Analyze results
        analysis = analyze_chain_effect(audio, sample_rate, effect_chain)
        click.echo(f"📊 Chain applied: {analysis['chain_name']}")
        click.echo(f"🔊 LUFS change: {analysis['differences']['loudness']['integrated_lufs']:+.1f}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--input', '-i', required=True, help='Input audio file path')
@click.option('--output', '-o', help='Output visualization file')
@click.option('--type', '-t', default='advanced', 
              type=click.Choice(['basic', 'advanced', 'comparison']),
              help='Visualization type')
@click.option('--compare', help='Audio file to compare against')
def visualize(input, output, type, compare):
    """Create audio visualizations and analysis."""
    try:
        # Load audio
        audio, sample_rate = load_audio(input)
        click.echo(f"📁 Loaded audio: {len(audio)/sample_rate:.2f}s, {sample_rate}Hz")
        
        if type == 'basic':
            from resonova.utils import create_audio_visualization
            create_audio_visualization(audio, sample_rate, output, "Basic Audio Analysis")
            click.echo("📊 Basic visualization created")
            
        elif type == 'advanced':
            create_advanced_visualization(audio, sample_rate, output, "Advanced Audio Analysis")
            click.echo("📊 Advanced visualization created")
            
        elif type == 'comparison':
            if not compare:
                click.echo("❌ Comparison visualization requires --compare file", err=True)
                sys.exit(1)
            
            compare_audio, compare_sr = load_audio(compare)
            if compare_sr != sample_rate:
                click.echo("⚠️  Sample rates differ, resampling comparison audio...")
                from librosa import resample
                compare_audio = resample(compare_audio, orig_sr=compare_sr, target_sr=sample_rate)
            
            create_audio_comparison(audio, compare_audio, sample_rate, 
                                  ('Original', 'Comparison'), output)
            click.echo("📊 Comparison visualization created")
        
        if output:
            click.echo(f"💾 Visualization saved to: {output}")
        else:
            click.echo("👁️  Visualization displayed")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--input', '-i', required=True, help='Input audio file path')
@click.option('--output', '-o', help='Output analysis file')
@click.option('--features', '-f', default='basic',
              type=click.Choice(['basic', 'mfcc', 'chroma', 'full']),
              help='Feature extraction level')
def analyze(input, output, features):
    """Analyze audio and extract features."""
    try:
        # Load audio
        audio, sample_rate = load_audio(input)
        click.echo(f"📁 Loaded audio: {len(audio)/sample_rate:.2f}s, {sample_rate}Hz")
        
        # Extract features
        click.echo(f"🔍 Extracting {features} features...")
        extracted_features = extract_audio_features(audio, sample_rate, features)
        
        # Analyze loudness
        loudness = analyze_loudness(audio, sample_rate)
        
        # Detect beats
        try:
            beat_data = detect_beats(audio, sample_rate)
            click.echo(f"🎵 Tempo: {beat_data['tempo']:.1f} BPM")
        except:
            click.echo("⚠️  Beat detection failed")
        
        # Analyze quality
        quality = analyze_audio_quality(audio, sample_rate)
        
        # Compile results
        results = {
            'file_info': {
                'path': input,
                'duration': len(audio) / sample_rate,
                'sample_rate': sample_rate
            },
            'loudness_analysis': loudness,
            'quality_analysis': quality,
            'extracted_features': {
                k: v.tolist() if isinstance(v, np.ndarray) else v 
                for k, v in extracted_features.items()
            }
        }
        
        if 'tempo' in locals():
            results['beat_analysis'] = beat_data
        
        # Display summary
        click.echo(f"🔊 RMS: {loudness['rms_db']:.1f} dB")
        click.echo(f"📈 Peak: {loudness['peak_db']:.1f} dB")
        click.echo(f"📊 LUFS: {loudness['integrated_lufs']:.1f}")
        click.echo(f"⭐ Quality Score: {quality['quality_score']:.1f}/100")
        
        # Save results
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            click.echo(f"💾 Analysis saved to: {output}")
        else:
            click.echo("📋 Analysis results displayed above")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--input', '-i', required=True, help='Input audio file path')
@click.option('--output', '-o', help='Output file path')
@click.option('--effect', '-e', required=True, help='DSP effect to apply')
@click.option('--params', '-p', multiple=True, help='Effect parameters (key=value)')
def apply_effect(input, output, effect, params):
    """Apply a single DSP effect to audio."""
    try:
        # Load audio
        audio, sample_rate = load_audio(input)
        click.echo(f"📁 Loaded audio: {len(audio)/sample_rate:.2f}s, {sample_rate}Hz")
        
        # Parse parameters
        effect_params = {}
        for param in params:
            if '=' in param:
                key, value = param.split('=', 1)
                try:
                    # Try to convert to float/int if possible
                    if '.' in value:
                        effect_params[key] = float(value)
                    else:
                        effect_params[key] = int(value)
                except ValueError:
                    effect_params[key] = value
            else:
                effect_params[key] = True
        
        # Get effect function
        effect_func = get_dsp_module(effect)
        if effect_func is None:
            click.echo(f"❌ Unknown effect: {effect}", err=True)
            click.echo("Available effects:")
            for effect_name in list_dsp_modules():
                click.echo(f"  - {effect_name}")
            sys.exit(1)
        
        # Apply effect
        click.echo(f"⚡ Applying {effect} with params: {effect_params}")
        processed_audio = effect_func(audio, sample_rate, **effect_params)
        
        # Save output
        if not output:
            input_path = Path(input)
            output = f"{input_path.stem}_{effect}.wav"
        
        save_audio(output, processed_audio, sample_rate)
        click.echo(f"✅ Processed audio saved to: {output}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def list_effects():
    """List all available DSP effects."""
    try:
        effects = list_dsp_modules()
        click.echo("🔧 Available DSP Effects:")
        click.echo("=" * 40)
        
        for effect in sorted(effects):
            effect_func = get_dsp_module(effect)
            if effect_func and effect_func.__doc__:
                # Extract first line of docstring
                doc = effect_func.__doc__.strip().split('\n')[0]
                click.echo(f"📌 {effect}: {doc}")
            else:
                click.echo(f"📌 {effect}")
        
        click.echo(f"\nTotal effects: {len(effects)}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def list_chains():
    """List all available preset effect chains."""
    try:
        chains = get_preset_chains()
        click.echo("🔗 Available Preset Effect Chains:")
        click.echo("=" * 50)
        
        categories = {
            'Mastering': [k for k in chains.keys() if 'mastering' in k],
            'Vocals': [k for k in chains.keys() if 'vocal' in k],
            'Drums': [k for k in chains.keys() if 'drums' in k],
            'Bass': [k for k in chains.keys() if 'bass' in k]
        }
        
        for category, chain_list in categories.items():
            if chain_list:
                click.echo(f"\n🎯 {category}:")
                for chain in sorted(chain_list):
                    chain_obj = chains[chain]
                    click.echo(f"  📌 {chain} ({chain_obj.get_info()['total_effects']} effects)")
        
        click.echo(f"\nTotal chains: {len(chains)}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--input', '-i', required=True, help='Input audio file path')
@click.option('--output', '-o', help='Output file path')
@click.option('--type', '-t', default='brightness',
              type=click.Choice(['brightness', 'warmth', 'clarity', 'presence']),
              help='Enhancement type')
@click.option('--amount', '-a', default=0.5, type=float, help='Enhancement amount (0.0-1.0)')
def enhance(input, output, type, amount):
    """Enhance audio using various enhancement techniques."""
    try:
        # Load audio
        audio, sample_rate = load_audio(input)
        click.echo(f"📁 Loaded audio: {len(audio)/sample_rate:.2f}s, {sample_rate}Hz")
        
        # Apply enhancement
        click.echo(f"✨ Applying {type} enhancement (amount: {amount})...")
        enhanced_audio = enhance_audio(audio, sample_rate, type, amount)
        
        # Save output
        if not output:
            input_path = Path(input)
            output = f"{input_path.stem}_{type}_enhanced.wav"
        
        save_audio(output, enhanced_audio, sample_rate)
        click.echo(f"✅ Enhanced audio saved to: {output}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--input', '-i', required=True, help='Input audio file path')
@click.option('--output', '-o', help='Output file path')
@click.option('--method', '-m', default='spectral_gate',
              type=click.Choice(['spectral_gate', 'wiener', 'spectral_subtraction']),
              help='Noise removal method')
@click.option('--threshold', '-t', default=0.1, type=float, help='Noise threshold')
def denoise(input, output, method, threshold):
    """Remove noise from audio."""
    try:
        # Load audio
        audio, sample_rate = load_audio(input)
        click.echo(f"📁 Loaded audio: {len(audio)/sample_rate:.2f}s, {sample_rate}Hz")
        
        # Remove noise
        click.echo(f"🧹 Removing noise using {method} method (threshold: {threshold})...")
        denoised_audio = remove_noise(audio, sample_rate, method, threshold)
        
        # Save output
        if not output:
            input_path = Path(input)
            output = f"{input_path.stem}_denoised.wav"
        
        save_audio(output, denoised_audio, sample_rate)
        click.echo(f"✅ Denoised audio saved to: {output}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
