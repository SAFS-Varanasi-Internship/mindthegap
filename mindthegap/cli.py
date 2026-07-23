#!/usr/bin/env python
"""
Command-line interface for mindthegap model bundle management.

Usage examples:

    # Save a model bundle (typically done in a training script)
    python -m mindthegap.cli save \\
        --model-path models/temp_model.keras \\
        --stats-path models/temp_stats.json \\
        --bundle-path models/arabsea_2015 \\
        --metadata region="Arabian Sea" train_year=2015

    # Inspect a bundle
    python -m mindthegap.cli info models/arabsea_2015

    # List all bundles in a directory
    python -m mindthegap.cli list models/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def cmd_save(args):
    """Save a model bundle from separate files."""
    import tensorflow as tf
    from .model_bundle import save_model_bundle
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)
    
    # Load stats
    print(f"Loading stats from {args.stats_path}...")
    with open(args.stats_path) as f:
        stats = json.load(f)
    
    # Parse metadata from args
    metadata = {}
    if args.metadata:
        for item in args.metadata:
            if '=' in item:
                key, value = item.split('=', 1)
                # Try to parse as JSON, fall back to string
                try:
                    metadata[key] = json.loads(value)
                except json.JSONDecodeError:
                    metadata[key] = value
    
    # Load history if provided
    history = None
    if args.history_path:
        print(f"Loading history from {args.history_path}...")
        with open(args.history_path) as f:
            history = json.load(f)
    
    # Save bundle
    print(f"Saving bundle to {args.bundle_path}...")
    bundle = save_model_bundle(
        model=model,
        bundle_path=args.bundle_path,
        stats=stats,
        metadata=metadata,
        history=history,
        overwrite=args.overwrite,
    )
    
    print(f"✓ Bundle saved successfully!")
    print(bundle.info())


def cmd_info(args):
    """Show bundle information."""
    from .model_bundle import load_model_bundle
    
    print(f"Loading bundle from {args.bundle_path}...")
    bundle = load_model_bundle(args.bundle_path)
    print(bundle.info())


def cmd_list(args):
    """List bundles in a directory."""
    from .model_bundle import load_model_bundle
    
    base_path = Path(args.directory)
    if not base_path.exists():
        print(f"Error: Directory not found: {base_path}", file=sys.stderr)
        sys.exit(1)
    
    # Find all directories that look like bundles (have model.keras)
    bundles = []
    for path in base_path.rglob("model.keras"):
        bundle_dir = path.parent
        try:
            bundle = load_model_bundle(bundle_dir)
            bundles.append((bundle_dir, bundle))
        except Exception as e:
            print(f"Warning: Could not load {bundle_dir}: {e}", file=sys.stderr)
    
    if not bundles:
        print(f"No bundles found in {base_path}")
        return
    
    print(f"Found {len(bundles)} bundle(s) in {base_path}:")
    print()
    
    for bundle_path, bundle in sorted(bundles, key=lambda x: x[0]):
        rel_path = bundle_path.relative_to(base_path)
        print(f"📦 {rel_path}")
        
        # Show key metadata
        if bundle.metadata:
            if 'region' in bundle.metadata:
                print(f"   Region: {bundle.metadata['region']}")
            if 'train_year' in bundle.metadata:
                print(f"   Train year: {bundle.metadata['train_year']}")
            if 'model_params' in bundle.metadata:
                params = bundle.metadata['model_params']
                print(f"   Parameters: {params:,}")
        
        # Show final training metrics if available
        if bundle.history:
            if 'val_loss' in bundle.history:
                final_val = bundle.history['val_loss'][-1]
                print(f"   Final val_loss: {final_val:.6f}")
            elif 'loss' in bundle.history:
                final_loss = bundle.history['loss'][-1]
                print(f"   Final loss: {final_loss:.6f}")
        
        print()


def cmd_export_stats(args):
    """Export stats from a bundle to a JSON file."""
    from .model_bundle import load_model_bundle
    import numpy as np
    
    print(f"Loading bundle from {args.bundle_path}...")
    bundle = load_model_bundle(args.bundle_path)
    
    # Make stats JSON-serializable
    stats_out = {}
    for key, value in bundle.stats.items():
        if isinstance(value, np.ndarray):
            stats_out[key] = value.tolist()
        elif isinstance(value, dict):
            stats_out[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v 
                for k, v in value.items()
            }
        else:
            stats_out[key] = value
    
    with open(args.output, 'w') as f:
        json.dump(stats_out, f, indent=2)
    
    print(f"✓ Stats exported to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="mindthegap model bundle CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # save command
    save_parser = subparsers.add_parser(
        'save',
        help='Save a model bundle from separate files'
    )
    save_parser.add_argument(
        '--model-path',
        required=True,
        help='Path to the trained .keras model file'
    )
    save_parser.add_argument(
        '--stats-path',
        required=True,
        help='Path to the stats JSON file'
    )
    save_parser.add_argument(
        '--bundle-path',
        required=True,
        help='Output path for the bundle directory'
    )
    save_parser.add_argument(
        '--metadata',
        nargs='*',
        help='Metadata as key=value pairs (e.g., region="Arabian Sea" train_year=2015)'
    )
    save_parser.add_argument(
        '--history-path',
        help='Optional path to training history JSON'
    )
    save_parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing bundle if present'
    )
    
    # info command
    info_parser = subparsers.add_parser(
        'info',
        help='Show information about a bundle'
    )
    info_parser.add_argument(
        'bundle_path',
        help='Path to the bundle directory'
    )
    
    # list command
    list_parser = subparsers.add_parser(
        'list',
        help='List all bundles in a directory'
    )
    list_parser.add_argument(
        'directory',
        help='Directory to search for bundles'
    )
    
    # export-stats command
    export_parser = subparsers.add_parser(
        'export-stats',
        help='Export stats from a bundle to JSON'
    )
    export_parser.add_argument(
        'bundle_path',
        help='Path to the bundle directory'
    )
    export_parser.add_argument(
        '--output',
        default='stats.json',
        help='Output JSON file (default: stats.json)'
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'save':
            cmd_save(args)
        elif args.command == 'info':
            cmd_info(args)
        elif args.command == 'list':
            cmd_list(args)
        elif args.command == 'export-stats':
            cmd_export_stats(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
