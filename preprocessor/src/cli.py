"""Command-line interface for HemoFlow preprocessor."""

import sys
import argparse
import logging

from src.config import PreprocessorConfig
from src.io_utils import setup_logging
from src.pipeline import run_preprocessing_pipeline


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command-line argument parser.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description='HemoFlow geometry preprocessor - voxelizes vessel geometry and prepares simulation input',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  preprocessor config.json
  preprocessor config.json --log-level DEBUG
  preprocessor config.json --debug-outputs fluid_only,geometry
  preprocessor config.json --no-debug --output-dir output/
        """
    )

    parser.add_argument(
        'config',
        help='Path to JSON configuration file'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )

    parser.add_argument(
        '--debug-outputs',
        help='Comma-separated list of debug outputs to generate: '
             'fluid_only, wall_fluid, geometry, coil, stent_final, stent_linear, stent_quadratic'
    )

    parser.add_argument(
        '--no-debug',
        action='store_true',
        help='Disable all debug outputs'
    )

    parser.add_argument(
        '--output-dir',
        help='Override output directory from config file'
    )

    return parser


def main() -> None:
    """Main CLI entry point."""
    # Parse command-line arguments
    parser = setup_argparse()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Load configuration with CLI overrides
    cli_overrides = {
        'output_dir': args.output_dir,
        'debug_outputs': args.debug_outputs,
        'no_debug': args.no_debug,
    }

    try:
        config = PreprocessorConfig.load_from_json(args.config, cli_overrides)
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Configuration error: {e}")
        sys.exit(1)

    # Run preprocessing pipeline
    try:
        run_preprocessing_pipeline(config)
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
