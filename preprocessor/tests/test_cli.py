"""Tests for CLI functionality."""

import pytest
from src.cli import setup_argparse


class TestSetupArgparse:
    """Tests for setup_argparse function."""

    def test_parser_creation(self):
        parser = setup_argparse()
        assert parser is not None

    def test_required_argument(self):
        parser = setup_argparse()
        # Test that config is required
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_config_argument(self):
        parser = setup_argparse()
        args = parser.parse_args(['config.json'])
        assert args.config == 'config.json'
        assert args.log_level == 'INFO'  # default

    def test_log_level_argument(self):
        parser = setup_argparse()
        args = parser.parse_args(['config.json', '--log-level', 'DEBUG'])
        assert args.log_level == 'DEBUG'

    def test_debug_outputs_argument(self):
        parser = setup_argparse()
        args = parser.parse_args(['config.json', '--debug-outputs', 'fluid_only,geometry'])
        assert args.debug_outputs == 'fluid_only,geometry'

    def test_no_debug_flag(self):
        parser = setup_argparse()
        args = parser.parse_args(['config.json', '--no-debug'])
        assert args.no_debug is True

    def test_output_dir_argument(self):
        parser = setup_argparse()
        args = parser.parse_args(['config.json', '--output-dir', '/tmp/output'])
        assert args.output_dir == '/tmp/output'
