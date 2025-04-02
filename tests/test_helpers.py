from unittest.mock import patch

import pytest

from multiversum.helpers import calculate_cpu_count


class TestCalculateCPUCount:
    def test_zero_raises_assertion_error(self):
        """Test that n_jobs=0 raises an AssertionError."""
        with pytest.raises(AssertionError):
            calculate_cpu_count(0)

    @patch("os.cpu_count", return_value=8)
    def test_positive_values(self, mock_cpu_count):
        """Test that positive n_jobs values are used directly."""
        assert calculate_cpu_count(1) == 1
        assert calculate_cpu_count(4) == 4
        assert calculate_cpu_count(8) == 8

    @patch("os.cpu_count", return_value=8)
    def test_too_large_positive_values(self, mock_cpu_count):
        """Test that positive n_jobs values are capped at cpu_count."""
        assert calculate_cpu_count(10) == 8
        assert calculate_cpu_count(16) == 8

    @patch("os.cpu_count", return_value=8)
    def test_negative_one(self, mock_cpu_count):
        """Test that n_jobs=-1 uses all available CPUs."""
        assert calculate_cpu_count(-1) == 8

    @patch("os.cpu_count", return_value=8)
    def test_negative_two(self, mock_cpu_count):
        """Test that n_jobs=-2 uses all but one CPU."""
        assert calculate_cpu_count(-2) == 7

    @patch("os.cpu_count", return_value=8)
    def test_other_negative_values(self, mock_cpu_count):
        """Test other negative n_jobs values."""
        assert calculate_cpu_count(-3) == 6
        assert calculate_cpu_count(-4) == 5

    @patch("os.cpu_count", return_value=8)
    def test_negative_values_minimum_one(self, mock_cpu_count):
        """Test that negative n_jobs values always result in at least 1 CPU."""
        assert calculate_cpu_count(-8) == 1
        assert calculate_cpu_count(-9) == 1
        assert calculate_cpu_count(-100) == 1

    @patch("os.cpu_count", return_value=None)
    def test_cpu_count_none(self, mock_cpu_count):
        """Test behavior when os.cpu_count() returns None."""
        assert calculate_cpu_count(1) == 1
        assert calculate_cpu_count(-1) == 1
        assert calculate_cpu_count(-2) == 1

    @patch("os.cpu_count", return_value=1)
    def test_single_cpu_system(self, mock_cpu_count):
        """Test behavior on a single-CPU system."""
        assert calculate_cpu_count(1) == 1
        assert calculate_cpu_count(-1) == 1
        assert calculate_cpu_count(-2) == 1
