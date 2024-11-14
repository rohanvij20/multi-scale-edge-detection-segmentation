#!/usr/bin/python3
import shutil
from pathlib import Path


def clear_results(results_dir: str = "results") -> None:
    """Clear all results directories while maintaining the structure."""
    results_path = Path(results_dir)

    # Define directories to clear
    directories = [
        results_path / "edges" / "sobel",
        results_path / "edges" / "canny",
        results_path / "segments" / "sobel",
        results_path / "segments" / "canny",
    ]

    # Remove and recreate each directory
    for dir_path in directories:
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Cleared all results in {results_dir}/")


if __name__ == "__main__":
    clear_results()
