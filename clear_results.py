#!/usr/bin/python3
import shutil
from pathlib import Path


def clear_results(results_dir: str = "results") -> None:
    """Clear all results directories while maintaining the structure."""
    results_path = Path(results_dir)

    # Define all directories to maintain
    directories = [
        # Original structure
        results_path / "edges" / "sobel",
        results_path / "edges" / "canny",
        results_path / "segments" / "sobel",
        results_path / "segments" / "canny",
        # Comparison structure
        results_path / "comparisons",
    ]

    # Files to remove
    files_to_remove = [
        results_path / "edge_detection_comparison.csv",
        results_path / "summary_report.txt",
        results_path / "parameter_results.csv",
    ]

    # Remove and recreate directories
    for dir_path in directories:
        if dir_path.exists():
            print(f"Clearing directory: {dir_path}")
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

    # Remove result files
    for file_path in files_to_remove:
        if file_path.exists():
            print(f"Removing file: {file_path}")
            file_path.unlink()

    print(f"\nCleared all results in {results_dir}/")
    print("\nRecreated directory structure:")
    print("└── results/")
    print("    ├── edges/")
    print("    │   ├── sobel/")
    print("    │   └── canny/")
    print("    ├── segments/")
    print("    │   ├── sobel/")
    print("    │   └── canny/")
    print("    └── comparisons/")


if __name__ == "__main__":
    clear_results()
