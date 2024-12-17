#!/usr/bin/python3
import os
import shutil
from pathlib import Path
from typing import List, Dict
import sys
import argparse


class ResultsCleaner:
    """Handles clearing and recreating the results directory structure."""

    def __init__(self, results_dir: str = "results"):
        """
        Initialize the results cleaner.

        Args:
            results_dir: Base directory for results
        """
        self.results_path = Path(results_dir)

        # Define directory structure
        self.directory_structure = {
            # Basic edge detection
            "edges": ["sobel", "canny"],
            # Basic segmentation
            "segments": ["sobel", "canny"],
            # Advanced edge detection
            "advanced_edges": {
                "log": ["edges", "visualizations"],
                "multiscale": ["edges", "visualizations"],
            },
            # Advanced segmentation
            "advanced_segments": {
                "watershed": ["segments", "visualizations"],
                "region_growing": ["segments", "visualizations"],
            },
            # Evaluation and comparison
            "evaluation": ["metrics", "benchmarks"],
            "comparisons": [],
            # Method-specific results
            "method_comparison": ["edge_detection", "segmentation"],
        }

        # Files to remove
        self.files_to_remove = [
            # Basic comparison files
            "edge_detection_comparison.csv",
            "summary_report.txt",
            "parameter_results.csv",
            # Advanced comparison files
            "advanced_edge_comparison.csv",
            "advanced_segmentation_comparison.csv",
            # Method-specific results
            "method_comparison/edge_detection_metrics.csv",
            "method_comparison/segmentation_metrics.csv",
            # Evaluation files
            "evaluation/metrics/edge_metrics.csv",
            "evaluation/metrics/segmentation_metrics.csv",
            "evaluation/benchmarks/performance_metrics.csv",
            # Region growing results
            "advanced_segments/region_growing/results.csv",
            # Watershed results
            "advanced_segments/watershed/results.csv",
        ]

    def get_all_directories(self) -> List[Path]:
        """
        Get all directories that should be maintained.

        Returns:
            List of Path objects for all directories
        """
        directories = []

        def add_nested_dirs(base_path: Path, structure: dict) -> None:
            """Recursively add nested directories."""
            for key, value in structure.items():
                current_path = base_path / key
                if isinstance(value, list):
                    if value:  # If there are subdirectories
                        for subdir in value:
                            directories.append(current_path / subdir)
                    else:  # If it's an empty list, add the current path
                        directories.append(current_path)
                elif isinstance(value, dict):  # Handle nested structure
                    for subkey, subvalue in value.items():
                        nested_path = current_path / subkey
                        if isinstance(subvalue, list):
                            for subdir in subvalue:
                                directories.append(nested_path / subdir)
                        else:
                            add_nested_dirs(nested_path, subvalue)

        add_nested_dirs(self.results_path, self.directory_structure)
        return directories

    def clear_results(self) -> Dict[str, int]:
        """
        Clear all results directories and files.

        Returns:
            Dictionary containing counts of cleared items
        """
        stats = {"directories": 0, "files": 0}

        # Clear and recreate directories
        for dir_path in self.get_all_directories():
            try:
                if dir_path.exists():
                    print(f"Clearing directory: {dir_path}")
                    shutil.rmtree(dir_path)
                    stats["directories"] += 1
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(
                    f"Error processing directory {dir_path}: {str(e)}", file=sys.stderr
                )

        # Remove result files
        for filename in self.files_to_remove:
            file_path = self.results_path / filename
            try:
                if file_path.exists():
                    print(f"Removing file: {file_path}")
                    file_path.unlink()
                    stats["files"] += 1
            except Exception as e:
                print(f"Error removing file {file_path}: {str(e)}", file=sys.stderr)

        return stats

    def print_directory_structure(self) -> None:
        """Print the recreated directory structure."""
        print("\nRecreated directory structure:")
        print("└── results/")

        def print_nested_structure(
            structure: dict, prefix: str = "    ", last: bool = True
        ) -> None:
            """Recursively print nested directory structure."""
            items = list(structure.items())
            for i, (name, value) in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = prefix + ("└── " if is_last else "├── ")
                print(f"{current_prefix}{name}/")

                if isinstance(value, dict):
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    print_nested_structure(value, new_prefix)
                elif isinstance(value, list) and value:
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    for j, subdir in enumerate(value):
                        is_last_sub = j == len(value) - 1
                        sub_prefix = new_prefix + ("└── " if is_last_sub else "├── ")
                        print(f"{sub_prefix}{subdir}/")

        print_nested_structure(self.directory_structure)


def main():
    """Main function to execute the results cleaner."""
    parser = argparse.ArgumentParser(
        description="Clear and recreate results directories."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Base directory for results (default: results)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress output except for errors"
    )

    args = parser.parse_args()

    if args.quiet:
        sys.stdout = open(os.devnull, "w")

    try:
        cleaner = ResultsCleaner(args.results_dir)
        stats = cleaner.clear_results()
        print(
            f"\nCleared {stats['directories']} directories and {stats['files']} files."
        )
        cleaner.print_directory_structure()
    finally:
        if args.quiet:
            sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
