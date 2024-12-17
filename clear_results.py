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
            "edges": ["sobel", "canny"],
            "segments": ["sobel", "canny"],
            "advanced_edges": ["log", "multiscale"],
            "advanced_segments": ["watershed", "region_growing"],
            "evaluation": ["metrics", "benchmarks"],
            "comparisons": [],
        }

        # Files to remove
        self.files_to_remove = [
            "edge_detection_comparison.csv",
            "summary_report.txt",
            "parameter_results.csv",
            "advanced_edge_comparison.csv",
            "advanced_segmentation_comparison.csv",
            "evaluation_metrics.csv",
        ]

    def get_all_directories(self) -> List[Path]:
        """Get all directories that should be maintained."""
        directories = []
        for main_dir, subdirs in self.directory_structure.items():
            main_path = self.results_path / main_dir
            if subdirs:
                for subdir in subdirs:
                    directories.append(main_path / subdir)
            else:
                directories.append(main_path)
        return directories

    def clear_results(self) -> Dict[str, int]:
        """Clear all results directories and files."""
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

        for i, (main_dir, subdirs) in enumerate(self.directory_structure.items()):
            is_last_main = i == len(self.directory_structure) - 1
            prefix = "    └── " if is_last_main else "    ├── "
            print(f"{prefix}{main_dir}/")
            if subdirs:
                for j, subdir in enumerate(subdirs):
                    is_last_sub = j == len(subdirs) - 1
                    sub_prefix = "    │   " if not is_last_main else "        "
                    sub_prefix += "└── " if is_last_sub else "├── "
                    print(f"{sub_prefix}{subdir}/")


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
    args = parser.parse_args()

    cleaner = ResultsCleaner(args.results_dir)
    stats = cleaner.clear_results()

    print(f"\nCleared {stats['directories']} directories and {stats['files']} files.")
    cleaner.print_directory_structure()


if __name__ == "__main__":
    main()
