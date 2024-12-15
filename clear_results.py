#!/usr/bin/python3
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
            # Basic edge detection results
            "edges": ["sobel", "canny"],
            # Segmentation results
            "segments": ["sobel", "canny"],
            # Advanced edge detection results
            "advanced_edges": ["log", "multiscale"],
            # Advanced segmentation results
            "advanced_segments": ["watershed", "region_growing"],
            # Evaluation and comparison results
            "evaluation": ["metrics", "benchmarks"],
            # Visualization results
            "comparisons": [],
        }

        # Files to remove
        self.files_to_remove = [
            # Basic comparison results
            "edge_detection_comparison.csv",
            "summary_report.txt",
            "parameter_results.csv",
            # Advanced analysis results
            "advanced_edge_comparison.csv",
            "advanced_segmentation_comparison.csv",
            "evaluation_metrics.csv",
        ]

    def get_all_directories(self) -> List[Path]:
        """
        Get all directories that should be maintained.

        Returns:
            List of Path objects for all directories
        """
        directories = []
        for main_dir, subdirs in self.directory_structure.items():
            if subdirs:
                for subdir in subdirs:
                    directories.append(self.results_path / main_dir / subdir)
            else:
                directories.append(self.results_path / main_dir)
        return directories

    def clear_directories(self) -> Dict[str, int]:
        """
        Clear all result directories while maintaining structure.

        Returns:
            Dictionary with counts of cleared items
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

        # Print main directories
        for i, (main_dir, subdirs) in enumerate(self.directory_structure.items()):
            is_last_main = i == len(self.directory_structure) - 1
            prefix = "    └── " if is_last_main else "    ├── "
            print(f"{prefix}{main_dir}/")

            # Print subdirectories if any
            if subdirs:
                for j, subdir in enumerate(subdirs):
                    is_last_sub = j == len(subdirs) - 1
                    sub_prefix = "    │   " if not is_last_main else "        "
                    sub_prefix += "└── " if is_last_sub else "├── "
                    print(f"{sub_prefix}{subdir}/")


def clear_results(results_dir: str = "results") -> None:
    """
    Clear all results directories while maintaining the structure.

    Args:
        results_dir: Base directory for results
    """
    cleaner = ResultsCleaner(results_dir)

    try:
        # Clear directories and get statistics
        stats = cleaner.clear_directories()

        # Print summary
        print(f"\nCleared all results in {results_dir}/")
        print(f"Cleared {stats['directories']} directories and {stats['files']} files")

        # Print directory structure
        cleaner.print_directory_structure()

    except Exception as e:
        print(f"Error clearing results: {str(e)}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clear results directories while maintaining structure."
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

    # Temporarily redirect stdout if quiet mode is enabled
    if args.quiet:
        sys.stdout = open(os.devnull, "w")

    try:
        clear_results(args.results_dir)
    finally:
        if args.quiet:
            sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
