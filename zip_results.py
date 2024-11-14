#!/usr/bin/python3
import zipfile
from pathlib import Path
from datetime import datetime
import os


def zip_results(results_dir: str = "results", archive_dir: str = "archives") -> str:
    """
    Zip all results into a timestamped archive.

    Args:
        results_dir: Directory containing results to zip
        archive_dir: Directory to store the zip file

    Returns:
        Path to the created zip file
    """
    # Create archive directory if it doesn't exist
    archive_path = Path(archive_dir)
    archive_path.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for the zip file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"edge_detection_results_{timestamp}.zip"
    zip_path = archive_path / zip_filename

    # Count files to process
    results_path = Path(results_dir)
    total_files = sum(1 for _ in results_path.rglob("*") if _.is_file())
    processed_files = 0

    print(f"\nZipping results from {results_dir}/")
    print(f"Creating archive: {zip_filename}")
    print("Progress:")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Walk through all directories and files in results
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                file_path = Path(root) / file
                # Get path relative to results directory
                arc_path = file_path.relative_to(results_path)
                # Add file to zip
                zipf.write(file_path, arc_path)

                # Update progress
                processed_files += 1
                progress = (processed_files / total_files) * 100
                print(
                    f"\r[{'=' * int(progress/2)}{' ' * (50-int(progress/2))}] {progress:.1f}%",
                    end="",
                )

    # Get zip file size
    zip_size = zip_path.stat().st_size / (1024 * 1024)  # Convert to MB

    print(f"\n\nArchive created successfully!")
    print(f"Location: {zip_path}")
    print(f"Size: {zip_size:.2f} MB")
    print(f"Files archived: {processed_files}")

    return str(zip_path)


def list_archive_contents(zip_path: str):
    """List the contents of the created archive."""
    print("\nArchive contents:")
    print("=" * 50)

    with zipfile.ZipFile(zip_path, "r") as zipf:
        # Get list of files sorted by directory
        files = sorted(zipf.namelist())

        # Print directory tree
        current_dir = ""
        for file in files:
            # Split path into directories
            parts = Path(file).parts

            # Print directories
            for i, part in enumerate(parts[:-1]):
                if f"{'/'.join(parts[:i+1])}/" != current_dir:
                    print("  " * i + "├── " + part)
                    current_dir = f"{'/'.join(parts[:i+1])}/"

            # Print file
            print("  " * (len(parts) - 1) + "└── " + parts[-1])


if __name__ == "__main__":
    # Zip the results
    zip_path = zip_results()

    # List contents of created archive
    list_archive_contents(zip_path)
