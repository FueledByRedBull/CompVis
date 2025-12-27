"""CLI interface for facial expression analysis."""

import argparse
import sys
from pathlib import Path
import cv2

from emotion_analyzer import EmotionAnalyzer, format_analysis_report


# Supported image extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}


def load_image(image_path: str | Path):
    """Load an image from disk. Returns BGR numpy array or None."""
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return None

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return None

    return image


def get_image_files(directory: Path) -> list[Path]:
    """Get all supported image files from a directory."""
    image_files = []

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_files.append(file_path)

    return sorted(image_files)


def analyze_directory(directory: str, output_dir: str | None = None,
                      save_annotated: bool = False, verbose: bool = True) -> dict:
    """Analyze all images in a directory for facial expressions."""
    dir_path = Path(directory)

    if not dir_path.exists():
        print(f"Error: Directory not found: {directory}")
        sys.exit(1)

    if not dir_path.is_dir():
        print(f"Error: Not a directory: {directory}")
        sys.exit(1)

    # Get all image files
    image_files = get_image_files(dir_path)

    if not image_files:
        print(f"No images found in {directory}")
        print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(1)

    if verbose:
        print(f"\nFound {len(image_files)} image(s) to analyze.")
        print("Initializing emotion analyzer (this may take a moment)...\n")

    # Initialize analyzer
    analyzer = EmotionAnalyzer(use_ensemble=True)

    # Create output directory if needed
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if save_annotated:
            annotated_dir = output_path / "annotated"
            annotated_dir.mkdir(exist_ok=True)

    # Process each image
    all_results = {}

    for i, image_path in enumerate(image_files, 1):
        if verbose:
            print(f"Processing [{i}/{len(image_files)}]: {image_path.name}")

        # Load image
        image = load_image(image_path)
        if image is None:
            continue

        # Analyze emotions
        analyses = analyzer.analyze_image(image)

        # Store results
        all_results[str(image_path)] = {
            'filename': image_path.name,
            'faces_detected': len(analyses),
            'analyses': analyses
        }

        # Generate and print report
        report = format_analysis_report(analyses, image_path.name)

        if verbose:
            print(report)

        # Save results if output directory specified
        if output_dir:
            # Save text report
            report_file = output_path / f"{image_path.stem}_analysis.txt"
            report_file.write_text(report, encoding='utf-8')

            # Save annotated image if requested
            if save_annotated and analyses:
                annotated_image = analyzer.draw_emotions(image, analyses)
                annotated_file = annotated_dir / f"{image_path.stem}_annotated{image_path.suffix}"
                cv2.imwrite(str(annotated_file), annotated_image)

                if verbose:
                    print(f"  Saved annotated image: {annotated_file}")

    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        total_faces = sum(r['faces_detected'] for r in all_results.values())
        print(f"Images processed: {len(all_results)}")
        print(f"Total faces detected: {total_faces}")

        if output_dir:
            print(f"Results saved to: {output_dir}")

    return all_results


def analyze_single_image(image_path: str, save_annotated: bool = False) -> dict | None:
    """Analyze a single image for facial expressions."""
    path = Path(image_path)

    if not path.exists():
        print(f"Error: Image not found: {image_path}")
        return None

    print("Initializing emotion analyzer...")
    analyzer = EmotionAnalyzer(use_ensemble=True)

    print(f"Analyzing: {path.name}\n")

    image = load_image(path)
    if image is None:
        return None

    analyses = analyzer.analyze_image(image)

    report = format_analysis_report(analyses, path.name)
    print(report)

    if save_annotated and analyses:
        annotated_image = analyzer.draw_emotions(image, analyses)
        annotated_file = path.parent / f"{path.stem}_annotated{path.suffix}"
        cv2.imwrite(str(annotated_file), annotated_image)
        print(f"Saved annotated image: {annotated_file}")

    return {
        'filename': path.name,
        'faces_detected': len(analyses),
        'analyses': analyses
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze facial expressions in images')

    parser.add_argument(
        'path',
        help='Path to image directory (or single image with --single)'
    )

    parser.add_argument(
        '--output', '-o',
        help='Directory to save analysis reports'
    )

    parser.add_argument(
        '--save-annotated', '-a',
        action='store_true',
        help='Save annotated images with emotion labels'
    )

    parser.add_argument(
        '--single', '-s',
        action='store_true',
        help='Analyze a single image instead of a directory'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output verbosity'
    )

    args = parser.parse_args()

    if args.single:
        analyze_single_image(args.path, save_annotated=args.save_annotated)
    else:
        analyze_directory(
            args.path,
            output_dir=args.output,
            save_annotated=args.save_annotated,
            verbose=not args.quiet
        )


if __name__ == '__main__':
    main()
