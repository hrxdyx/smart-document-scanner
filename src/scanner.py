import argparse
import os
import sys

from src.preprocessing import preprocess_image
from src.edge_detection import detect_edges
from src.document_detection import (
    find_contours,
    find_document_corners,
    draw_document_contour
)
from src.perspective import four_point_transform
from src.enhancement import enhance_document
from src.utils import (
    load_image,
    save_image,
    ensure_directory
)


def create_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Smart Document Scanner using "
            "Computer Vision"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input document image"
    )

    parser.add_argument(
        "--output",
        default="output/scanned_document.jpg",
        help="Path for the final scanned image"
    )

    parser.add_argument(
        "--mode",
        choices=["color", "gray", "bw"],
        default="color",
        help="Output mode: color, gray, or bw"
    )

    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate processing stages"
    )

    return parser


def main():

    parser = create_parser()

    args = parser.parse_args()

    try:

        print("\n====================================")
        print("     SMART DOCUMENT SCANNER")
        print("====================================\n")

        # ---------------------------------
        # Step 1: Load image
        # ---------------------------------

        print("[1/7] Loading image...")

        image = load_image(args.input)

        # ---------------------------------
        # Step 2: Preprocessing
        # ---------------------------------

        print("[2/7] Preprocessing image...")

        resized, gray, blurred = preprocess_image(
            image
        )

        # ---------------------------------
        # Step 3: Edge detection
        # ---------------------------------

        print("[3/7] Detecting document edges...")

        edges = detect_edges(
            blurred
        )

        # ---------------------------------
        # Step 4: Find document
        # ---------------------------------

        print("[4/7] Detecting document boundary...")

        contours = find_contours(
            edges
        )

        corners = find_document_corners(
            contours
        )

        if corners is None:

            print(
                "\nERROR: No suitable document "
                "boundary was detected."
            )

            print(
                "Try using an image with a clear "
                "rectangular document."
            )

            return 1

        contour_image = draw_document_contour(
            resized,
            corners
        )

        # ---------------------------------
        # Step 5: Perspective correction
        # ---------------------------------

        print("[5/7] Correcting perspective...")

        warped = four_point_transform(
            resized,
            corners
        )

        # ---------------------------------
        # Step 6: Enhancement
        # ---------------------------------

        print("[6/7] Enhancing scanned document...")

        final_image = enhance_document(
            warped,
            args.mode
        )

        # ---------------------------------
        # Step 7: Save output
        # ---------------------------------

        print("[7/7] Saving output...")

        output_directory = os.path.dirname(
            args.output
        )

        if output_directory:
            ensure_directory(
                output_directory
            )

        save_image(
            args.output,
            final_image
        )

        # ---------------------------------
        # Save intermediate images
        # ---------------------------------

        if args.save_intermediate:

            base_directory = (
                output_directory
                if output_directory
                else "output"
            )

            ensure_directory(
                base_directory
            )

            save_image(
                os.path.join(
                    base_directory,
                    "01_original.jpg"
                ),
                resized
            )

            save_image(
                os.path.join(
                    base_directory,
                    "02_grayscale.jpg"
                ),
                gray
            )

            save_image(
                os.path.join(
                    base_directory,
                    "03_blurred.jpg"
                ),
                blurred
            )

            save_image(
                os.path.join(
                    base_directory,
                    "04_edges.jpg"
                ),
                edges
            )

            save_image(
                os.path.join(
                    base_directory,
                    "05_document_contour.jpg"
                ),
                contour_image
            )

            save_image(
                os.path.join(
                    base_directory,
                    "06_perspective_corrected.jpg"
                ),
                warped
            )

        print("\n====================================")
        print("          SCAN COMPLETE")
        print("====================================")
        print(f"Output: {args.output}")

        if args.save_intermediate:
            print(
                f"Intermediate files: "
                f"{output_directory or 'output'}/"
            )

        print()

        return 0

    except FileNotFoundError as error:

        print(f"\nERROR: {error}")
        return 1

    except ValueError as error:

        print(f"\nERROR: {error}")
        return 1

    except Exception as error:

        print(
            f"\nUnexpected error: {error}"
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())
