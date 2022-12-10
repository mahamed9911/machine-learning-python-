from pathlib import Path
from typing import Union
import argparse
import cv2
import numpy
def is_cartoon(
    image: Union[str, Path],
    threshold: float = 4500,
    preview: bool = False,
) -> bool:
    # read and resize image
    img = cv2.imread(str(image))
    img = cv2.resize(img, (1024, 1024))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.medianBlur(gray, 3)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10
    )
    blurred_edges = cv2.adaptiveThreshold(
        blurred_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10
    )
    if preview:
        cv2.imshow("edges", edges)
        cv2.imshow("blurred edges", blurred_edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    count_1 = numpy.count_nonzero(edges)
    count_2 = numpy.count_nonzero(blurred_edges)
    return abs(count_2 - count_1) < threshold
def command_line_options():
    args = argparse.ArgumentParser(
        "blur_compare",
        description="Determine if a image is likely a cartoon or photo.",
    )
    args.add_argument(
        "-p",
        "--preview",
        action="store_true",
        help="Show the blurred image",
    )
    args.add_argument(
        "-t",
        "--threshold",
        type=float,
        help="Cutoff threshold",
        default=4500,
    )
    args.add_argument(
        "image",
        type=Path,
        help="Path to image file",
    )
    return vars(args.parse_args())
if __name__ == "__main__":
    options = command_line_options()
    if not options["image"].exists():
        raise FileNotFoundError(f"No image exists at {options['image'].absolute()}")
    if is_cartoon(**options):
        print(f"{options['image'].name} is a cartoon!")
    else:
        print(f"{options['image'].name} is a photo!")
