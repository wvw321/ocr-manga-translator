import argparse
from EasyLaMa import TextRemover
from .util import load_image
from .util import load_images
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="Images to process. Required")
    parser.add_argument("-e", "--edge", type=int, default=1, help="Extra margin at the edges of detected boxes. Default: 1")
    parser.add_argument("-r", "--radius", type=int, default=1, help="Radius for rounded corners. 0 = no rounding. Default: 1")
    parser.add_argument("-o", "--output", default=".", help="Output folder. Default: . (current dir)")
    parser.add_argument("-of","--output_format", default=None, help="Output format (jpg, png...). Default: same as input")
    parser.add_argument("-s", "--suffix", default="_result", help="Suffix for results. Default: _result")
    parser.add_argument("-m", "--mask_suffix", default=None, help="Suffix for storing mask. Default: don't store mask")
    parser.add_argument("-c", "--copy", action="store_true", help="Copy original image to output folder.")
    parser.add_argument("-d", "--device", default="cuda", help="Device to use (cuda, cuda:0, cpu...). Default: cuda")
    parser.add_argument("-l", "--languages", type=str, nargs="+", default=["en"], help="Languages to detect. See https://www.jaided.ai/easyocr/ for supported languages and their abbreviations. Default: en")
    return parser.parse_args()

def cli():
    args = get_args()
    image_names = [os.path.splitext(os.path.basename(image)) for image in args.images]
    images = load_images(args.images)
    tr = TextRemover(languages=args.languages, device=args.device)
    os.makedirs(args.output, exist_ok=True)
    for (image, (name, ext)) in zip(images, image_names):
        result_name = os.path.join(args.output, name + args.suffix + (args.output_format or ext))
        mask_name = os.path.join(args.output, name + (args.mask_suffix or "") + (args.output_format or ext))
        image_name = os.path.join(args.output, name + (args.output_format or ext))

        result, mask = tr(image, mask_edge=args.edge, radius=args.radius)

        result.save(result_name)
        if args.mask_suffix:
            mask.save(mask_name)
        if args.copy:
            image.save(image_name)
        print("result: {}{}".format(result_name, ("\nmask: " + mask_name) if args.mask_suffix else ""))
