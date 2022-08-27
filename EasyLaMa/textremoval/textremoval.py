import torch
import easyocr
import os
from ..utils import download_model, image_to_tensor, pad, load_image
from PIL import Image, ImageDraw
import numpy as np

class TextRemover:
    def __init__(self, languages=["en"], device="cuda", easyocr=True, lama=True):
        self.languages = languages
        self.device = torch.device(device)
        if easyocr:
            self.load_easyocr(languages, gpu=(self.device.type == "cuda"))
        if lama:
            self.load_lama()

    def load_lama(self):
        print("Loading LaMa model... ", end="")
        if os.environ.get("LAMA_MODEL"):
            self.lama_path = os.environ.get("LAMA_MODEL")
            if not os.path.exists(self.lama_path):
                raise FileNotFoundError(f"lama torchscript model not found: {self.lama_path}")
        else:
            self.lama_path = download_model()

        self.lama = torch.jit.load(self.lama_path, map_location="cpu")
        self.lama = self.lama.to(self.device)
        self.lama.eval()
        print("Done.")
    
    def load_easyocr(self, languages=["en"], gpu=True):
        print("Loading EasyOCR model... ", end="")
        self.languages = languages
        self.reader = easyocr.Reader(self.languages, gpu=gpu)
        print("Done.")

    def get_text(self, image):
        results = self.reader.readtext(np.array(image))
        results = {i : {"box": result[0], "text" : result[1], "conf" : result[2]} for i, result in enumerate(results)}
        return results

    def get_mask(self, image, results, mask_edge:int=1, radius=0):
        mask = Image.new(mode="L", size=image.size)
        draw = ImageDraw.Draw(mask)
        for result in results.values():
            box = result["box"][0] + result["box"][2]
            box = [coordinate-mask_edge for coordinate in box[:2]] + [coordinate+mask_edge for coordinate in box[2:]]
            draw.rounded_rectangle(xy=box, fill=255, outline=255, width=1, radius=radius)
        return mask
    
    @torch.inference_mode()
    def inpaint(self, image, mask):
        h, w = image.size

        image_tensor = image_to_tensor(image, device=self.device)
        mask_tensor= image_to_tensor(mask, device=self.device)

        padded_image = pad(image_tensor, mod=8, padding_mode="symmetric").unsqueeze(0)
        padded_mask = pad(mask_tensor, mod=8, padding_mode="constant").unsqueeze(0)

        inpainted = self.lama(padded_image, padded_mask)

        cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = cur_res[0:w, 0:h, :]
        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        cur_res = Image.fromarray(cur_res, mode="RGB")

        return cur_res
    
    def run(self, image, mask_edge=1, radius=0):
        image = load_image(image)
        print("Running EasyOCR... ", end="")
        ocr_results = self.get_text(image)
        print("Done.")
        mask = self.get_mask(image, ocr_results, mask_edge, radius)
        print("Running LaMa... ", end="")
        infilled = self.inpaint(image, mask)
        print("Done.")
        return infilled, mask

    def create_gif(self, image, outfile, include_mask=False, mask_edge=1, radius=0, **kwargs):
        image = load_image(image)
        frames = [image]
        result, mask = self.run(image, mask_edge, radius)
        if include_mask:
            frames.append(mask)
        frames.append(result)
        frames[0].save(outfile, save_all=True, append_images=frames[1:], **kwargs)
        
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)
    
    def __repr__(self):
        return f"EasyLaMa.TextRemover(languages={self.languages}, device={self.device})"
