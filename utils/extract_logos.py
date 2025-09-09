#!/usr/bin/env python3
"""
Extract rectangular regions from a PDF page and save each as a sub-image at a chosen DPI.

Dependencies:
  pip install pymupdf opencv-python pillow numpy

Example:
  python extract_rect_regions.py --pdf first_page_reference.pdf --page 0 --dpi 300 --out out_boxes
"""
import argparse
import os
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import cv2
from pydantic import BaseModel
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from state import State

class ExtractLogos():
    def __init__(self, pdf_path: str, page_number: int, dpi: int, out_dir: str, min_area_ratio: float = 0.002, approx_eps_ratio: float = 0.02, pad: int = 4, state: State = None):
        self.pdf_path = pdf_path
        self.page_number = page_number
        self.dpi = dpi
        self.out_dir = out_dir
        self.min_area_ratio = min_area_ratio
        self.approx_eps_ratio = approx_eps_ratio
        self.pad = pad
        self.img = None
        self.boxes = None
        self.state = state

    def extract_logos(self):
        self.img = self.render_page()
        self.boxes = self.find_rect_boxes()
        self.save_crops()
        if self.state:
            self.state.artwork_slices_folder = self.out_dir
            return self.state
        else:
            return None

    def render_page(self):
        """Render a single PDF page to an RGB numpy array at the given DPI."""
        doc = fitz.open(self.pdf_path)
        if self.page_number < 0 or self.page_number >= len(doc):
            raise IndexError(f"Page {self.page_number} out of range (0..{len(doc)-1})")
        page = doc[self.page_number]

        zoom = self.dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert to HxWxC RGB array
        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        img = arr.reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img = img[:, :, :3]  # drop alpha if present
        # PyMuPDF returns RGB already
        return img


    def find_rect_boxes(self):
        """
        Detect rectangular boxes (black strokes on gray background).
        Returns list of bounding boxes (x, y, w, h).
        """
        h, w = self.img.shape[:2]
        img_area = w * h

        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)

        # Invert-binarize so dark strokes become white on black (robust to gray bg).
        # Otsu picks a threshold automatically.
        _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Close small gaps in rectangle borders
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find external contours (each rectangle appears as a large contour)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area_ratio * img_area:
                continue

            # Polygonal approximation to check "rectangleness"
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, self.approx_eps_ratio * peri, True)
            if len(approx) != 4:
                # Not a clean quadrilateral; skip (feel free to relax if needed)
                continue

            x, y, bw, bh = cv2.boundingRect(approx)
            # Filter out super-thin shapes (likely lines)
            if min(bw, bh) < 10:
                continue

            boxes.append((x, y, bw, bh))

        # Sort left-to-right, then top-to-bottom for consistent naming
        boxes.sort(key=lambda b: (b[1] // 10, b[0]))
        return boxes


    def save_crops(self):
        """Crop and save each bounding box as its own file with the requested DPI."""
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        h, w = self.img.shape[:2]

        for i, (x, y, bw, bh) in enumerate(self.boxes, start=1):
            x0 = max(0, x - self.pad)
            y0 = max(0, y - self.pad)
            x1 = min(w, x + bw + self.pad)
            y1 = min(h, y + bh + self.pad)
            crop = self.img[y0:y1, x0:x1]

            im = Image.fromarray(crop)  # still RGB
            out_path = os.path.join(self.out_dir, f"box_{i:02d}.png")
            im.save(out_path, dpi=(self.dpi, self.dpi))  # store DPI metadata
            print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to PDF")
    ap.add_argument("--page", type=int, default=0, help="Zero-based page index")
    ap.add_argument("--dpi", type=int, default=300, help="Render DPI for detection & output")
    ap.add_argument("--out", default="out_boxes", help="Output directory")
    ap.add_argument("--min_area_ratio", type=float, default=0.002,
                    help="Min contour area as fraction of page (default 0.2%)")
    ap.add_argument("--approx_eps_ratio", type=float, default=0.02,
                    help="Polygon approximation epsilon as fraction of perimeter")
    ap.add_argument("--pad", type=int, default=4, help="Padding (pixels) around crops")
    args = ap.parse_args()
    extractor = ExtractLogos(args.pdf, args.page, args.dpi, args.out, args.min_area_ratio, args.approx_eps_ratio, args.pad)
    extractor.extract_logos()
    print("Logos extracted")


if __name__ == "__main__":
    main()
