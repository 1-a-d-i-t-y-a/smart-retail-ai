"""Generate small synthetic sample datasets so every notebook runs end-to-end.

Run:  python -m data.generate_samples
Outputs:
  data/products/{shirts,shoes,bags}/img_*.jpg   — 224x224 solid-colour images
  data/faces/{alice,bob}/img_*.jpg              — synthetic Haar-detectable face
  data/reviews/reviews.csv                      — 60 labelled retail reviews
"""
from __future__ import annotations

import csv
import os
import random

import cv2
import numpy as np

random.seed(42)
np.random.seed(42)

ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------- products ----------
PRODUCT_CLASSES = {
    "shirts": (60, 120, 200),   # BGR blue
    "shoes":  (30, 30, 30),     # near-black
    "bags":   (40, 90, 60),     # olive green
}


def make_product_images(n_per_class: int = 6) -> None:
    for cls, color in PRODUCT_CLASSES.items():
        out = os.path.join(ROOT, "products", cls)
        os.makedirs(out, exist_ok=True)
        for i in range(n_per_class):
            img = np.full((224, 224, 3), color, dtype=np.uint8)
            # Add per-image noise + a shape so classes are visually distinct.
            noise = np.random.randint(-25, 25, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            if cls == "shirts":
                cv2.rectangle(img, (60, 40), (160, 180), (255, 255, 255), 3)
            elif cls == "shoes":
                cv2.ellipse(img, (112, 150), (80, 30), 0, 0, 360, (200, 200, 200), -1)
            else:  # bags
                cv2.rectangle(img, (50, 70), (170, 190), (20, 60, 40), -1)
                cv2.rectangle(img, (80, 40), (140, 80), (20, 60, 40), 3)
            cv2.imwrite(os.path.join(out, f"img_{i:02d}.jpg"), img)


# ---------- faces ----------
def _synth_face(seed: int) -> np.ndarray:
    """Draw a very simple face that OpenCV's Haar cascade can detect."""
    rng = np.random.default_rng(seed)
    img = np.full((300, 300, 3), 220, dtype=np.uint8)
    skin = tuple(int(x) for x in rng.integers(180, 230, 3))
    cv2.ellipse(img, (150, 160), (85, 110), 0, 0, 360, skin, -1)
    # eyes
    cv2.circle(img, (120, 140), 10, (30, 30, 30), -1)
    cv2.circle(img, (180, 140), 10, (30, 30, 30), -1)
    # nose + mouth
    cv2.line(img, (150, 155), (150, 185), (90, 60, 40), 2)
    cv2.ellipse(img, (150, 210), (25, 10), 0, 0, 180, (60, 30, 30), 2)
    return img


def make_face_images() -> None:
    for i, name in enumerate(["alice", "bob"]):
        out = os.path.join(ROOT, "faces", name)
        os.makedirs(out, exist_ok=True)
        for k in range(3):
            cv2.imwrite(os.path.join(out, f"img_{k}.jpg"), _synth_face(seed=i * 10 + k))


# ---------- reviews ----------
POSITIVE = [
    "Absolutely love this product, great quality!",
    "Fantastic value for money, will buy again.",
    "The material feels premium and fits perfectly.",
    "Excellent customer service and fast shipping.",
    "Exceeded my expectations, highly recommend.",
    "Beautiful design and very comfortable to wear.",
    "Best purchase I have made this year.",
    "Amazing colours and the size chart was accurate.",
    "Super fast delivery, item as described.",
    "Really happy with the fit and finish.",
    "Perfect gift, they loved it.",
    "Five stars, will shop here again.",
    "Great fabric, holds up well after washing.",
    "The shoes are stylish and very comfy.",
    "Quality is top-notch, worth every penny.",
    "Exactly what I wanted, thank you!",
    "Sturdy build and looks fantastic.",
    "Excellent stitching and beautiful colour.",
    "Very pleased, will recommend to friends.",
    "Lovely product, arrived earlier than expected.",
]
NEGATIVE = [
    "Terrible quality, ripped after one wear.",
    "Not as described, very disappointed.",
    "Waste of money, do not recommend.",
    "The size runs way too small.",
    "Bad stitching and cheap material.",
    "Customer service was unhelpful.",
    "Arrived damaged and refund is slow.",
    "Colour is completely different from photo.",
    "Fell apart within a week.",
    "Awful smell out of the package.",
    "Very poor fit, sending it back.",
    "Zipper broke on first use.",
    "Overpriced for what you get.",
    "Fabric feels like plastic.",
    "Rude support and long wait times.",
    "Bag strap snapped on day two.",
    "Faded after a single wash.",
    "Completely misleading product listing.",
    "Would give zero stars if I could.",
    "Absolutely not worth the money.",
]
NEUTRAL = [
    "It is okay, nothing special.",
    "Product matches the description.",
    "Average quality for the price.",
    "Fits as expected, no complaints.",
    "Delivery took the standard time.",
    "Colour is fine, not exactly as shown.",
    "Does the job, but nothing more.",
    "Mid-range item, decent value.",
    "It works, though I expected more.",
    "Neither impressed nor disappointed.",
    "Standard packaging, standard product.",
    "Fine for casual use.",
    "Meets expectations, no wow factor.",
    "It is what it is.",
    "Basic item that does the job.",
    "Acceptable quality, average build.",
    "Not bad but not great either.",
    "Okay for the price paid.",
    "Product is fine, service was slow.",
    "Serviceable, would consider alternatives.",
]


def make_reviews_csv() -> None:
    out_dir = os.path.join(ROOT, "reviews")
    os.makedirs(out_dir, exist_ok=True)
    rows = (
        [(t, "positive") for t in POSITIVE]
        + [(t, "negative") for t in NEGATIVE]
        + [(t, "neutral") for t in NEUTRAL]
    )
    random.shuffle(rows)
    path = os.path.join(out_dir, "reviews.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["review", "sentiment"])
        w.writerows(rows)


def main() -> None:
    make_product_images()
    make_face_images()
    make_reviews_csv()
    print("Sample datasets generated under", ROOT)


if __name__ == "__main__":
    main()
