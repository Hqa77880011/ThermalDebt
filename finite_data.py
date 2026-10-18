"""Rendered shape classification with explicit finite costs, not measured device timings."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def render(shape, rng):
    image = Image.new("RGB", (128, 128), (30, 30, 30))
    draw = ImageDraw.Draw(image)
    x, y = rng.integers(35, 70, 2)
    radius = int(rng.integers(12, 30))
    color = tuple(int(v) for v in rng.integers(100, 240, 3))
    if shape == "circle":draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
    elif shape == "square":draw.rectangle([x-radius, y-radius, x+radius, y+radius], fill=color)
    else:draw.polygon([(x, y-radius), (x-radius, y+radius), (x+radius, y+radius)], fill=color)
    return image.filter(ImageFilter.GaussianBlur(float(rng.uniform(0, 2.5))))


def classify(image, resolution):
    """Actual pixel occupancy classifier; no class annotation is an input."""
    pixels = np.asarray(image.resize((resolution, resolution)), dtype=float)
    mask = pixels.mean(2) > 75
    yy, xx = np.where(mask)
    if not len(xx):return 0
    occupancy = mask.sum() / ((xx.max()-xx.min()+1) * (yy.max()-yy.min()+1))
    return 2 if occupancy < 0.63 else 0 if occupancy < 0.9 else 1


def generate(output, seed):
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    profile = json.loads((Path(__file__).parent / "thermaldebt/defaults.json").read_text(encoding="utf-8"))
    for action, resolution in zip(profile["actions"][:-1], [16, 24, 40]):
        action["resolution"] = resolution
        action["latency_s"] = 0.004 + resolution**2 * 0.000035
        action["load_s"] = 0.01
    profile["input_source"] = "finite_pixel_classifier_and_declared_cost_model"
    profile["timing_source"] = "0.004 + 0.000035*resolution^2 seconds; not hardware measurements"
    (output / "finite_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    for role, count in [("development", 90), ("evaluation", 90)]:
        (output / role).mkdir()
        records = []
        for index in range(count):
            shape = ["circle", "square", "triangle"][index % 3]
            name = f"{role}/image-{index:03d}.png"
            render(shape, rng).save(output / name)
            records.append(dict(id=f"{role}-{index}", path=name, label=index % 3))
        with (output / f"{role}_images.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)


def profile_predictions(source):
    profile = json.loads((source / "finite_profile.json").read_text(encoding="utf-8"))
    for role in ("development", "evaluation"):
        with (source / f"{role}_images.csv").open(newline="", encoding="utf-8") as stream:
            images = list(csv.DictReader(stream))
        rows = []
        for image in images:
            with Image.open(source / image["path"]) as opened:
                for action in profile["actions"][:-1]:
                    prediction = classify(opened.convert("RGB"), action["resolution"])
                    rows.append(dict(id=image["id"], action=action["name"], label=int(image["label"]),
                                     prediction=prediction, latency_s=action["latency_s"],
                                     load_s=action["load_s"], timing_source="finite_cost_model"))
        with (source / f"{role}.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["generate", "profile"])
    parser.add_argument("--output")
    parser.add_argument("--source")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if args.stage == "generate":
        if not args.output:parser.error("generate requires --output")
        generate(Path(args.output), args.seed)
    else:
        if not args.source:parser.error("profile requires --source")
        profile_predictions(Path(args.source))


if __name__ == "__main__":
    main()
