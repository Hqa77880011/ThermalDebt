import unittest
from PIL import Image, ImageDraw
from finite_data import classify


class FiniteInputTests(unittest.TestCase):
    def test_pixel_classifier_geometry(self):
        shapes = []
        for shape in ("circle", "square", "triangle"):
            image = Image.new("RGB", (64, 64), (30, 30, 30))
            draw = ImageDraw.Draw(image)
            if shape == "circle":draw.ellipse([12, 12, 52, 52], fill="white")
            elif shape == "square":draw.rectangle([12, 12, 52, 52], fill="white")
            else:draw.polygon([(32, 12), (12, 52), (52, 52)], fill="white")
            shapes.append(classify(image, 64))
        self.assertEqual(shapes, [0, 1, 2])
