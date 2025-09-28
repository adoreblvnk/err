import random
import math
from PIL import Image, ImageDraw
from io import BytesIO

def generate_synthetic_jpeg_bytes(width=320, height=240, frame_num=0):
    """
    Generate conveyor belt style frames with shapes (circle, square, triangle)
    scrolling left to right. Shapes vary in size, color, and rotation.
    """
    im = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(im)

    # Conveyor belt background
    belt_height = height // 3
    belt_y = height // 2 - belt_height // 2
    draw.rectangle([0, belt_y, width, belt_y + belt_height], fill=(60, 60, 60))

    # Fixed seed for reproducibility of objects across frames
    random.seed(42)
    num_objects = 6
    spacing = width // num_objects
    speed = 5  # pixels per frame

    for i in range(num_objects):
        # Horizontal position based on frame number
        offset = (frame_num * speed + i * spacing) % (width + spacing)
        x = width - offset
        y_center = height // 2

        shape_type = ["circle", "square", "triangle"][i % 3]
        color = random.choice([
            (200, 50, 50), (50, 200, 50), (50, 50, 200),
            (200, 200, 50), (200, 100, 200), (50, 200, 200)
        ])

        # Each shape has deterministic random size/rotation tied to index
        random.seed(i)
        size = random.randint(20, 60)
        rotation = random.randint(0, 360)

        if shape_type == "circle":
            # Circles don't need rotation
            draw.ellipse(
                [x - size//2, y_center - size//2, x + size//2, y_center + size//2],
                fill=color
            )

        elif shape_type == "square":
            # Draw a rotated square as polygon
            half = size // 2
            points = [
                (-half, -half),
                (half, -half),
                (half, half),
                (-half, half)
            ]
            angle = math.radians(rotation)
            rot_points = [
                (x + px * math.cos(angle) - py * math.sin(angle),
                 y_center + px * math.sin(angle) + py * math.cos(angle))
                for px, py in points
            ]
            draw.polygon(rot_points, fill=color)

        elif shape_type == "triangle":
            # Equilateral triangle rotated
            r = size // 2
            base_points = [
                (0, -r),
                (-r * math.sin(math.radians(60)), r * math.cos(math.radians(60))),
                (r * math.sin(math.radians(60)), r * math.cos(math.radians(60)))
            ]
            angle = math.radians(rotation)
            rot_points = [
                (x + px * math.cos(angle) - py * math.sin(angle),
                 y_center + px * math.sin(angle) + py * math.cos(angle))
                for px, py in base_points
            ]
            draw.polygon(rot_points, fill=color)

    buf = BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return buf.getvalue()

