import cv2
import numpy as np
import os


def create_neon_drone():
    # Create a 64x64 transparent image
    img = np.zeros((64, 64, 4), dtype=np.uint8)

    # Draw glowing cyan body
    cv2.circle(img, (32, 32), 10, (255, 255, 0, 255), -1)  # Center (Cyan in BGRA)

    # Rotors
    cv2.circle(img, (10, 10), 8, (255, 100, 0, 200), 2)
    cv2.circle(img, (54, 10), 8, (255, 100, 0, 200), 2)
    cv2.circle(img, (10, 54), 8, (255, 100, 0, 200), 2)
    cv2.circle(img, (54, 54), 8, (255, 100, 0, 200), 2)

    # Arms
    cv2.line(img, (32, 32), (10, 10), (255, 255, 255, 255), 2)
    cv2.line(img, (32, 32), (54, 10), (255, 255, 255, 255), 2)
    cv2.line(img, (32, 32), (10, 54), (255, 255, 255, 255), 2)
    cv2.line(img, (32, 32), (54, 54), (255, 255, 255, 255), 2)

    cv2.imwrite("src/drone.png", img)
    print("Generated src/drone.png")


def create_fire_particle():
    # 32x32 Glow
    img = np.zeros((32, 32, 4), dtype=np.uint8)
    for r in range(16, 0, -1):
        alpha = int(255 * (1 - r / 16))
        color = (0, 165, 255, alpha)  # Orange BGR
        cv2.circle(img, (16, 16), r, color, -1)
    cv2.imwrite("src/fire_flare.png", img)
    print("Generated src/fire_flare.png")


def create_map_bg():
    # Dark Tech Background
    img = np.zeros((900, 1600, 3), dtype=np.uint8)
    img[:] = (10, 10, 15)  # Dark Blue-Black

    # Draw Grid
    for i in range(0, 1600, 50):
        cv2.line(img, (i, 0), (i, 900), (30, 30, 40), 1)
    for i in range(0, 900, 50):
        cv2.line(img, (0, i), (1600, i), (30, 30, 40), 1)

    cv2.imwrite("src/map_bg.jpg", img)
    print("Generated src/map_bg.jpg")


if __name__ == "__main__":
    if not os.path.exists("src"): os.makedirs("src")
    create_neon_drone()
    create_fire_particle()
    create_map_bg()