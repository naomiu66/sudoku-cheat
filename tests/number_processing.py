from transformers import AutoImageProcessor
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def make_mnist_style(image: Image.Image) -> Image.Image:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    histogram = np.bincount(gray.ravel(), minlength=256)
    total = gray.size
    total_sum = np.dot(np.arange(256), histogram)
    background_sum = 0
    background_count = 0
    best_variance = -1.0
    threshold = 0

    for value, count in enumerate(histogram):
        background_count += count
        if background_count == 0:
            continue
        foreground_count = total - background_count
        if foreground_count == 0:
            break
        background_sum += value * count
        background_mean = background_sum / background_count
        foreground_mean = (total_sum - background_sum) / foreground_count
        variance = background_count * foreground_count * (background_mean - foreground_mean) ** 2
        if variance > best_variance:
            best_variance = variance
            threshold = value

    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    light_background = np.median(border) > threshold
    contrast = max(int(gray.max()) - int(gray.min()), 1)
    gray_float = gray.astype(np.float32)
    if light_background:
        foreground = gray <= threshold
        ink = (int(gray.max()) - gray_float) * 255 / contrast
    else:
        foreground = gray > threshold
        ink = (gray_float - int(gray.min())) * 255 / contrast

    if not foreground.any():
        return Image.new("L", (28, 28), 0)

    ys, xs = np.nonzero(foreground)
    ink_image = Image.fromarray(ink.astype(np.uint8)).crop(
        (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    )
    scale = 20 / max(ink_image.size)
    resized_size = tuple(max(1, round(size * scale)) for size in ink_image.size)
    ink_image = ink_image.resize(resized_size, Image.Resampling.BILINEAR)

    canvas = np.zeros((48, 48), dtype=np.float32)
    resized = np.asarray(ink_image, dtype=np.float32)
    x = (48 - ink_image.width) // 2
    y = (48 - ink_image.height) // 2
    canvas[y:y + ink_image.height, x:x + ink_image.width] = resized

    mass = canvas.sum()
    if mass:
        ys, xs = np.indices(canvas.shape)
        center_x = (canvas * xs).sum() / mass
        center_y = (canvas * ys).sum() / mass
        shift_x = round(23.5 - center_x)
        shift_y = round(23.5 - center_y)
        shifted = np.zeros_like(canvas)
        source_x = slice(max(0, -shift_x), min(48, 48 - shift_x))
        source_y = slice(max(0, -shift_y), min(48, 48 - shift_y))
        target_x = slice(max(0, shift_x), min(48, 48 + shift_x))
        target_y = slice(max(0, shift_y), min(48, 48 + shift_y))
        shifted[target_y, target_x] = canvas[source_y, source_x]
        canvas = shifted

    return Image.fromarray(np.clip(canvas[10:38, 10:38], 0, 255).astype(np.uint8))


processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")

cells_dir = Path("data/sudoku_debug/20261011_160612_480097/individual_cells")

if not cells_dir.is_dir():
    raise FileNotFoundError(f"Папка с изображениями не найдена: {cells_dir.resolve()}")

image_paths = sorted(
    p for p in cells_dir.iterdir()
    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
)

print(f"Найдено изображений: {len(image_paths)}")

if not image_paths:
    raise FileNotFoundError(f"В папке нет изображений: {cells_dir.resolve()}")

images_to_show = image_paths[:4]
fig, axes = plt.subplots(
    len(images_to_show),
    4,
    figsize=(16, 4 * len(images_to_show)),
    squeeze=False,
)

for row, image_path in enumerate(images_to_show):
    image = Image.open(image_path).convert("RGB")
    mnist_image = make_mnist_style(image)
    inputs = processor(images=image, return_tensors="pt")
    mnist_inputs = processor(images=mnist_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"][0].permute(1, 2, 0).cpu().numpy()
    mnist_pixel_values = mnist_inputs["pixel_values"][0].permute(1, 2, 0).cpu().numpy()
    mean = np.asarray(processor.image_mean, dtype=np.float32)
    std = np.asarray(processor.image_std, dtype=np.float32)
    processed_image = np.clip(pixel_values * std + mean, 0, 1)
    processed_mnist_image = np.clip(mnist_pixel_values * std + mean, 0, 1)

    print(f"{image_path.name}: processor={inputs['pixel_values'].shape}, MNIST={mnist_image.size}")

    axes[row, 0].imshow(image)
    axes[row, 0].set_title(f"{image_path.name} — оригинал")
    axes[row, 0].axis("off")

    axes[row, 1].imshow(processed_image)
    axes[row, 1].set_title("После processor (размер модели)")
    axes[row, 1].axis("off")

    axes[row, 2].imshow(mnist_image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[row, 2].set_title("MNIST-стиль (28×28)")
    axes[row, 2].axis("off")
    
    axes[row, 3].imshow(processed_mnist_image)
    axes[row, 3].set_title("MNIST-стиль processed (28×28)")
    axes[row, 3].axis("off")

plt.tight_layout()
plt.show()