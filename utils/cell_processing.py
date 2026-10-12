from statistics import median


BOARD_SIZE = 9
CELL_MARGIN_RATIO = 0.09
PIXEL_DIFFERENCE_THRESHOLD = 35
MIN_FOREGROUND_RATIO = 0.015


def is_empty_cell(image):
    grayscale = image.convert("L")
    width, height = grayscale.size
    border_width = max(1, min(width, height) // 10)
    border_pixels = [
        grayscale.getpixel((x, y))
        for y in range(height)
        for x in range(width)
        if x < border_width or x >= width - border_width or y < border_width or y >= height - border_width
    ]
    background = median(border_pixels)
    foreground_pixels = sum(
        abs(pixel - background) >= PIXEL_DIFFERENCE_THRESHOLD
        for pixel in grayscale.getdata()
    )
    return foreground_pixels / (width * height) < MIN_FOREGROUND_RATIO


def crop_board_cells(screenshot):
    cell_width = screenshot.width / BOARD_SIZE
    cell_height = screenshot.height / BOARD_SIZE
    cells = []

    for row in range(BOARD_SIZE):
        cell_row = []
        for column in range(BOARD_SIZE):
            x1 = round(column * cell_width)
            y1 = round(row * cell_height)
            x2 = round((column + 1) * cell_width)
            y2 = round((row + 1) * cell_height)
            margin_x = round((x2 - x1) * CELL_MARGIN_RATIO)
            margin_y = round((y2 - y1) * CELL_MARGIN_RATIO)
            cell_row.append(screenshot.crop((x1 + margin_x, y1 + margin_y, x2 - margin_x, y2 - margin_y)))
        cells.append(cell_row)

    return cells
