from pathlib import Path
from datetime import datetime

import pyautogui
from PIL import Image, ImageDraw

from utils.hotkey_listener import EXIT_HOTKEYS, KeyboardPoller, RUN_HOTKEY
from utils.sudoku_interactor import BOARD_BOTTOM_RIGHT, BOARD_TOP_LEFT, crop_board_cells

CELL_DISPLAY_SIZE = 160
LABEL_HEIGHT = 26
OUTPUT_DIR = Path("data/sudoku_debug")
INDIVIDUAL_CELLS_DIR = OUTPUT_DIR / "individual_cells"


def capture_once():
    left, top = BOARD_TOP_LEFT
    right, bottom = BOARD_BOTTOM_RIGHT
    screenshot = pyautogui.screenshot(region=(left, top, right - left, bottom - top))
    cells = crop_board_cells(screenshot)

    capture_dir = OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    individual_cells_dir = capture_dir / "individual_cells"
    individual_cells_dir.mkdir(parents=True, exist_ok=True)
    board_path = capture_dir / "board.png"
    cells_path = capture_dir / "cells.png"
    screenshot.convert("RGB").save(board_path, optimize=True)

    contact_sheet = Image.new(
        "RGB",
        (9 * CELL_DISPLAY_SIZE, 9 * (CELL_DISPLAY_SIZE + LABEL_HEIGHT)),
        "white",
    )
    draw = ImageDraw.Draw(contact_sheet)

    for row, cell_row in enumerate(cells):
        for column, cell in enumerate(cell_row):
            x = column * CELL_DISPLAY_SIZE
            y = row * (CELL_DISPLAY_SIZE + LABEL_HEIGHT)
            draw.text((x + 4, y + 5), f"row {row + 1}, col {column + 1}", fill="black")
            cell_path = individual_cells_dir / f"cell_r{row + 1:02}_c{column + 1:02}.png"
            cell.convert("RGB").save(cell_path, optimize=True)
            enlarged_cell = cell.resize(
                (CELL_DISPLAY_SIZE, CELL_DISPLAY_SIZE),
                Image.Resampling.NEAREST,
            )
            contact_sheet.paste(enlarged_cell, (x, y + LABEL_HEIGHT))

    contact_sheet.save(cells_path, optimize=True)
    print(f"Saved full board: {board_path.resolve()}")
    print(f"Saved cropped cells: {cells_path.resolve()}")
    print(f"Saved 81 individual cells: {individual_cells_dir.resolve()}")


def main():
    listener = KeyboardPoller()
    listener.start()
    exit_keys = " or ".join(key.upper() for key in EXIT_HOTKEYS)
    print(f"Polling keyboard every 50 ms. Press {RUN_HOTKEY.upper()} to capture; press {exit_keys} to exit.")
    try:
        while True:
            hotkey = listener.get()
            print(f"Hotkey received: {hotkey.upper()}")
            if hotkey in EXIT_HOTKEYS:
                break
            try:
                capture_once()
            except Exception as error:
                print(f"Capture failed: {error}")
    finally:
        listener.stop()

    print("Stopped.")


if __name__ == "__main__":
    main()