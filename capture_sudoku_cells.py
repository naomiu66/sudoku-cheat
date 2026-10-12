import argparse
import shutil
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import pyautogui
from PIL import Image, ImageDraw, ImageTk

from utils.cell_processing import crop_board_cells, is_empty_cell
from utils.hotkey_listener import EXIT_HOTKEYS, KeyboardPoller, RUN_HOTKEY
from utils.sudoku_interactor import BOARD_BOTTOM_RIGHT, BOARD_TOP_LEFT

CELL_DISPLAY_SIZE = 160
LABEL_HEIGHT = 26
OUTPUT_DIR = Path("data/sudoku_debug")
DATASET_DIR = Path("data/dataset/val")


def capture_once(output_dir: Path = OUTPUT_DIR):
    left, top = BOARD_TOP_LEFT
    right, bottom = BOARD_BOTTOM_RIGHT
    screenshot = pyautogui.screenshot(region=(left, top, right - left, bottom - top))
    cells = crop_board_cells(screenshot)

    capture_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
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
    return capture_dir


def _cell_file_for(row: int, column: int) -> str:
    return f"cell_r{row + 1:02}_c{column + 1:02}.png"


def _label_from_empty_cell(cell_path: Path) -> int | None:
    image = Image.open(cell_path).convert("RGB")
    if is_empty_cell(image):
        return 0
    return None


def _label_cells(cells: list[tuple[int, int, Path, int | None]]) -> list[int]:
    if not cells:
        return []

    root = tk.Tk()
    root.title("Label Sudoku cells")
    root.resizable(False, False)

    image_label = ttk.Label(root)
    image_label.pack(padx=16, pady=(16, 8))
    progress_label = ttk.Label(root, font=("Segoe UI", 12, "bold"))
    progress_label.pack(pady=4)
    hint_label = ttk.Label(root)
    hint_label.pack(pady=4)
    buttons = ttk.Frame(root)
    buttons.pack(padx=16, pady=(8, 16))

    labels: list[int] = []
    current_index = 0
    current_photo = None

    def choose(label: int) -> None:
        nonlocal current_index
        labels.append(label)
        current_index += 1
        show_current()

    def show_current() -> None:
        nonlocal current_photo
        if current_index >= len(cells):
            root.destroy()
            return

        row, column, cell_path, guessed_label = cells[current_index]
        with Image.open(cell_path) as source:
            image = source.convert("RGB")
        scale = min(480 / image.width, 480 / image.height)
        display_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(display_size, Image.Resampling.NEAREST)
        current_photo = ImageTk.PhotoImage(image)
        image_label.configure(image=current_photo)
        progress_label.configure(
            text=f"Клетка {current_index + 1}/{len(cells)} — строка {row + 1}, столбец {column + 1}"
        )
        hint_label.configure(text="Выберите цифру. Клавиши 0–9 или кнопки ниже; точка — пустая клетка.")

    for digit in range(10):
        ttk.Button(
            buttons,
            text=str(digit),
            width=4,
            command=lambda value=digit: choose(value),
        ).grid(row=0, column=digit, padx=2)

    for digit in range(10):
        root.bind(f"<KeyPress-{digit}>", lambda event, value=digit: choose(value))
    root.bind("<KeyPress-period>", lambda event: choose(0))
    root.bind("<Escape>", lambda event: root.destroy())
    show_current()
    root.mainloop()

    if len(labels) != len(cells):
        raise RuntimeError("Labeling was cancelled before all cells were labeled")
    return labels


def build_dataset_from_capture(capture_dir: Path, dataset_root: Path = DATASET_DIR, auto_empty: bool = True):
    capture_dir = Path(capture_dir)
    cells_dir = capture_dir / "individual_cells"
    if not cells_dir.is_dir():
        raise FileNotFoundError(f"Cells directory not found: {cells_dir.resolve()}")

    dataset_dir = Path(dataset_root)
    for label in range(10):
        (dataset_dir / str(label)).mkdir(parents=True, exist_ok=True)

    cells_to_label = []
    labeled_cells = []
    for row in range(9):
        for column in range(9):
            source_path = cells_dir / _cell_file_for(row, column)
            if not source_path.exists():
                raise FileNotFoundError(f"Missing cell file: {source_path.resolve()}")
            guessed_label = _label_from_empty_cell(source_path) if auto_empty else None
            cell = (row, column, source_path, guessed_label)
            if guessed_label == 0:
                labeled_cells.append((cell, 0))
            else:
                cells_to_label.append(cell)

    labels = _label_cells(cells_to_label)
    labeled_cells.extend(zip(cells_to_label, labels))
    for (row, column, source_path, _), label in labeled_cells:
        class_dir = dataset_dir / str(label)
        target_name = f"{capture_dir.name}_r{row + 1:02}_c{column + 1:02}.png"
        target_path = class_dir / target_name
        for existing_class in range(10):
            existing_path = dataset_dir / str(existing_class) / target_name
            if existing_path.exists():
                existing_path.unlink()
        shutil.copy2(source_path, target_path)

    print(f"Dataset saved to: {dataset_dir.resolve()}")
    return dataset_dir


def _capture_and_label(capture_dir: Path | None = None, dataset_root: Path = DATASET_DIR, auto_empty: bool = True):
    capture_dir = capture_dir or capture_once(OUTPUT_DIR)
    print(f"\nLabeling captured board: {capture_dir.resolve()}")
    if auto_empty:
        print("Detected empty cells are saved as class 0 automatically. Label the remaining cells with digit buttons/keys 0-9; Escape cancels.")
    else:
        print("Label every cell with digit buttons/keys 0-9; use 0 for empty cells. Escape cancels.")
    return build_dataset_from_capture(capture_dir, dataset_root=dataset_root, auto_empty=auto_empty)


def _interactive_capture_loop():
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
                _capture_and_label()
            except Exception as error:
                print(f"Capture failed: {error}")
    finally:
        listener.stop()

    print("Stopped.")


def _wait_for_capture_hotkey() -> bool:
    listener = KeyboardPoller()
    listener.start()
    exit_keys = " or ".join(key.upper() for key in EXIT_HOTKEYS)
    print(f"Press {RUN_HOTKEY.upper()} to capture; press {exit_keys} to cancel.")
    try:
        hotkey = listener.get()
        if hotkey in EXIT_HOTKEYS:
            print("Capture cancelled.")
            return False
        print(f"Hotkey received: {hotkey.upper()}")
        return True
    finally:
        listener.stop()


def main():
    parser = argparse.ArgumentParser(description="Capture Sudoku cells and collect a labelled dataset.")
    parser.add_argument("--capture-only", action="store_true", help="Capture only one board and save the raw crop images.")
    parser.add_argument("--capture-and-label", action="store_true", help="Capture one board then label each cell and save dataset files.")
    parser.add_argument("--label-dir", type=str, help="Path to a capture directory with individual_cells/ that should be labelled into a dataset.")
    parser.add_argument("--dataset-root", type=str, default=str(DATASET_DIR), help="Root directory for the generated dataset.")
    parser.add_argument("--no-auto-empty", action="store_true", help="Disable auto-empty detection for cells.")
    args = parser.parse_args()

    if args.label_dir:
        build_dataset_from_capture(Path(args.label_dir), dataset_root=Path(args.dataset_root), auto_empty=not args.no_auto_empty)
        return

    if args.capture_and_label:
        if _wait_for_capture_hotkey():
            _capture_and_label(dataset_root=Path(args.dataset_root), auto_empty=not args.no_auto_empty)
        return

    if args.capture_only:
        if _wait_for_capture_hotkey():
            capture_once(OUTPUT_DIR)
        return

    _interactive_capture_loop()


if __name__ == "__main__":
    main()