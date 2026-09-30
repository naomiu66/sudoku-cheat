from pathlib import Path

import torch
from transformers import AutoImageProcessor

from sudoku_solver.sudoku_solver import SudokuSolver
from encoder_model.model import DinoClassifier
from utils.hotkey_listener import EXIT_HOTKEYS, KeyboardPoller, RUN_HOTKEY
from utils.sudoku_interactor import read_board, write_solution

def run_once(model, processor):
    board = read_board(model, processor)
    print("Recognized board (solver input):")
    for row in board:
        print(" ".join(row))

    solved_board = [row.copy() for row in board]
    SudokuSolver().solveSudoku(solved_board)
    print("Solved board:")
    for row in solved_board:
        print(" ".join(row))

    entered_count = write_solution(board, solved_board)
    print(f"Entered {entered_count} digits into empty cells.")


def main():
    model_dir = Path("models/v3")
    model = DinoClassifier(n_hid=384, n_classes=10)
    model.load_state_dict(torch.load(model_dir / "model.pth", map_location="cpu"))
    model.eval()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")

    listener = KeyboardPoller()
    listener.start()
    exit_keys = " or ".join(key.upper() for key in EXIT_HOTKEYS)
    print(f"Polling keyboard every 50 ms. Press {RUN_HOTKEY.upper()} to solve; press {exit_keys} to exit.")
    try:
        while True:
            hotkey = listener.get()
            print(f"Hotkey received: {hotkey.upper()}")
            if hotkey in EXIT_HOTKEYS:
                break
            try:
                run_once(model, processor)
            except Exception as error:
                print(f"Run failed: {error}")
    finally:
        listener.stop()

    print("Stopped.")


if __name__ == "__main__":
    main()