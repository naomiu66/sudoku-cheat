import pyautogui
# import time
from statistics import median

from encoder_model.inference import predict

BOARD_TOP_LEFT = (340, 274)
BOARD_BOTTOM_RIGHT = (960, 895)
BOARD_SIZE = 9
CELL_MARGIN_RATIO = 0.09
PIXEL_DIFFERENCE_THRESHOLD = 35
MIN_FOREGROUND_RATIO = 0.015
# CELL_ENTRY_DELAY_SECONDS = 0.0001


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


def read_board(model, processor, top_left=BOARD_TOP_LEFT, bottom_right=BOARD_BOTTOM_RIGHT):
	left, top = top_left
	right, bottom = bottom_right
	if right <= left or bottom <= top:
		raise ValueError("The bottom-right corner must be below and to the right of the top-left corner")

	screenshot = pyautogui.screenshot(region=(left, top, right - left, bottom - top))
	cells = crop_board_cells(screenshot)
	board = []

	for row in range(BOARD_SIZE):
		board_row = []
		for column in range(BOARD_SIZE):
			cell = cells[row][column]

			if is_empty_cell(cell):
				board_row.append(".")
			else:
				digit, _ = predict(cell, model, processor)
				if digit == 0:
					board_row.append(".")
				elif digit in range(1, 10):
					board_row.append(str(digit))
				else:
					raise ValueError(f"Invalid digit prediction {digit} at row {row + 1}, column {column + 1}")
		board.append(board_row)

	return board


def write_solution(original_board, solved_board, top_left=BOARD_TOP_LEFT, bottom_right=BOARD_BOTTOM_RIGHT):
	if len(original_board) != BOARD_SIZE or len(solved_board) != BOARD_SIZE:
		raise ValueError("Both boards must be 9x9 grids")
	if any(len(row) != BOARD_SIZE for row in original_board + solved_board):
		raise ValueError("Both boards must be 9x9 grids")

	left, top = top_left
	right, bottom = bottom_right
	if right <= left or bottom <= top:
		raise ValueError("The bottom-right corner must be below and to the right of the top-left corner")

	entries = []
	for row in range(BOARD_SIZE):
		for column in range(BOARD_SIZE):
			original_value = original_board[row][column]
			solved_value = solved_board[row][column]
			if solved_value not in "123456789":
				raise ValueError(f"Invalid solved value {solved_value!r} at row {row + 1}, column {column + 1}")
			if original_value != "." and original_value != solved_value:
				raise ValueError(f"Solution changes clue at row {row + 1}, column {column + 1}")
			if original_value == ".":
				entries.append((row, column, solved_value))

	width = right - left
	height = bottom - top
	for row, column, digit in entries:
		x = left + round((column + 0.5) * width / BOARD_SIZE)
		y = top + round((row + 0.5) * height / BOARD_SIZE)
		pyautogui.click(x, y)
		pyautogui.write(digit)
		# time.sleep(CELL_ENTRY_DELAY_SECONDS)

	return len(entries)