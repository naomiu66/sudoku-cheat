import ctypes
import os
from queue import Queue
from threading import Event, Thread

RUN_HOTKEY = "f8"
EXIT_HOTKEYS = ("esc", "ctrl+shift+q")
POLL_INTERVAL_SECONDS = 0.05

_KEY_CODES = {
	RUN_HOTKEY: (0x77,),
	"esc": (0x1B,),
	"ctrl+shift+q": (0x11, 0x10, 0x51),
}

if os.name != "nt":
	raise OSError("The global keyboard poller currently supports Windows only")

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_get_async_key_state = _user32.GetAsyncKeyState
_get_async_key_state.argtypes = [ctypes.c_int]
_get_async_key_state.restype = ctypes.c_short


def _is_pressed(key_code: int) -> bool:
	return bool(_get_async_key_state(key_code) & 0x8000)


class KeyboardPoller:
	def __init__(self):
		self.events: Queue[str] = Queue()
		self._stop_event = Event()
		self._thread = Thread(target=self._poll, daemon=True)

	def start(self) -> None:
		self._thread.start()

	def get(self) -> str:
		return self.events.get()

	def stop(self) -> None:
		self._stop_event.set()
		self._thread.join(timeout=POLL_INTERVAL_SECONDS * 2)


	def _poll(self) -> None:
		previous = {hotkey: False for hotkey in _KEY_CODES}
		while not self._stop_event.wait(POLL_INTERVAL_SECONDS):
			current = {
				hotkey: all(_is_pressed(key_code) for key_code in key_codes)
				for hotkey, key_codes in _KEY_CODES.items()
			}
			for hotkey, is_down in current.items():
				if is_down and not previous[hotkey]:
					self.events.put(hotkey)
			previous = current


__all__ = ["EXIT_HOTKEYS", "KeyboardPoller", "RUN_HOTKEY"]