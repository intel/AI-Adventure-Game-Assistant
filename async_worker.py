import multiprocessing
from abc import ABC, abstractmethod

class AsyncWorker(ABC):
    def __init__(self):
        self._process = None
        self._running = multiprocessing.Value('b', False)  # Shared boolean flag to control the loop

    @abstractmethod
    def _work_loop(self):
        """
        The main loop that runs in the separate process.
        """
        pass

    def start(self):
        """
        Start the worker process.
        """
        if self._process is None or not self._process.is_alive():
            self._running.value = True
            self._process = multiprocessing.Process(target=self._work_loop)
            self._process.start()
            print(f"{self.__class__.__name__}: Process started.")

    def stop(self):
        """
        Stop the worker process.
        """
        if self._process is not None:
            self._running.value = False
            self._process.join()  # Wait for the process to exit
            print(f"{self.__class__.__name__}: Process stopped.")
            self._process = None