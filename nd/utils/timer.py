from termcolor import cprint
from time import time


def timer(func):
    def wrapper(*args, **kwargs):
        start = time()

        result = func(*args, **kwargs)

        end = time()

        cprint(f"{func.__name__} took {end - start} seconds", "yellow")

        return result

    return wrapper
