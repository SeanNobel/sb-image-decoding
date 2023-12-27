from termcolor import cprint
from time import time


def timer(func):
    def wrapper(*args, **kwargs):
        start = time()

        result = func(*args, **kwargs)

        end = time()

        # logger.info(f"{func.__name__} took {end - start} seconds")
        cprint(f"{func.__name__} took {end - start} seconds")

        return result

    return wrapper
