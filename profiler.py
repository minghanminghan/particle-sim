from functools import wraps
import cProfile, pstats
from typing import Callable
import time
import os
from datetime import datetime


def f8_alt(x):
    return f"{x:14.9f}"

pstats.f8 = f8_alt

def profile(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        p = cProfile.Profile()
        p.enable()
        result = func(*args, **kwargs)
        p.disable()
        stats = pstats.Stats(p)
        for label in ['cumulative', 'calls', 'time']:
            sorted_stats = stats.sort_stats(label)
            sorted_stats.print_stats()
        return result
    return wrapper


def profile_and_save(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        p = cProfile.Profile()
        p.enable()
        result = func(*args, **kwargs)
        p.disable()
        
        stats = pstats.Stats(p)
        for label in ['cumulative', 'calls', 'time']:
            sorted_stats = stats.sort_stats(label)
            sorted_stats.print_stats(100)
        
        filename = os.path.join(os.getcwd(), "tests", f"{datetime.now().strftime("%Y-%m-%d_%H%M%S")}.txt")
        with open(filename, 'w') as stream:
            stats = pstats.Stats(p, stream=stream)
            for label in ['cumulative', 'calls', 'time']:
                sorted_stats = stats.sort_stats(label)
                sorted_stats.print_stats(100)

        return result
    return wrapper

@profile
def main():
    for i in range(5):
        time.sleep(0.2)


if __name__ == '__main__':
    main()