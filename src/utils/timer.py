from contextlib import contextmanager
from collections import defaultdict
import time

class Timer:
    """
    Lightweight nested timer. Records cumulative time and call count per label,
    then prints a hierarchical report sorted by total time descending.

    Usage:
        timer = Timer()
        with timer.measure("my step"):
            do_work()
        timer.report()
    """

    def __init__(self) -> None:
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int]   = defaultdict(int)
        self._order:  list[str]        = []   # insertion order for display

    @contextmanager
    def measure(self, label: str):
        if label not in self._totals:
            self._order.append(label)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._totals[label] += time.perf_counter() - t0
            self._counts[label] += 1

    def report(self, title: str = "Timing report") -> None:
        if not self._totals:
            print("  (no timings recorded)")
            return

        total_elapsed = sum(self._totals.values())
        col_w = max(len(l) for l in self._order) + 2

        print(f"\n  {'─' * 62}")
        print(f"  {title}")
        print(f"  {'─' * 62}")
        print(f"  {'Step':<{col_w}}  {'Calls':>6}  {'Total (s)':>10}  {'Per call (ms)':>14}  {'Share':>6}")
        print(f"  {'─' * 62}")

        for label in sorted(self._order, key=lambda l: self._totals[l], reverse=True):
            t   = self._totals[label]
            n   = self._counts[label]
            pct = 100.0 * t / total_elapsed if total_elapsed > 0 else 0.0
            print(f"  {label:<{col_w}}  {n:>6}  {t:>10.4f}  {1000*t/n:>14.3f}  {pct:>5.1f}%")

        print(f"  {'─' * 62}")
        print(f"  {'TOTAL':<{col_w}}  {'':>6}  {total_elapsed:>10.4f}")
        print(f"  {'─' * 62}\n")