from types import SimpleNamespace
import unittest

import watchdog


class FakeProcess:
    def __init__(self, pid, rss, peak, children=()):
        self.pid, self.rss, self.peak, self.descendants = pid, rss, peak, children
    def children(self, recursive=False):
        assert recursive
        return self.descendants
    def memory_info(self):
        return SimpleNamespace(rss=self.rss, peak_wset=self.peak)


class WatchdogTests(unittest.TestCase):
    def test_actual_worker_peak_is_included(self):
        root = FakeProcess(1, 5, 8, [FakeProcess(2, 300, 900)])
        peaks = {}
        sample = watchdog.sample_tree(root, peaks)
        self.assertEqual(sample['rss_bytes'], 305)
        self.assertEqual(sample['conservative_peak_bytes'], 908)
        self.assertEqual(sample['pids'], [1,2])

    def test_previous_peak_survives_worker_exit(self):
        peaks = {2:900}
        sample = watchdog.sample_tree(FakeProcess(1, 5, 8), peaks)
        self.assertEqual(sample['conservative_peak_bytes'], 908)
        self.assertEqual(sample['rss_bytes'], 5)


if __name__ == '__main__':
    unittest.main()
