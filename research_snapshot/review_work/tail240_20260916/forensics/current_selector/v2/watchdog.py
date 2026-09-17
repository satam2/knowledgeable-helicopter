"""Account for Windows venv launcher and its recursive Python worker tree."""
import psutil


def sample_tree(root, peaks):
    try:
        processes = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        processes = []
    rss, pids = 0, []
    for process in processes:
        try:
            info = process.memory_info()
        except psutil.NoSuchProcess:
            continue
        pids.append(process.pid)
        rss += info.rss
        peaks[process.pid] = max(peaks.get(process.pid, 0), info.rss, getattr(info, 'peak_wset', info.rss))
    return dict(rss_bytes=rss, conservative_peak_bytes=sum(peaks.values()), pids=pids)


def terminate_tree(root):
    try:
        processes = [*reversed(root.children(recursive=True)), root]
    except psutil.NoSuchProcess:
        return
    for process in processes:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(processes, timeout=5)
