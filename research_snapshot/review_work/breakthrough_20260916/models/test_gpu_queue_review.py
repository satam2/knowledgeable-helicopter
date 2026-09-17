"""Disposable CPU subprocess lifecycle checks; no CUDA libraries or jobs."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import psutil

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'review_work/breakthrough_20260916/gpu_queue.py'
spec=importlib.util.spec_from_file_location('reviewed_gpu_queue',SOURCE)
queue=importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)
ORIGINAL_POPEN=subprocess.Popen
SCRIPT='import subprocess,sys,time; child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(20)"]); print(child.pid,flush=True); time.sleep(20)'


def spawn_tree():
    process=ORIGINAL_POPEN([sys.executable,'-u','-c',SCRIPT],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    descendant=psutil.Process(int(process.stdout.readline().strip()))
    return process,descendant


class QueueLifecycle(unittest.TestCase):
    def test_terminate_owned_child_and_grandchild(self):
        began=time.monotonic()
        process,descendant=spawn_tree()
        try:
            queue.stop_tree(process,[descendant])
            self.assertIsNotNone(process.poll())
            self.assertFalse(descendant.is_running())
            queue.stop_tree(process,[descendant])
        finally:
            if process.poll() is None or descendant.is_running():
                queue.stop_tree(process,[descendant])
            process.stdout.close()
            process.stderr.close()
        self.assertLess(time.monotonic()-began,10.)

    def test_monitor_exception_cleans_owned_process_tree(self):
        self.run_exception_case('monitor')

    def test_pid_receipt_exception_cleans_owned_process_tree(self):
        self.run_exception_case('receipt')

    def run_exception_case(self,where):
        began=time.monotonic()
        spawned=[]

        def create(*args,**kwargs):
            process,descendant=spawn_tree()
            spawned.append((process,descendant))
            return process

        original_write=queue.write
        def failing_write(path,value):
            if where=='receipt' and path.name=='results.json' and any('pid' in r for r in value.values()):
                raise RuntimeError('injected_pid_receipt_failure')
            return original_write(path,value)

        available=SimpleNamespace(available=32*1024**3)
        values=[available]*3+[RuntimeError('injected_monitor_failure')] if where=='monitor' else [available]*10
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            plan=base/'test_plan.json'
            plan.write_text(json.dumps({'deadline_utc':(datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat(),
                                        'jobs':[{'name':'cpu_dummy','arguments':['-c','pass'],'max_seconds':5}]}))
            try:
                with patch.object(queue,'external_path',side_effect=lambda path:Path(path)), \
                     patch.object(queue.subprocess,'Popen',side_effect=create), \
                     patch.object(queue.psutil,'virtual_memory',side_effect=values), \
                     patch.object(queue,'write',side_effect=failing_write), \
                     patch.object(sys,'argv',[str(SOURCE),'--plan',str(plan),'--output',str(base/'output')]):
                    with self.assertRaisesRegex(RuntimeError,'injected_'):
                        queue.main()
                self.assertEqual(len(spawned),1)
                process,descendant=spawned[0]
                self.assertIsNotNone(process.poll())
                self.assertFalse(descendant.is_running())
            finally:
                for process,descendant in spawned:
                    if process.poll() is None or descendant.is_running():
                        queue.stop_tree(process,[descendant])
                    process.stdout.close()
                    process.stderr.close()
        self.assertLess(time.monotonic()-began,10.)


if __name__=='__main__':
    unittest.main(verbosity=2)
