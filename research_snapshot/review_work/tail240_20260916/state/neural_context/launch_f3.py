"""Stricter coordinated host-reserve check for the declared grouped-bin run."""
import torch
import runpy
import run_f3_v2 as wrapper


if __name__=='__main__':
    protocol=wrapper.declare()
    available=wrapper.frozen.psutil.virtual_memory().available
    assert available>=28*1024**3,'Coordinated F3 launch requires>=28GiB free'
    path=wrapper.OUT/'launch_resource.json'
    assert not path.exists()
    wrapper.frozen.common.write_json(path,dict(
        available_bytes=available,required_available_bytes=28*1024**3,
        launcher_sha256=wrapper.frozen.common.sha256(__file__),
        protocol_sha256=wrapper.frozen.common.sha256(wrapper.OUT/'protocol.json'),
        created_utc=wrapper.frozen.common.utc_now(),
        policy='Parent-coordinated CPU release required; stricter28GiBstartcheck supersedesweaker25GiBwrapperassert. Unchanged20GiBRSS/8GiBreserve budgets.'))
    runpy.run_path(wrapper.__file__,run_name='__main__')
