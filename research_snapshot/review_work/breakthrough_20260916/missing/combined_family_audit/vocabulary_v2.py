"""Normalize in-memory tuple split fields before JSON-manifest comparison."""
import json
import vocabulary as frozen


if __name__=='__main__':
    original_fold=frozen.common.fold_data
    def fold_data(*args,**kwargs):
        indices,split,ids=original_fold(*args,**kwargs)
        return indices,json.loads(json.dumps(split)),ids
    frozen.common.fold_data=fold_data
    original_write=frozen.common.write_json
    def write(path,payload):
        payload['wrapper_sha256']=frozen.common.sha256(__file__)
        original_write(path.with_name(path.stem+'_v2.json'),payload)
    frozen.common.write_json=write
    frozen.main()
