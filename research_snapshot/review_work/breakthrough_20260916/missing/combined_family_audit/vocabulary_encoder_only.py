"""Verify XGB vocabularies without instantiating large native GPU-trained boosters."""
import json
import joblib.numpy_pickle
import vocabulary as frozen


class DiscardedEstimator:
    def __setstate__(self,state):
        self.discarded=True


class EncoderUnpickler(joblib.numpy_pickle.NumpyUnpickler):
    def find_class(self,module,name):
        if module.startswith('xgboost') and name in ['Booster','XGBRegressor']:
            return DiscardedEstimator
        return super().find_class(module,name)


def encoder_only(path):
    with open(path,'rb') as source:
        return EncoderUnpickler(str(path),source,ensure_native_byte_order=True).load()


if __name__=='__main__':
    frozen.joblib.load=encoder_only
    original_fold=frozen.common.fold_data
    def fold_data(*args,**kwargs):
        indices,split,ids=original_fold(*args,**kwargs)
        return indices,json.loads(json.dumps(split,default=str)),ids
    frozen.common.fold_data=fold_data
    original_write=frozen.common.write_json
    def write(path,payload):
        payload['wrapper_sha256']=frozen.common.sha256(__file__)
        payload['native_booster_instantiated']=False
        payload['estimator_discarded_by_custom_unpickler']=True
        original_write(path.with_name(path.stem+'_encoder_only.json'),payload)
    frozen.common.write_json=write
    frozen.main()
