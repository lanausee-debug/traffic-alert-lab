"""Evaluation-only reader requiring an intact execution seal and preflight."""
import contextlib
from unittest.mock import patch
import h5py
import numpy as np
from pems_bay_reader import SealedPartitionReader
from freeze_pems_bay_evaluation import ROOT,OUTPUT,SEAL,verify_execution
from train_pems_bay_forecaster import read_json
from freeze_external_validation import sha256


def validate_request(partition,start,stop,bounds,enabled,consumed):
    if not enabled: raise PermissionError('Evaluation reader is sealed')
    if partition!='evaluation' or (start,stop)!=tuple(bounds):
        raise PermissionError('Only the complete bound evaluation partition is allowed')
    if consumed: raise PermissionError('Evaluation speed read already consumed for this reader')


class EvaluationReader(SealedPartitionReader):
    def __init__(self,path,config,permit=False):
        self.execution=verify_execution(); self._permitted=False; self._consumed=False
        if permit:
            preflight=OUTPUT/'preflight.json'
            if not preflight.exists(): raise PermissionError('Pre-unseal audit missing')
            proof=read_json(preflight)
            if not proof.get('passed') or proof['execution_freeze_sha256']!=sha256(SEAL):
                raise PermissionError('Pre-unseal audit is not bound to this execution freeze')
            if proof['audit_source_sha256']!=self.execution['source_sha256']['audit_pems_bay_evaluation.py']:
                raise PermissionError('Preflight audit source differs')
            self._permitted=True
        super().__init__(path,config)
        if list(self.bounds['evaluation'])!=self.execution['evaluation_bounds'] or self.selected_ids!=self.execution['selected_ids'] or self.columns.tolist()!=self.execution['selected_columns']:
            raise PermissionError('Reader metadata differs from execution freeze')

    def _authorize(self,partition,start,stop):
        validate_request(partition,start,stop,self.execution['evaluation_bounds'],self._permitted,self._consumed)

    def _read_values(self,partition,start,stop):
        self._authorize(partition,start,stop)
        # Parent calls _authorize again before indexing, so consume only after it returns.
        raw=super()._read_values(partition,start,stop); self._consumed=True
        return raw


@contextlib.contextmanager
def guard_evaluation_reads(reader,trace,allow=True):
    original=h5py.Dataset.__getitem__; columns=sorted(reader.columns.tolist())
    def guarded(dataset,key,*args,**kwargs):
        if dataset.name=='/speed/block0_values':
            if not allow or len(trace): raise PermissionError('Evaluation speed access denied or already consumed')
            if not isinstance(key,tuple) or len(key)!=2 or not isinstance(key[0],slice):
                raise PermissionError('Unbounded evaluation speed access denied')
            rows=key[0]
            if (rows.start,rows.stop)!=tuple(reader.execution['evaluation_bounds']) or rows.step not in (None,1) or np.asarray(key[1]).tolist()!=columns:
                raise PermissionError('Evaluation request crosses bound rows/columns')
            trace.append({'partition':'evaluation','row_start':rows.start,'row_stop_exclusive':rows.stop,'columns':columns})
        return original(dataset,key,*args,**kwargs)
    with patch.object(h5py.Dataset,'__getitem__',guarded): yield
