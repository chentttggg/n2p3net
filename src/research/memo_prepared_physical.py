"""Reuse identical scoped white images/dots while retaining every roster cell.

The original raw, noise, flat-measure and Gaussian owners remain unchanged.
Multiple-response segments use the original compiler; the local optimization
applies to one complete response scope of arbitrary candidate/state widths.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from research.physical_affine_frontier import PreparedPhysicalSegment, _white
from research.prepared_continuous_frontier import PreparedEndpoint, QuadraticFactor
from transfer.noise_model import array_digest, metadata_digest

IMPLEMENTATION_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _immutable_bytes_backing(array):
    owner=array
    while isinstance(owner,np.ndarray):
        if owner.flags.writeable:
            return False
        owner=owner.base
    return isinstance(owner,bytes)


class MemoPreparedPhysicalSegment(PreparedPhysicalSegment):
    def __init__(self,*args,**kwargs):
        self.performance_diagnostics={}
        super().__init__(*args,**kwargs)

    def _identity(self):
        return metadata_digest({'original_owner_numeric_identity':super()._identity(),
            'memoized_compiler_producer_sha256':IMPLEMENTATION_SHA256,
            'law':'same_complete_state_cells_original_white_and_dot_operations_reused_on_identical_inputs'})

    def _compile(self,base,stop,selections,pairs):
        if len(selections)!=1 or pairs or not hasattr(selections[0][0],'affine_difference'):
            self.performance_diagnostics[stop]={'route':'original_compatible_response_compiler'}
            return super()._compile(base,stop,selections,pairs)
        started=clock=time.perf_counter()
        d,m,c=self.dimension+1,self.rank+1,self.channels
        common=np.zeros((d,d))
        row=selections[0]
        unary=np.zeros((*row[2],d,d))
        table=unary.reshape(-1,d,d)
        flat_root,ledger,maximum_error=np.empty((0,2*c)),[],np.zeros(d)
        prefix,order=base.carrier.noise.conditioning_samples,base.carrier.noise.order
        possible_white=actual_white=actual_gram=0
        callback_seconds=white_seconds=dot_seconds=0.
        for begin in range(0,stop,self.block_rows):
            end=min(stop,begin+self.block_rows)
            start=0 if begin<prefix+order else begin-order
            raw,raw_error=np.zeros((end-start,c,d)),np.zeros((end-start,c,d))
            raw[:,:,0]=base.y[start:end]
            nuisance,nuisance_error=base.nuisance(start,end)
            raw[:,:,m:],raw_error[:,:,m:]=-nuisance,nuisance_error
            active=row[4]<end and row[5]>begin
            query_bindings=[]
            reference_bytes=0
            if active:
                clock=time.perf_counter()
                reference,error,binding=row[0].affine(row[3][0],native_start=base.origin+start,native_stop=base.origin+end)
                callback_seconds+=time.perf_counter()-clock
                raw[:,:,:m]-=reference
                raw_error[:,:,:m]+=error
                reference_bytes=reference.nbytes+error.nbytes
                query_bindings.append(metadata_digest(binding))
            clock=time.perf_counter()
            b,error,owned=_white(base,raw,raw_error,start,begin,end)
            white_seconds+=time.perf_counter()-clock
            maximum_error=np.maximum(maximum_error,error)
            clock=time.perf_counter()
            common-=.5*(b.T@b)
            flat_root=np.linalg.qr(np.vstack((flat_root,b[:,m:])),mode='reduced')[1]
            dot_seconds+=time.perf_counter()-clock
            local,immutable_objects={},{}
            scope_error=np.zeros(m)
            if active:
                for state_index,state in enumerate(row[3]):
                    possible_white+=1
                    clock=time.perf_counter()
                    difference,err,binding=row[0].affine_difference(state,row[3][0],
                        native_start=base.origin+start,native_stop=base.origin+end)
                    callback_seconds+=time.perf_counter()-clock
                    object_key=(id(difference),id(err))
                    object_value=immutable_objects.get(object_key)
                    if object_value is None:
                        key=array_digest(difference,err)
                        if _immutable_bytes_backing(difference) and _immutable_bytes_backing(err):
                            immutable_objects[object_key]=(difference,err,key)
                    else:
                        key=object_value[2]
                    cached=local.get(key)
                    if cached is None:
                        clock=time.perf_counter()
                        value,white_error,response_owned=_white(base,difference,err,start,begin,end)
                        white_seconds+=time.perf_counter()-clock
                        clock=time.perf_counter()
                        cross=b.T@value
                        gram=value.T@value
                        dot_seconds+=time.perf_counter()-clock
                        cached=(value,white_error,response_owned,cross,gram)
                        local[key]=cached
                        actual_white+=1
                        actual_gram+=2
                    value,white_error,response_owned,cross,gram=cached
                    scope_error=np.maximum(scope_error,white_error)
                    query_bindings.append(metadata_digest({'affine':binding,'white':response_owned}))
                    table[state_index,:,:m]+=.5*cross
                    table[state_index,:m,:]+=.5*cross.T
                    table[state_index,:m,:m]-=.5*gram
            ledger.append({'native_block':[base.origin+begin,base.origin+end],
                'raw_halo_start':base.origin+start,'original_white_map':owned,
                'affine_query_bindings':metadata_digest(query_bindings),
                'maximum_white_affine_column_error_by_scope':{'0':scope_error.tolist()} if active else {}})
            self.maximum_named_workspace_bytes=max(self.maximum_named_workspace_bytes,
                raw.nbytes+raw_error.nbytes+b.nbytes+reference_bytes+
                sum(v[0].nbytes+v[3].nbytes+v[4].nbytes for v in local.values()))
        self.flat_roots[stop]=flat_root
        self.compilation_ledgers[stop]={'blocks':ledger,'maximum_white_common_column_error':maximum_error.tolist(),
            'error_role':'original_affine_and_noise_errors_carried; binary64_Q_dot_and_integral_reference',
            'extra_observation_rows':0,'noise_startups':1 if stop else 0}
        normalizer=base.carrier.leading_log_jacobian(stop)-.5*stop*c*np.log(2*np.pi) if stop else 0.
        self.performance_diagnostics[stop]={'route':'same_input_white_and_gram_reuse_all_scope_states_retained',
            'wall_seconds':time.perf_counter()-started,'callback_seconds':callback_seconds,'white_seconds':white_seconds,
            'dot_seconds':dot_seconds,'state_white_calls_without_memo':possible_white,'state_white_calls_executed':actual_white,
            'state_gram_products_executed':actual_gram,'complete_scope_cells':len(row[3]),'proper_latents':self.rank}
        return PreparedEndpoint(stop,common,(QuadraticFactor(row[1],unary,False),),float(normalizer))

    def save(self,path):
        metadata=super().save(path)
        metadata['memoized_compiler_sha256']=IMPLEMENTATION_SHA256
        metadata['performance_diagnostics']=self.performance_diagnostics
        Path(path).with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
        return metadata

    @classmethod
    def load(cls,metadata_path,*,expected_metadata_sha256):
        metadata=json.loads(Path(metadata_path).read_bytes())
        if metadata.get('memoized_compiler_sha256')!=IMPLEMENTATION_SHA256:
            raise ValueError('memoized physical compiler differs from its original executed producer')
        model=super().load(metadata_path,expected_metadata_sha256=expected_metadata_sha256)
        model.performance_diagnostics=metadata.get('performance_diagnostics',{})
        return model
