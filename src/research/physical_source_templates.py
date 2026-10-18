"""Original scoped physical plan -> exact response-template affine callback.

Only original exact support permits design reuse. Every q/h/d state and prior
remains in the caller's roster. Operational cache statistics do not enter input
identity, and no physical clock or observation permission is recreated here.
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from transfer.noise_model import array_digest, metadata_digest

IMPLEMENTATION_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class PhysicalTemplateResponse:
    def __init__(self,response,source,*,maximum_design_cache_bytes=64*1024**2,maximum_affine_cache_bytes=64*1024**2):
        self.response,self.source=response,source
        self.rank,self.channels,self.source_chart=source.rank,source.channels,source.source_chart
        self.scope=response.scope
        self.physical_capability_identity=response.plan.chart.capability_identity_sha256
        self.physical_target_lock=response.plan.binding['target_lock_group']
        self.physical_candidate_variable=response.candidate_variable
        self.physical_clock_program=response.plan.clock_program
        self.physical_shape_block_id=response.block_id
        self.physical_shape_variable=response.shape_variable
        self.physical_shape_parameters=tuple(tuple(vars(s).items()) for s in response.shapes)
        self.physical_exposure_ids=tuple(e['event_id'] for e in response.plan.exposures if e['block_id']==response.block_id)
        self.physical_unknown_identity_bindings={}
        for event in response.plan.exposures:
            if event['block_id']==response.block_id and event['membership'] is None:
                law=response.exposure_assignments
                self.physical_unknown_identity_bindings[event['event_id']]={'variable':law.variable_id,
                    'state_to_candidate':{state:assignment[event['event_id']] for state,assignment in law.assignments_by_state_id.items()}}
        self.identity=metadata_digest({'original_physical_plan':response.plan.identity,'block':response.block_id,
            'scope':self.scope,'source_chart':self.source_chart,'template_header':source.header,
            'physical_shape_parameters':self.physical_shape_parameters,
            'actual_unknown_identity_bindings':self.physical_unknown_identity_bindings,'adapter_producer':IMPLEMENTATION_SHA256})
        self.maximum_design_bytes,self.maximum_affine_bytes=maximum_design_cache_bytes,maximum_affine_cache_bytes
        self.design_cache,self.affine_cache,self.support_cache=OrderedDict(),OrderedDict(),{}
        self.design_bytes=self.affine_bytes=0
        self.design_hits=self.design_misses=self.affine_hits=self.affine_misses=0
        self.design_seconds=self.affine_seconds=0.
        self.original_generation_ledger_identities=[]

    def diagnostics(self):
        return {'design_hits':self.design_hits,'design_misses':self.design_misses,'affine_hits':self.affine_hits,
            'affine_misses':self.affine_misses,'design_cache_bytes':self.design_bytes,'affine_cache_bytes':self.affine_bytes,
            'design_seconds':self.design_seconds,'affine_seconds':self.affine_seconds,
            'original_generation_ledger_identities':self.original_generation_ledger_identities}

    def support(self,*,native_start,native_stop):
        return self.response.support(native_start=native_start,native_stop=native_stop)

    def _design(self,selected_states,start,stop):
        clock=time.perf_counter()
        response=self.response
        q=selected_states[response.candidate_variable].q
        h=selected_states[response.shape_variable].shape
        shape=response.shapes[h]
        support_key=(start,stop,h)
        if support_key not in self.support_cache:
            proof=response.plan.support(response.block_id,(shape,),native_start=start,native_stop=stop)
            self.support_cache[support_key]=(frozenset(row['event_id'] for row in proof['hulls']),metadata_digest(proof))
        active,support_identity=self.support_cache[support_key]
        assignments=({} if response.exposure_assignments is None else
            response.exposure_assignments.assignments_by_state_id[selected_states[response.exposure_assignments.variable_id].q])
        indicators=[]
        for event in response.plan.exposures:
            if event['block_id']!=response.block_id:
                continue
            included,_=response.plan.target_indicator(event,q,assignments)
            if event['event_id'] in active:
                indicators.append((event['event_id'],included))
        key=(start,stop,h,tuple(indicators))
        item=self.design_cache.pop(key,None)
        if item is None:
            self.design_misses+=1
            design,error,original=response.design(selected_states,native_start=start,native_stop=stop)
            self.original_generation_ledger_identities.append(metadata_digest(original))
            design.flags.writeable=error.flags.writeable=False
            generation={'original_plan':response.plan.identity,'numerical_error_role':original['numerical_error_role'],
                'response_shape_parameters':vars(shape),'native_half_open_rows':[start,stop],
                'numeric_design_and_error_sha256':array_digest(design,error),
                'exact_support_identity':support_identity,'active_event_indicators':indicators,
                'inactive_event_rule':'exact_original_half_open_support_zero_no_amplitude_threshold'}
            cost=design.nbytes+error.nbytes+len(str(key))*2
            item=(design,error,generation,cost)
            while self.design_cache and self.design_bytes+cost>self.maximum_design_bytes:
                _,old=self.design_cache.popitem(last=False)
                self.design_bytes-=old[3]
            if cost<=self.maximum_design_bytes:
                self.design_cache[key]=item
                self.design_bytes+=cost
        else:
            self.design_hits+=1
            self.design_cache[key]=item
        self.design_seconds+=time.perf_counter()-clock
        state_binding={'numeric_design':item[2],'selected_states':{name:vars(state) for name,state in selected_states.items()},
            'native_half_open_rows':[start,stop],'candidate':q,'shape_index':h,
            'actual_unknown_assignments':assignments,'original_response_scope':self.scope}
        return item[0],item[1],state_binding,key

    def _cache_affine(self,key,compile_image):
        value=self.affine_cache.pop(key,None)
        if value is None:
            self.affine_misses+=1
            result=compile_image()
            cost=result.values.nbytes+result.absolute_error.nbytes
            value=(result,cost)
            while self.affine_cache and self.affine_bytes+cost>self.maximum_affine_bytes:
                _,old=self.affine_cache.popitem(last=False)
                self.affine_bytes-=old[1]
            if cost<=self.maximum_affine_bytes:
                self.affine_cache[key]=value
                self.affine_bytes+=cost
        else:
            self.affine_hits+=1
            self.affine_cache[key]=value
        return value[0]

    def affine(self,selected_states,*,native_start,native_stop):
        clock=time.perf_counter()
        design,error,binding,_=self._design(selected_states,native_start,native_stop)
        numeric=binding['numeric_design']['numeric_design_and_error_sha256']
        key=('absolute',numeric)
        result=self._cache_affine(key,lambda:self.source.compile_temporal(design,temporal_error=error,
            query_binding={'original_physical_numeric_design':numeric,'native_half_open_rows':[native_start,native_stop]}))
        self.affine_seconds+=time.perf_counter()-clock
        return result.values,result.absolute_error,{'actual_physical_design':binding,'exact_source_image':result.binding,
            'numeric_affine_value_identity':metadata_digest({'source':self.source.header,'kind':'absolute','design':numeric}),
            'adapter_producer_sha256':IMPLEMENTATION_SHA256}

    def affine_difference(self,selected_states,reference_states,*,native_start,native_stop):
        clock=time.perf_counter()
        actual,actual_error,actual_binding,actual_key=self._design(selected_states,native_start,native_stop)
        reference,reference_error,reference_binding,reference_key=self._design(reference_states,native_start,native_stop)
        a=actual_binding['numeric_design']['numeric_design_and_error_sha256']
        b=reference_binding['numeric_design']['numeric_design_and_error_sha256']
        if actual_key==reference_key:
            error=np.zeros_like(actual_error)
        else:
            error=actual_error+reference_error
            error=np.where(error==0,0.,np.nextafter(error,np.inf))
        error_identity=array_digest(error)
        key=('difference',a,b,error_identity)
        result=self._cache_affine(key,lambda:self.source.compile_temporal_difference(actual,reference,temporal_error=error,
            query_binding={'actual_numeric_design':a,'reference_numeric_design':b,
                'native_half_open_rows':[native_start,native_stop],'operation':'exact_query_state_minus_reference'}))
        self.affine_seconds+=time.perf_counter()-clock
        return result.values,result.absolute_error,{'actual_physical_design':actual_binding,'reference_physical_design':reference_binding,
            'exact_source_image':result.binding,'same_physical_functional_error_cancels':actual_key==reference_key,
            'numeric_affine_value_identity':metadata_digest({'source':self.source.header,'kind':'difference',
                'actual':a,'reference':b,'error':error_identity}),'adapter_producer_sha256':IMPLEMENTATION_SHA256}
