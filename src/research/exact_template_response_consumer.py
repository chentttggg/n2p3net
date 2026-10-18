"""Integer contractions of exact physical row templates, with shared latent u.

Templates are exact rational operators for the explicitly requested spatial map.
No rounded coefficient mean/factor defines the response law. Numerical entries
are rounded only after the actual temporal functional has been contracted.
"""
from __future__ import annotations

import hashlib
import math
import time
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path

import numpy as np
from research.cached_response_consumer import ExactMeanResponseSource, _Rows
from research.conditional_response_graph import exact, product, rows, string_rows
from research.numeric_response_consumer import CompiledAffineImage, _round_rational, _round_surd
from transfer.noise_model import array_digest, metadata_digest

IMPLEMENTATION_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _integer_columns(matrix):
    width=len(matrix[0])
    denominators=[]
    columns=[]
    for j in range(width):
        denominator=math.lcm(*(row[j].denominator for row in matrix))
        denominators.append(denominator)
        columns.append(tuple(row[j].numerator*(denominator//row[j].denominator) for row in matrix))
    return np.array(columns,dtype=object).T,tuple(denominators)


def _integer_row(row):
    denominator=math.lcm(*(v.denominator for v in row))
    return np.array([v.numerator*(denominator//v.denominator) for v in row],dtype=object),denominator


class ExactTemplateAffineSource:
    def __init__(self,reference,*,native_spatial_map,observed_channel_ids,maximum_cache_bytes=512*1024**2):
        start=time.perf_counter()
        self.reference,self.graph=reference,reference.graph
        self.rank,self.source_chart,self.coordinate_ids=reference.rank,reference.source_chart,reference.coordinate_ids
        self.compiler_sha256=IMPLEMENTATION_SHA256
        self.spatial=rows(native_spatial_map)
        self.channel_ids=tuple(observed_channel_ids)
        if (not self.spatial or len(self.spatial)!=len(self.channel_ids)
                or any(len(row)!=self.graph.chart.native_dimension for row in self.spatial)):
            raise ValueError('physical templates require an explicit complete native spatial map and output IDs')
        self.channels=len(self.spatial)
        self.spatial_chart=product(self.spatial,self.graph.chart.exact_inverse)
        self.pivots=tuple(next((j for j,v in enumerate(row) if v),None) for row in self.spatial_chart)
        self.coefficients=self.graph.coefficient_count
        self.source_rank=getattr(reference,'source_rank',self.graph.source_factor.shape[1])
        self.common_diagonal=getattr(reference,'common_diagonal',None)
        self.graph_digest=self.graph.digest()
        common,rest=self.graph.common_rows,self.graph.completion_rows
        common_image=(reference.common_factor if self.common_diagonal is not None else
            product(self.graph.source_map,self.graph.source_factor))
        common_mean=tuple(row[0] for row in product(self.graph.source_map,tuple((exact(v),) for v in self.graph.source_mean)))
        lifted=product(self.graph.conditional_gain,tuple((self.graph.reference_chart_mean[i],) for i in common)) if rest else ()
        offset=tuple(self.graph.reference_chart_mean[i]-lifted[j][0] for j,i in enumerate(rest))
        self.integers,self.denominators=[],[]
        self.absolute_template_bounds=np.zeros((self.channels,self.coefficients,1+self.rank))
        for channel,spatial in enumerate(self.spatial_chart):
            templates=[]
            for coefficient in range(self.coefficients):
                w=tuple(value if block==coefficient else Fraction() for block in range(self.coefficients) for value in spatial)
                wc,wr=tuple(w[i] for i in common),tuple(w[i] for i in rest)
                adjustment=product((wr,),self.graph.conditional_gain)[0] if rest else (Fraction(),)*len(common)
                effective=tuple(a+b for a,b in zip(wc,adjustment,strict=True))
                mean=sum((a*b for a,b in zip(effective,common_mean,strict=True)),Fraction())
                mean+=sum((a*b for a,b in zip(wr,offset,strict=True)),Fraction())
                source=product((effective,),common_image)[0]
                conditional=product((wr,),reference.lower)[0] if rest else ()
                template=(mean,*source,*conditional)
                templates.append(template)
                for j,value in enumerate(template):
                    rounded,error=self._round_entry(j,abs(value))
                    upper=Fraction.from_float(rounded)+Fraction.from_float(error)
                    bound=float(upper)
                    if Fraction.from_float(bound)<upper:
                        bound=float(np.nextafter(bound,np.inf))
                    self.absolute_template_bounds[channel,coefficient,j]=bound
            integer,denominators=_integer_columns(templates)
            self.integers.append(integer)
            self.denominators.append(denominators)
        self.cache=_Rows(maximum_cache_bytes)
        self.mean_source=ExactMeanResponseSource(self.graph)
        self.header=metadata_digest({'graph':self.graph_digest,'source_chart':self.source_chart,
            'native_spatial_map':string_rows(self.spatial),'observed_channel_ids':self.channel_ids,
            'compiler':IMPLEMENTATION_SHA256,'arithmetic':'exact_rational_templates_integer_dot_then_original_correct_round'})
        self.setup_seconds=time.perf_counter()-start
        self.compile_seconds=self.integer_dot_seconds=self.round_seconds=0.
        self.scalar_rows_generated=self.fallback_calls=0
        self.temporal_rows=OrderedDict()
        self.temporal_differences=OrderedDict()
        self.temporal_row_cache_limit=20000
        self.temporal_row_hits=self.temporal_row_misses=0

    def _round_entry(self,index,value):
        if index==0 or index<=self.source_rank and self.common_diagonal is None:
            return _round_rational(value)
        diagonal=(self.common_diagonal[index-1] if index<=self.source_rank else
                  self.reference.diagonal[index-1-self.source_rank])
        return _round_surd(value,diagonal)

    def diagnostics(self):
        return {'setup_seconds':self.setup_seconds,'compile_seconds':self.compile_seconds,
            'integer_dot_seconds':self.integer_dot_seconds,'round_seconds':self.round_seconds,
            'scalar_rows_generated':self.scalar_rows_generated,'fallback_calls':self.fallback_calls,
            'temporal_row_hits':self.temporal_row_hits,'temporal_row_misses':self.temporal_row_misses,
            'temporal_row_records':len(self.temporal_rows),'temporal_difference_records':len(self.temporal_differences),
            'cache':self.cache.record()}

    @staticmethod
    def _row_record(numeric,denominator):
        divisor=denominator
        for value in numeric:
            divisor=math.gcd(divisor,int(value))
        if divisor!=1:
            denominator//=divisor
            numeric=np.array([int(value)//divisor for value in numeric],dtype=object)
        token=int(metadata_digest({'numerators':[str(int(v)) for v in numeric],'denominator':str(denominator)}),16)
        return numeric,denominator,token

    def _remember(self,cache,key,value):
        cache[key]=value
        if len(cache)>self.temporal_row_cache_limit:
            cache.popitem(last=False)
        return value

    def _parse_row(self,row):
        key=row.tobytes() if row.dtype.kind=='f' else tuple(exact(v) for v in row)
        value=self.temporal_rows.pop(key,None)
        if value is not None:
            self.temporal_row_hits+=1
            self.temporal_rows[key]=value
            return value
        self.temporal_row_misses+=1
        rational=tuple(exact(v) for v in row)
        numeric,denominator=_integer_row(rational)
        return self._remember(self.temporal_rows,key,self._row_record(numeric,denominator))

    def compile_mean(self,functionals,*,query_binding):
        values,errors,binding=self.mean_source.compile_mean(functionals,query_binding=query_binding)
        stable={k:v for k,v in binding.items() if k!='cache'}
        return values,errors,stable

    def compile_temporal(self,temporal,*,query_binding,temporal_error=None,_row_records=None):
        start=time.perf_counter()
        design=np.asarray(temporal)
        if design.ndim!=2 or not len(design) or design.shape[1]!=self.coefficients:
            raise ValueError('temporal templates require the complete source coefficient groups')
        values=np.empty((len(design),self.channels,1+self.rank))
        errors=np.empty_like(values)
        identities=[]
        records=[self._parse_row(row) for row in design] if _row_records is None else _row_records
        for i,(numeric,denominator,token) in enumerate(records):
            row_ids=[]
            for channel in range(self.channels):
                key=(channel,token)
                entry=self.cache.get(key)
                if entry is None:
                    clock=time.perf_counter()
                    numerators=numeric@self.integers[channel]
                    self.integer_dot_seconds+=time.perf_counter()-clock
                    clock=time.perf_counter()
                    value,bound=np.empty(1+self.rank),np.empty(1+self.rank)
                    for j,num in enumerate(numerators):
                        coefficient=Fraction(int(num),denominator*self.denominators[channel][j])
                        value[j],bound[j]=self._round_entry(j,coefficient)
                    self.round_seconds+=time.perf_counter()-clock
                    self.scalar_rows_generated+=1
                    entry=self.cache.put(key,value,bound)
                values[i,channel],errors[i,channel]=entry[:2]
                row_ids.append(entry[2])
            identities.append(row_ids)
        arithmetic_error_sha=array_digest(errors)
        if temporal_error is not None:
            error=np.asarray(temporal_error,dtype=float)
            if error.shape!=(len(design),self.coefficients) or not np.isfinite(error).all() or np.any(error<0):
                raise ValueError('original temporal design error differs from its complete coefficient chart')
            if np.any(error):
                active=np.any(error!=0,axis=1)
                propagated=np.einsum('nk,ckr->ncr',error,self.absolute_template_bounds,optimize=False)
                gamma=self.coefficients*np.finfo(float).eps/(1-self.coefficients*np.finfo(float).eps)
                propagated=np.nextafter((propagated+self.coefficients*np.nextafter(0.,1.))/(1-gamma),np.inf)
                propagated[~active]=0
                errors=np.nextafter((errors+propagated)/(1-np.finfo(float).eps),np.inf)
            design_error_sha=array_digest(error)
        else:
            design_error_sha=None
        self.compile_seconds+=time.perf_counter()-start
        binding={'schema':'n2p3_exact_physical_template_affine/1','source_chart':self.source_chart,
            'graph_digest':self.graph_digest,'proper_latent_dimensions':self.rank,'template_header':self.header,
            'actual_query_binding':query_binding,'query_identity_scheme':'ordered_temporal_rows_and_explicit_native_spatial_map/1',
            'exact_temporal_row_identity':'sha256_canonical_shared_denominator_integer_row/1',
            'exact_query_sha256':metadata_digest({'template_header':self.header,'row_ids':identities}),
            'arithmetic_error_sha256':arithmetic_error_sha,'temporal_design_error_sha256':design_error_sha,
            'numerical_producer_sha256':IMPLEMENTATION_SHA256,
            'source_latents_removed':self.graph.source_factor.shape[1]-self.source_rank,'conditional_latents_removed':0,
            'error_role':'outward_rounding_entry_error_plus_propagation_of_original_declared_temporal_error',
            'coefficient_mean_or_factor_rounded_before_query':False}
        return CompiledAffineImage(values,errors,binding)

    def compile_temporal_difference(self,actual,reference,*,query_binding,temporal_error=None):
        actual,reference=np.asarray(actual),np.asarray(reference)
        if actual.shape!=reference.shape or actual.ndim!=2 or actual.shape[1]!=self.coefficients:
            raise ValueError('actual/reference temporal functions differ in their original coefficient chart')
        records=[]
        for left,right in zip(actual,reference,strict=True):
            key=(left.tobytes(),right.tobytes()) if left.dtype.kind==right.dtype.kind=='f' else (
                tuple(exact(v) for v in left),tuple(exact(v) for v in right))
            value=self.temporal_differences.pop(key,None)
            if value is None:
                a,da,_=self._parse_row(left)
                b,db,_=self._parse_row(right)
                denominator=math.lcm(da,db)
                numeric=a*(denominator//da)-b*(denominator//db)
                value=self._row_record(numeric,denominator)
                self._remember(self.temporal_differences,key,value)
            else:
                self.temporal_differences[key]=value
            records.append(value)
        return self.compile_temporal(actual,query_binding=query_binding,temporal_error=temporal_error,_row_records=records)

    def compile(self,functionals,*,query_binding):
        raw=np.asarray(functionals,dtype=object)
        if raw.ndim!=3 or raw.shape[1:]!=(self.channels,self.graph.dimension) or None in self.pivots:
            self.fallback_calls+=1
            return self.reference.compile(functionals,query_binding=query_binding)
        temporal=[]
        for sample in raw:
            row=[]
            for coefficient in range(self.coefficients):
                block=tuple(exact(v) for v in sample[0,coefficient*self.graph.chart.native_dimension:(coefficient+1)*self.graph.chart.native_dimension])
                value=block[self.pivots[0]]/self.spatial_chart[0][self.pivots[0]]
                row.append(value)
            for channel,spatial in enumerate(self.spatial_chart):
                for coefficient,value in enumerate(row):
                    block=sample[channel,coefficient*self.graph.chart.native_dimension:(coefficient+1)*self.graph.chart.native_dimension]
                    if any(exact(v)!=value*a for v,a in zip(block,spatial,strict=True)):
                        self.fallback_calls+=1
                        return self.reference.compile(functionals,query_binding=query_binding)
            temporal.append(tuple(row))
        return self.compile_temporal(temporal,query_binding=query_binding)
