"""Exact-query reuse over the unchanged affine numerical reference owner."""
from __future__ import annotations

import hashlib
import sys
import time
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path

import numpy as np

from research.conditional_response_graph import exact, product, rows
from research.numeric_response_consumer import IMPLEMENTATION_SHA256 as REFERENCE_COMPILER_SHA256
from research.numeric_response_consumer import CompiledAffineImage, _round_rational
from transfer.noise_model import metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _query(functionals, dimension):
    raw = np.asarray(functionals, dtype=object)
    if raw.ndim != 3 or raw.shape[-1] != dimension or not raw.shape[0] or not raw.shape[1]:
        raise ValueError("actual functionals require sample/channel/response axes")
    return raw.shape, rows(raw.reshape(-1, dimension))


def _row_cost(key, arrays):
    return (sys.getsizeof(key) + sum(sys.getsizeof(v)+sys.getsizeof(v.numerator)+sys.getsizeof(v.denominator)
                                   for v in key) + sum(array.nbytes for array in arrays))


class _Rows:
    def __init__(self, maximum_bytes):
        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise ValueError("derived row cache requires a positive explicit byte budget")
        self.maximum_bytes = maximum_bytes
        self.data = OrderedDict()
        self.bytes = self.peak_bytes = self.hits = self.misses = self.evictions = 0

    def get(self, key):
        value = self.data.pop(key, None)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
            self.data[key] = value
        return value

    def put(self, key, values, errors):
        digest = metadata_digest([str(v) for v in key])
        cost = _row_cost(key, (values, errors))
        entry = (values.copy(), errors.copy(), digest, cost)
        if cost <= self.maximum_bytes:
            while self.data and self.bytes + cost > self.maximum_bytes:
                _, old = self.data.popitem(last=False)
                self.bytes -= old[3]
                self.evictions += 1
            self.data[key] = entry
            self.bytes += cost
            self.peak_bytes = max(self.bytes, self.peak_bytes)
        return entry

    def record(self):
        return {"rows":len(self.data),"hits":self.hits,"misses":self.misses,"evictions":self.evictions,
                "estimated_bytes":self.bytes,"peak_estimated_bytes":self.peak_bytes,"budget_bytes":self.maximum_bytes}


class ExactMeanResponseSource:
    """Query-first exact means, with no source-factor or conditional-LDL product."""

    def __init__(self, graph, *, maximum_cache_bytes=64*1024**2):
        self.graph = graph
        self.graph_digest = graph.digest()
        self.common_mean = tuple(row[0] for row in product(graph.source_map,
                                 tuple((exact(v),) for v in graph.source_mean)))
        common, rest = graph.common_rows, graph.completion_rows
        lifted = product(graph.conditional_gain, tuple((graph.reference_chart_mean[i],) for i in common)) if rest else ()
        self.reference_offset = tuple(graph.reference_chart_mean[i]-lifted[j][0] for j,i in enumerate(rest))
        self.cache = _Rows(maximum_cache_bytes)
        self.elapsed_seconds = 0.

    def compile_mean(self, functionals, *, query_binding):
        start = time.perf_counter()
        shape, query = _query(functionals, self.graph.dimension)
        values, errors = np.empty((len(query),1)), np.empty((len(query),1))
        digests = []
        common, rest = self.graph.common_rows, self.graph.completion_rows
        for i, key in enumerate(query):
            entry = self.cache.get(key)
            if entry is None:
                wc = tuple(key[j] for j in common)
                wr = tuple(key[j] for j in rest)
                adjustment = product((wr,), self.graph.conditional_gain)[0] if rest else (Fraction(),)*len(common)
                effective = tuple(a+b for a,b in zip(wc,adjustment,strict=True))
                value = sum((a*b for a,b in zip(effective,self.common_mean,strict=True)),Fraction())
                value += sum((a*b for a,b in zip(wr,self.reference_offset,strict=True)),Fraction())
                rounded, bound = _round_rational(value)
                entry = self.cache.put(key,np.array([rounded]),np.array([bound]))
            values[i],errors[i] = entry[:2]
            digests.append(entry[2])
        self.elapsed_seconds += time.perf_counter()-start
        binding = {"schema":"n2p3_exact_query_mean_only/1","graph_digest":self.graph_digest,
            "query_identity_scheme":"shape_and_exact_row_digest_sequence/1",
            "exact_query_sha256":metadata_digest({"shape":shape,"rows":digests}),"actual_query_binding":query_binding,
            "source_factor_projections":0,"conditional_innovation_products":0,"proper_latent_columns_compiled":0,
            "mean_formula":"exact_(Wcommon_plus_Wrest_B)_Ls_ms_plus_reference_intercept_then_correct_round",
            "numeric_producer_sha256":IMPLEMENTATION_SHA256,"cache":self.cache.record()}
        return values.reshape(shape[:2]+(1,)),errors.reshape(shape[:2]+(1,)),binding


class CachedAffineResponseSource:
    """Bounded memo of complete exact row queries; retains the original u chart."""

    def __init__(self, reference, *, maximum_cache_bytes=512*1024**2, missing_batch_rows=16):
        if type(missing_batch_rows) is not int or missing_batch_rows <= 0:
            raise ValueError("new exact query rows require a positive batch size")
        self.reference, self.graph = reference, reference.graph
        self.row_generator_sha256 = getattr(reference,"compiler_sha256",REFERENCE_COMPILER_SHA256)
        self.source_chart, self.rank, self.coordinate_ids = reference.source_chart, reference.rank, reference.coordinate_ids
        self.graph_digest = self.graph.digest()
        self.cache = _Rows(maximum_cache_bytes)
        self.mean_source = ExactMeanResponseSource(self.graph)
        self.missing_batch_rows = missing_batch_rows
        self.reference_compile_seconds = self.total_compile_seconds = 0.
        self.reference_scalar_rows_compiled = 0

    def compile_mean(self, functionals, *, query_binding):
        return self.mean_source.compile_mean(functionals,query_binding=query_binding)

    def compile(self, functionals, *, query_binding):
        start = time.perf_counter()
        shape, query = _query(functionals,self.graph.dimension)
        values, errors = np.empty((len(query),1+self.rank)), np.empty((len(query),1+self.rank))
        digests = [None]*len(query)
        pending = OrderedDict()
        for i,key in enumerate(query):
            entry = self.cache.get(key)
            if entry is None:
                pending.setdefault(key,[]).append(i)
            else:
                values[i],errors[i],digests[i] = entry[:3]
        missing = list(pending)
        for first in range(0,len(missing),self.missing_batch_rows):
            batch = missing[first:first+self.missing_batch_rows]
            clock = time.perf_counter()
            output = self.reference.compile(np.array(batch,dtype=object)[:,None,:],
                query_binding={"role":"exact_unique_row_memo_generation","parent_query":query_binding})
            self.reference_compile_seconds += time.perf_counter()-clock
            self.reference_scalar_rows_compiled += len(batch)
            for j,key in enumerate(batch):
                entry = self.cache.put(key,output.values[j,0],output.absolute_error[j,0])
                for i in pending[key]:
                    values[i],errors[i],digests[i] = entry[:3]
        self.total_compile_seconds += time.perf_counter()-start
        binding = {"schema":"n2p3_exact_query_affine_row_memo/1","graph_digest":self.graph_digest,
            "source_chart":self.source_chart,"proper_latent_dimensions":self.rank,
            "query_identity_scheme":"shape_and_exact_row_digest_sequence/1",
            "exact_query_sha256":metadata_digest({"shape":shape,"rows":digests}),"actual_query_binding":query_binding,
            "reference_numeric_producer_sha256":REFERENCE_COMPILER_SHA256,"numeric_producer_sha256":IMPLEMENTATION_SHA256,
            "row_generator_sha256":self.row_generator_sha256,
            "cache":self.cache.record(),"reference_scalar_rows_compiled":self.reference_scalar_rows_compiled,
            "reference_compile_seconds":self.reference_compile_seconds,"total_compile_seconds":self.total_compile_seconds,
            "semantics":"bit_identical_reference_values_and_errors_reused_only_for_equal_exact_functional_rows"}
        return CompiledAffineImage(values.reshape(shape[:2]+(1+self.rank,)),errors.reshape(shape[:2]+(1+self.rank,)),binding)
