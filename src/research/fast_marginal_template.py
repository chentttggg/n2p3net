"""Exact common marginal template with integer-only surd rounding."""
from __future__ import annotations

import hashlib
import math
from copy import copy
from fractions import Fraction
from pathlib import Path

from research.exact_template_response_consumer import ExactTemplateAffineSource
from research.fast_exact_surd import IMPLEMENTATION_SHA256 as ROUNDING_SHA256
from research.fast_exact_surd import round_surd_integer
from research.numeric_response_consumer import CompiledAffineImage
from transfer.noise_model import metadata_digest

IMPLEMENTATION_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class IntegerSurdMarginalTemplate(ExactTemplateAffineSource):
    def __init__(self,reference,**kwargs):
        columns=tuple(zip(*reference.common_factor,strict=True))
        scales=tuple(math.lcm(*(v.denominator for v in column)) for column in columns)
        scaled=copy(reference)
        scaled.common_factor=tuple(tuple(Fraction(v*scales[j]) for j,v in enumerate(row))
            for row in reference.common_factor)
        scaled.common_diagonal=tuple(d/scale**2 for d,scale in zip(reference.common_diagonal,scales,strict=True))
        if (any(v.denominator!=1 for row in scaled.common_factor for v in row)
                or any(d*scale**2!=old for d,scale,old in
                       zip(scaled.common_diagonal,scales,reference.common_diagonal,strict=True))):
            raise ArithmeticError('exact common marginal column denominators were not cleared')
        super().__init__(scaled,**kwargs)
        if self.common_diagonal is None:
            raise ValueError('this compiler requires the explicit exact common marginal chart')
        self.compiler_sha256=IMPLEMENTATION_SHA256
        self.header=metadata_digest({'original_exact_template_header':self.header,
            'marginal_template_producer':IMPLEMENTATION_SHA256,'integer_surd_rounding':ROUNDING_SHA256,
            'exact_column_scales':[str(v) for v in scales],
            'identity':'L_column_times_s_and_D_divided_by_s_squared_same_standard_Gaussian_chart'})
        self.exact_column_scale_identity=metadata_digest([str(v) for v in scales])

    def _round_entry(self,index,value):
        if index and index<=self.source_rank and self.common_diagonal is not None:
            return round_surd_integer(value,self.common_diagonal[index-1])
        return super()._round_entry(index,value)

    def compile_temporal(self,*args,**kwargs):
        image=super().compile_temporal(*args,**kwargs)
        binding={**dict(image.binding),'marginal_template_producer_sha256':IMPLEMENTATION_SHA256,
            'integer_surd_producer_sha256':ROUNDING_SHA256,
            'exact_column_denominator_clearing_identity':self.exact_column_scale_identity,
            'source_chart_preserved_by_exact_column_normalization':True,
            'common_surd_error_role':'half_exact_binade_spacing_or_minimum_subnormal_outward_entry_bound'}
        return CompiledAffineImage(image.values,image.absolute_error,binding)
