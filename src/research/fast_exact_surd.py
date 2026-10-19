"""Correct binary64 rounding of rational surds with integer comparisons.

Avoid constructing and repeatedly reducing the large rational square.  Its
unreduced integer numerator and denominator suffice for floor and midpoint.
The outward error is half the exact binade spacing, or one minimum subnormal.
"""
from __future__ import annotations

import hashlib
import math
from fractions import Fraction
from pathlib import Path

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def round_surd_integer(coefficient,radicand):
    if not coefficient:
        return 0.,0.
    if radicand<=0:
        raise ValueError('positive exact surd radicand required')
    numerator=coefficient.numerator**2*radicand.numerator
    denominator=coefficient.denominator**2*radicand.denominator
    exponent=(numerator.bit_length()-denominator.bit_length())//2
    if (numerator<denominator<<(2*exponent) if exponent>=0 else
            numerator<<(-2*exponent)<denominator):
        exponent-=1
    spacing=max(exponent-52,-1074)
    if spacing<0:
        scaled_numerator,scaled_denominator=numerator<<(-2*spacing),denominator
    else:
        scaled_numerator,scaled_denominator=numerator,denominator<<(2*spacing)
    lower=math.isqrt(scaled_numerator//scaled_denominator)
    comparison=4*scaled_numerator-(2*lower+1)**2*scaled_denominator
    nearest=lower+int(comparison>0 or comparison==0 and lower%2==1)
    try:
        rounded=math.ldexp(float(nearest),spacing)
    except OverflowError as error:
        raise ValueError('actual Gaussian functional is outside binary64 range') from error
    if not math.isfinite(rounded):
        raise ValueError('actual Gaussian functional is outside binary64 range')
    rational=Fraction.from_float(rounded)
    if numerator*rational.denominator**2==rational.numerator**2*denominator:
        error=0.
    else:
        error=math.ldexp(1.,spacing-1) if spacing>-1074 else math.ldexp(1.,-1074)
    return (-rounded if coefficient<0 else rounded),error
