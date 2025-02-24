# BuoyQC

[![Build Status](https://github.com/kosukesando/BuoyQC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kosukesando/BuoyQC.jl/actions/workflows/CI.yml?query=branch%3Amain)

A library developed for de-spiking heave measurement from GPS-based buoys.

# MWE
`heave`: A vector of heave measurement time history in meters.
```julia
_, signal_treated = BuoyQC.despike_conv(heave)
```
