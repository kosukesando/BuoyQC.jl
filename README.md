# BuoyQC

[![Build Status](https://github.com/kosukesando/BuoyQC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kosukesando/BuoyQC.jl/actions/workflows/CI.yml?query=branch%3Amain)

A library developed for de-spiking heave measurement from GPS-based buoys.

# MWE
`heave`: A vector of heave measurement time history in meters.
```julia
_, signal_treated = BuoyQC.despike_conv(heave)
```

# Spike detection
By default, `detect_spikes` chooses top 10% of convolution coefficient peaks as spikes unless `strength` is changed, or an integer is passed to `est_spikes` which overrides `strength`.

```julia
spikes = detect_spikes(heave;wl=wavelet(cSym4), fs=1.28, sup=10, hw=400, strength=0.1, est_spikes=nothing)
```
```julia
struct Spike <: TruncSignal
    offset::Integer
    wlcoeff::AbstractMatrix
    amp::AbstractFloat
end
```
