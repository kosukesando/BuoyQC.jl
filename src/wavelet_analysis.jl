using Wavelets
using ContinuousWavelets
using Peaks
using Statistics
using Optim
using DataFrames
using Suppressor

const DEFAULT_WAVELET = wavelet(cSym4)

abstract type TruncSignal end

struct Spike <: TruncSignal
    offset::Integer
    wlcoeff::AbstractMatrix
    amp::AbstractFloat
end

struct Template <: TruncSignal
    wlcoeff::AbstractMatrix
    amp::Union{AbstractFloat,Nothing}
    width::Union{Integer,Nothing}
end

function center(trunc_signal::TruncSignal)
    n = size(trunc_signal.wlcoeff, 1)
    return (n + 1) ÷ 2
end

function halfwindow(trunc_signal::TruncSignal)
    n = size(trunc_signal.wlcoeff, 1)
    return (n - 1) ÷ 2
end

function _find_amplitude(spike::Spike, signal_length::Integer; wl=DEFAULT_WAVELET, amp_init=spike.amp)
    hw = halfwindow(spike)
    c = center(spike)
    _template = @suppress_err create_spike_template(signal_length, hw, 1.0, 1; wl=wl)
    template_amplitude(x) = calc_cost(spike, Template(_template.wlcoeff .* x, nothing, nothing))
    optres = @suppress_err optimize(template_amplitude, [amp_init], BFGS())
    return optres.minimizer[1]
end

function _find_offset(spike::Spike, template::Template, amp, init_offset::Integer; width=1, wl=DEFAULT_WAVELET)
    n = size(spike.wlcoeff, 1)
    start = max(init_offset - round(Int, n / 10), 1)
    finish = min(init_offset + round(Int, n / 10), n)
    costs = zeros(length(start:finish))
    ###### Use LineSearch for maybe better results
    for i in start:finish
        costs[i-start+1] = calc_cost(
            spike,
            shift_template(template, i - center(template));
            method=:naive
        )
    end
    return start - 1 + argmin(costs) - center(template), minimum(costs)
end

function _find_width(spike::Spike, signal_length::Integer, amp::AbstractFloat, offset::Integer; init_width::Integer=1, largest_width::Integer=30)
    widths = init_width:largest_width
    _cost_prev = typemax(Float64)
    for i in eachindex(widths)
        _cost = calc_cost(
            spike,
            shift_template(create_spike_template(signal_length, halfwindow(spike), amp, widths[i]), offset)
        )
        if _cost < _cost_prev
            _cost_prev = _cost
        elseif i == lastindex(widths)
            return widths[i]
        else
            return widths[i-1]
        end
    end
    return init_width
end

function create_spike_template(signal_length::Integer, hw::Integer, amp, width; f0=0.01, fs=1.28, wl=DEFAULT_WAVELET)
    x = -(signal_length - 1)÷2:signal_length÷2
    y = highpass(hsigmoid.(x, amp, width); f0=f0, fs=fs)
    wlcoeff = cwt(y, wl)
    wlcoeff_trunc = _truncate_template(wlcoeff, hw)
    return Template(wlcoeff_trunc, amp, width)
end

function _truncate_template(template_cwt::AbstractMatrix, hw::Integer)
    n = size(template_cwt)[1]
    c = (n + 1) ÷ 2
    return template_cwt[c-hw:c+hw, :]
end

function shift_template(template::Template, offset)
    wlcoeff_shifted = circshift(template.wlcoeff, (offset, 0))
    if offset ≥ 0
        wlcoeff_shifted[1:offset, :] .= 0
    else
        wlcoeff_shifted[end+offset:end, :] .= 0
    end
    return Template(wlcoeff_shifted, template.amp, template.width)
end

function est_template(spike::Spike, signal_length; wl=DEFAULT_WAVELET, max_width=5)
    hw = halfwindow(spike)
    amp = _find_amplitude(spike, signal_length; wl=wl)
    template = create_spike_template(signal_length, hw, amp, 1; wl=wl)
    offset = nothing
    width = nothing
    cost = typemax(Float64)
    for _width in 1:max_width
        _offset, _cost = _find_offset(spike, template, amp, center(spike); wl=wl, width=_width)
        if _cost < cost
            cost = _cost
            offset = _offset
            width = _width
        end
    end
    template_fitted = shift_template(create_spike_template(signal_length, hw, amp, width; wl=wl), offset)
    if calc_cost(spike, template_fitted) > 1
        return create_spike_template(signal_length, hw, 0, width; wl=wl)
    end
    return template_fitted
end

function calc_cost(spike::Spike, template::Template; method=:lower)
    @assert size(spike.wlcoeff) == size(template.wlcoeff) "$(size(spike.wlcoeff)), $(size(template.wlcoeff))"
    ## Naive subtraction
    if method === :naive
        diff = abs.(spike.wlcoeff .- template.wlcoeff)
        cost = mean(diff) / mean(abs.(spike.wlcoeff))
    elseif method === :lower
        lrange = 1:size(spike.wlcoeff, 2)÷2
        diff = abs.(spike.wlcoeff[:, lrange] .- template.wlcoeff[:, lrange])
        cost = mean(diff) / mean(abs.(spike.wlcoeff[:, lrange]))
    else
        ArgumentError("No such method as $(method)")
    end
    return cost
end