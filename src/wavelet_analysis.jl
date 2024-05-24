using Wavelets
using ContinuousWavelets
using Peaks
using Statistics
using Optim
using DataFrames

const DEFAULT_WAVELET = wavelet(cSym4)

abstract type TruncSignal end

struct Spike <: TruncSignal
    offset::Integer
    res::AbstractMatrix
    amp::AbstractFloat
end

struct Template <: TruncSignal
    res::AbstractMatrix
end

function center(trunc_signal::TruncSignal)
    n = size(trunc_signal.res, 1)
    return (n + 1) ÷ 2
end

function halfwindow(trunc_signal::TruncSignal)
    n = size(trunc_signal.res, 1)
    return (n - 1) ÷ 2
end

function detect_spikes(signal; method::Symbol=:lower, hw=200, wl=DEFAULT_WAVELET, sr=1.28, prom=2.0, zeroupcross=true)
    signal_cwt = cwt(signal, wl)
    pks = zeroupcross ? _detect_spikes_zuc(signal, signal_cwt, method; wl=wl, sr=sr, prom=prom) : _detect_spikes_zdc(signal, signal_cwt, method; wl=wl, sr=sr, prom=prom)
    pks = unique(pks)
    spikes = Vector{Spike}()
    for pk in pks
        if pk - hw > firstindex(signal) && pk + hw < lastindex(signal)
            res_trunc = signal_cwt[pk-hw:pk+hw, :]
            _signal_trunc = signal[pk-10:pk+10]
            amp = (maximum(_signal_trunc) - minimum(_signal_trunc)) / 2
            push!(spikes, Spike(pk, res_trunc, amp))
        end
    end
    return spikes
end

function _detect_spikes_zuc(signal::AbstractVector, signal_cwt, method::Symbol; wl=DEFAULT_WAVELET, sr=1.28, prom=2.0)
    n = length(signal)
    # zuc spikes have neg-pos structure
    pks_neg_pre = _detect_spikes_cwt(signal_cwt, :neg, method; wl=wl, sr=sr, prom=prom)
    pks_neg = []
    for i in eachindex(pks_neg_pre)
        pk = _find_next_upcross(signal, pks_neg_pre[i])
        if !isnothing(pk)
            push!(pks_neg, pk)
        end
    end
    pks_pos_pre = _detect_spikes_cwt(signal_cwt, :pos, method; wl=wl, sr=sr, prom=prom)
    pks_pos = []
    for i in eachindex(pks_pos_pre)
        pk = _find_prev_upcross(signal, pks_pos_pre[i])
        if !isnothing(pk)
            push!(pks_pos, pk)
        end
    end
    pks = []
    for pp in pks_pos
        for pn in pks_neg
            if pp - pn < 5 && pp > pn
                push!(pks, (pp + pn) ÷ 2)
                break
            end
        end
    end
    return pks
end

function _detect_spikes_zdc(signal::AbstractVector, signal_cwt, method::Symbol; wl=DEFAULT_WAVELET, sr=1.28, prom=2.0)
    signal_cwt = cwt(signal, wl)
    n = length(signal)
    # zdc spikes have pos-neg structure
    pks_pos_pre = _detect_spikes_cwt(signal_cwt, :pos, method; wl=wl, sr=sr, prom=prom)
    pks_pos = []
    for i in eachindex(pks_pos_pre)
        pk = _find_next_dncross(signal, pks_pos_pre[i])
        if !isnothing(pk)
            push!(pks_pos, pk)
        end
    end
    pks_neg_pre = _detect_spikes_cwt(signal_cwt, :neg, method; wl=wl, sr=sr, prom=prom)
    pks_neg = []
    for i in eachindex(pks_neg_pre)
        pk = _find_prev_dncross(signal, pks_neg_pre[i])
        if !isnothing(pk)
            push!(pks_neg, pk)
        end
    end
    pks = unique([pks_pos; pks_neg])
    return pks
end

function _detect_spikes_cwt(signal_cwt, sign::Symbol, method::Symbol; mask=nothing, wl=DEFAULT_WAVELET, sr=1.28, prom=2.0)
    n = size(signal_cwt, 1)
    freqs = getMeanFreq(computeWavelets(n, wl)[1], sr)
    freqs[1] = 0
    if method === :lower
        res_fmean = vec(sum(signal_cwt[:, 1:length(freqs)÷2], dims=2))
    elseif method === :upper
        res_fmean = vec(sum(signal_cwt[:, length(freqs)÷2:end], dims=2))
    elseif method === :spectra
        if isnothing(mask)
            _tempmean = mean(abs.(signal_cwt), dims=1)
            mask = 1.0 .- _tempmean ./ maximum(_tempmean)
        end
        res_fmean = vec(sum(signal_cwt .* mask, dims=2))
        prom = 4 * std(res_fmean)
    else
        ArgumentError("No spike detection method of name $(method)")
    end

    if sign === :pos
        pks, vals = findmaxima(res_fmean)
    elseif sign === :neg
        pks, vals = findminima(res_fmean)
    else
        ArgumentError("Specify :pos or :neg")
    end
    pks, proms = peakproms!(pks, res_fmean, min=prom)
    return pks
end

function _find_amplitude(spike::Spike, signal_length::Integer; wl=DEFAULT_WAVELET)
    hw = halfwindow(spike)
    c = center(spike)
    _template = create_spike_template(signal_length, hw, 1.0, 1; wl=wl)
    template_amplitude(x) = calc_cost(spike, Template(_template.res .* x))

    # abs(maximum(spike.res[c-10:c+10, :]) - maximum(
    #     _template.res[c-10:c+10, :].*x
    # ))
    # optres = optimize(template_amplitude, [spike.amp], BFGS())
    # return optres.minimizer[1]

    return maximum(spike.res[c-10:c+10, 1:size(spike.res, 2)÷2]) / maximum(
        create_spike_template(signal_length, hw, 1, 1; wl=wl).res[c-10:c+10, 1:size(spike.res, 2)÷2]
    )
end

function _find_offset(spike::Spike, template::Template, amp, init_offset::Integer; width=1, wl=DEFAULT_WAVELET)
    n = size(spike.res, 1)
    start = max(init_offset - round(Int, n / 10), 1)
    finish = min(init_offset + round(Int, n / 10), n)
    costs = Vector{Float64}(undef, length(start:finish))
    ###### Use LineSearch for maybe better results
    for i in start:finish
        costs[i-start+1] = calc_cost(
            spike,
            _shift_template(template, i - center(template))
        )
    end
    return start - 1 + argmin(costs) - center(template), costs
end

function _find_width(spike::Spike, signal_length::Integer, amp::AbstractFloat, offset::Integer; init_width::Integer=1, largest_width::Integer=30, width=1, wl=DEFAULT_WAVELET)
    widths = init_width:largest_width
    _cost_prev = typemax(Float64)
    for i in eachindex(widths)
        _cost = calc_cost(
            spike,
            _shift_template(create_spike_template(signal_length, halfwindow(spike), amp, widths[i]), offset)
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
    y = highpass(hsigmoid.(x, amp, width))
    # y=filtfilt(digitalfilter(Highpass(f0; fs=fs), Butterworth(4)), hsigmoid.(x.-offset,amp,width))
    res = cwt(y, wl)
    res_trunc = _truncate_template(res, hw)
    return Template(res_trunc)
end

function _truncate_template(template_cwt::AbstractMatrix, hw::Integer)
    n = size(template_cwt)[1]
    c = (n + 1) ÷ 2
    return template_cwt[c-hw:c+hw, :]
end

function _shift_template(template::Template, offset)
    res_shifted = circshift(template.res, (offset, 0))
    if offset ≥ 0
        res_shifted[1:offset, :] .= 0
    else
        res_shifted[end+offset:end, :] .= 0
    end
    return Template(res_shifted)
end

function treat_spike(signal_cwt::AbstractMatrix, spike::Spike; wl=DEFAULT_WAVELET)
    hw = halfwindow(spike)
    amp = _find_amplitude(spike, size(signal_cwt, 1); wl=wl)
    template = create_spike_template(size(signal_cwt, 1), hw, amp, 1)
    offset, _ = _find_offset(spike, template, amp, center(spike))
    template_shifted = _shift_template(template, offset)
    treated_spike = signal_cwt[spike.offset-halfwindow(spike):spike.offset+halfwindow(spike), :] .- template_shifted.res
    return treated_spike
end


function treat_spike!(signal_cwt::AbstractMatrix, spike::Spike; wl=DEFAULT_WAVELET)
    hw = halfwindow(spike)
    amp = _find_amplitude(spike, size(signal_cwt, 1); wl=wl)
    template = create_spike_template(size(signal_cwt, 1), hw, amp, 1)
    offset, _ = _find_offset(spike, template, amp, center(spike))
    template_shifted = _shift_template(template, offset)
    signal_cwt[spike.offset-halfwindow(spike):spike.offset+halfwindow(spike), :] -= template_shifted.res
end

function fitted_template(spike::Spike, signal_length; wl=DEFAULT_WAVELET)
    hw = halfwindow(spike)
    amp = _find_amplitude(spike, signal_length; wl=wl)
    template = create_spike_template(signal_length, hw, amp, 1)
    offset, _ = _find_offset(spike, template, amp, center(spike))
    # offset = 0
    width = _find_width(spike, signal_length, amp, offset)
    template_fitted = _shift_template(create_spike_template(signal_length, hw, amp, width), offset)
    if calc_cost(spike, template_fitted) > 1
        return nothing
    end
    return template_fitted
end

function hoge()
    return "hoge"
end

function fitted_templates(spikes::Vector{Spike}, signal_length)
    templates = Vector{Template}(undef, length(spikes))
    Threads.@threads :static for i in eachindex(spikes)
        println(i)
        templates[i] = fitted_template(spikes[i], signal_length)
    end
    return templates
end

function calc_cost(spike::Spike, template::Template; method=:lower)
    @assert size(spike.res) == size(template.res) "$(size(spike.res)), $(size(template.res))"
    ## Naive subtraction

    if method === :naive
        diff = abs.(spike.res .- template.res)
        cost = mean(diff) / mean(abs.(spike.res))
    elseif method === :lower
        lrange = 1:size(spike.res, 2)÷2
        diff = abs.(spike.res[:, lrange] .- template.res[:, lrange])
        cost = mean(diff) / mean(abs.(spike.res[:, lrange]))
    else
        ArgumentError("No such method as $(method)")
    end
    return cost
end