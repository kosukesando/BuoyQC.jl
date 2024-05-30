using Wavelets
using ContinuousWavelets
using Peaks
using Statistics
using Optim
using DataFrames
using DSP
using PaddedViews

function minheight(indices, heights, minheight=0.0)
    i_delete = []
    for i in eachindex(indices)
        if heights[i] < minheight
            push!(i_delete, i)
        end
    end
    return deleteat!(indices, i_delete), deleteat!(heights, i_delete)
end

function maxheight(indices, heights, maxheight=0.0)
    i_delete = []
    for i in eachindex(indices)
        if heights[i] > maxheight
            push!(i_delete, i)
        end
    end
    return deleteat!(indices, i_delete), deleteat!(heights, i_delete)
end

function detect_spikes_conv(signal; fs=1.28, sup=10)
    n = length(signal)
    dt = 1 / fs
    x = -sup:dt:sup
    # convolution
    signal_conv_full = conv(signal, highpass(hsigmoid.(x, 1, 1)))
    pad = abs(n - length(signal_conv_full))
    signal_conv = signal_conv_full[pad÷2:pad÷2+n-1]
    # finding peaks and sorting them
    pks = []
    heights = []
    for (find, filter) in zip([findmaxima, findminima], [minheight, maxheight])
        _pks, _heights, _ = find(signal_conv)
        _pks, _heights = filter(_pks, _heights)
        _perm = sortperm(abs.(_heights)) # highest last

        push!(pks, _pks[_perm])
        push!(heights, _heights[_perm])
    end
    total_length = length(pks[1]) + length(pks[2])
    pks_comb = Vector{Integer}(undef, total_length)
    heights_comb = Vector{Float64}(undef, total_length)
    for i in 1:total_length
        if isempty(pks[1])
            j = 2
        elseif isempty(pks[2])
            j = 1
        else
            j = argmax(abs.([heights[1][end], heights[2][end]]))
        end
        pks_comb[i] = pop!(pks[j])
        heights_comb[i] = pop!(heights[j])
    end
    return pks_comb, heights_comb
end

function peaks_to_spikes(res_pad, pks; wl=wavelet(cSym4), hw=400)
    spikes = Vector{Spike}(undef, length(pks))
    for i in eachindex(pks)
        pk = pks[i]
        spike = Spike(pk + hw, res_pad[pk:pk+2*hw, :], 0)
        spikes[i] = spike
    end
    return spikes
end


function treat_conv(signal; wl=wavelet(cSym4), fs=1.28, sup=10, hw=400)
    n = length(signal)
    pks, heights = detect_spikes_conv(signal; sup=sup)
    res = cwt(signal, wl)
    res_pad = PaddedView(0, res,
        (1:size(res, 1)+hw*2, 1:size(res, 2)),
        (1+hw:size(res, 1)+hw, 1:size(res, 2))
    )
    @assert res_pad[1+hw:size(res, 1)+hw, :] == res
    spikes = peaks_to_spikes(res_pad, pks; wl=wl, hw=hw)

    templates = Vector{Template}(undef, length(spikes))
    for i in eachindex(spikes)
        templates[i] = fitted_template(spikes[i], n)
    end

    res_treated_conv = copy(res_pad)
    for (spike, template) in zip(spikes, templates)
        if isnothing(template)
            continue
        end
        rep = res_treated_conv[spike.offset-hw:spike.offset+hw, :] - template.res
        spike_temp = Spike(spike.offset,
            res_treated_conv[spike.offset-hw:spike.offset+hw, :],
            spike.amp
        )
        cost_treat = calc_cost(spike_temp, template)
        cost_notreat = calc_cost(spike_temp, Template(zeros(Float64, size(template.res))))
        if cost_treat < cost_notreat
            res_treated_conv[spike.offset-hw:spike.offset+hw, :] = rep
        end
    end
    res_treated_conv = res_treated_conv[1+hw:n+hw, :]

    signal_treated = icwt(res_treated_conv, wl, DualFrames())

    return res_treated_conv, signal_treated
end
