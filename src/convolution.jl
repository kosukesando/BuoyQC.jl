using Wavelets
using ContinuousWavelets
using Peaks
using Statistics
using Optim
using DataFrames
using DSP
using PaddedViews
using Suppressor

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
    signal = signal .- mean(signal)
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

function peaks_to_spikes(wlcoeff_pad, pks, heights; wl=wavelet(cSym4), hw=400)
    spikes = Vector{Spike}(undef, length(pks))
    for i in eachindex(pks)
        pk = pks[i]
        spike = Spike(pk + hw, wlcoeff_pad[pk:pk+2*hw, :], heights[i])
        spikes[i] = spike
    end
    return spikes
end

function detect_spikes(signal; wl=wavelet(cSym4), fs=1.28, sup=10, hw=400, strength=0.1, est_spikes=nothing)
    n = length(signal)
    pks_all, heights_all = @suppress_err detect_spikes_conv(signal; sup=sup)
    if isnothing(est_spikes)
        n_pks = length(pks_all)
        strength = max(min(1, strength), 0)
        est_spikes = round(Integer, n_pks * strength)
    end
    pks = []
    heights = []
    i = 1
    flags = fill(false, n)
    while true
        if (flags[pks_all[i]] == false)
            push!(pks, pks_all[i])
            push!(heights, heights_all[i])
            flags[max(1, pks_all[i] - hw):min(n, pks_all[i] + hw)] .= true
        end
        length(pks) >= est_spikes && break
        i >= length(pks_all) && break
        i += 1
    end
    wlcoeff = @suppress_err cwt(signal, wl)
    wlcoeff_pad = PaddedView(0, wlcoeff,
        (1:size(wlcoeff, 1)+hw*2, 1:size(wlcoeff, 2)),
        (1+hw:size(wlcoeff, 1)+hw, 1:size(wlcoeff, 2))
    )
    @assert wlcoeff_pad[1+hw:size(wlcoeff, 1)+hw, :] == wlcoeff
    spikes = @suppress_err peaks_to_spikes(wlcoeff_pad, pks, heights; wl=wl, hw=hw)
end


function despike_conv(signal; wl=wavelet(cSym4), fs=1.28, sup=10, hw=400, strength=0.1, est_spikes=nothing)
    n = length(signal)
    pks_all, heights_all = @suppress_err detect_spikes_conv(signal; sup=sup)
    if isnothing(est_spikes)
        n_pks = length(pks_all)
        strength = max(min(1, strength), 0)
        est_spikes = round(Integer, n_pks * strength)
    end
    pks = []
    heights = []
    i = 1
    flags = fill(false, n)
    while true
        if (flags[pks_all[i]] == false)
            push!(pks, pks_all[i])
            push!(heights, heights_all[i])
            flags[max(1, pks_all[i] - hw):min(n, pks_all[i] + hw)] .= true
        end
        length(pks) >= est_spikes && break
        i >= length(pks_all) && break
        i += 1
    end
    wlcoeff = @suppress_err cwt(signal, wl)
    wlcoeff_pad = PaddedView(0, wlcoeff,
        (1:size(wlcoeff, 1)+hw*2, 1:size(wlcoeff, 2)),
        (1+hw:size(wlcoeff, 1)+hw, 1:size(wlcoeff, 2))
    )
    @assert wlcoeff_pad[1+hw:size(wlcoeff, 1)+hw, :] == wlcoeff
    spikes = @suppress_err peaks_to_spikes(wlcoeff_pad, pks, heights; wl=wl, hw=hw)

    templates = Vector{Template}(undef, length(spikes))
    for i in eachindex(spikes)
        templates[i] = est_template(spikes[i], n)
    end

    wlcoeff_treated_conv = copy(wlcoeff_pad)
    for (spike, template) in zip(spikes, templates)
        if isnothing(template)
            continue
        end
        rep = wlcoeff_treated_conv[spike.offset-hw:spike.offset+hw, :] - template.wlcoeff
        spike_temp = Spike(spike.offset,
            wlcoeff_treated_conv[spike.offset-hw:spike.offset+hw, :],
            spike.amp
        )
        cost_treat = @suppress_err calc_cost(spike_temp, template)
        cost_notreat = @suppress_err calc_cost(spike_temp, Template(zeros(Float64, size(template.wlcoeff)), nothing, nothing))
        if cost_treat < cost_notreat
            wlcoeff_treated_conv[spike.offset-hw:spike.offset+hw, :] = rep
        end
    end
    wlcoeff_treated_conv = wlcoeff_treated_conv[1+hw:n+hw, :]

    signal_treated = @suppress_err icwt(wlcoeff_treated_conv, wl, DualFrames())

    return wlcoeff_treated_conv, signal_treated
end
