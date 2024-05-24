using DSP
function hsigmoid(t, a=0.5, w=0)
    abs(t) < w ? a / w * t : a * (sign(t))
end

function highpass(x; f0=0.01, fs=1.28)
    filtfilt(digitalfilter(Highpass(f0; fs=fs), Butterworth(4)), x)
end

function _find_next_upcross(signal::AbstractVector, idx::Integer)
    j = max(idx, firstindex(signal))
    while j < lastindex(signal) - 1
        if signal[j-1] < 0 && signal[j+1] > 0
            return j
        end
        j += 1
    end
    return nothing
end

function _find_prev_upcross(signal::AbstractVector, idx::Integer)
    j = min(idx, lastindex(signal) - 1)
    while j > firstindex(signal) + 1
        if signal[j-1] < 0 && signal[j+1] > 0
            return j
        end
        j -= 1
    end
    return nothing
end

function _find_next_dncross(signal::AbstractVector, idx::Integer)
    j = max(idx, firstindex(signal))
    while j < lastindex(signal) - 1
        if signal[j-1] > 0 && signal[j+1] < 0
            return j
        end
        j += 1
    end
    return nothing
end

function _find_prev_dncross(signal::AbstractVector, idx::Integer)
    j = min(idx, lastindex(signal) - 1)
    while j > firstindex(signal) + 1
        if signal[j-1] > 0 && signal[j+1] < 0
            return j
        end
        j += 1
    end
    return nothing
end