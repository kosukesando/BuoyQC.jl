module BuoyQC

export
    hsigmoid,
    highpass,
    detect_spikes,
    create_spike_template,
    shift_template,
    calc_cost,
    Spike,
    Template,
    treat_spike,
    treat_spike!,
    est_template,
    est_templates,
    detect_spikes_conv,
    despike_conv

include("utils.jl")
include("wavelet_analysis.jl")
include("convolution.jl")

end
