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
    fitted_template,
    fitted_templates,
    detect_spikes_conv

include("utils.jl")
include("wavelet_analysis.jl")
include("convolution.jl")

end
