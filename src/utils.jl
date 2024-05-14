function hsigmoid(t, a=0.5, w=0)
    abs(t) < w ? a / w * t : a * (sign(t))
end