function C2D4(ξ::Float64, η::Float64)
     values::Array{Float64, 1} = [(1 - ξ) * (1 - η),
                                  ξ * (1 - η),
                                  ξ * η,
                                  (1 - ξ) * η]
    return values
end

function C3D8(ξ::Float64, η::Float64, μ::Float64)
    values::Array{Float64, 1} = 1 / 8 * [(1 - ξ) * (1 - η) * (1 - μ),
                                         (1 + ξ) * (1 - η) * (1 - μ),
                                         (1 + ξ) * (1 + η) * (1 - μ),
                                         (1 - ξ) * (1 + η) * (1 - μ),
                                         (1 - ξ) * (1 - η) * (1 + μ),
                                         (1 + ξ) * (1 - η) * (1 + μ),
                                         (1 + ξ) * (1 + η) * (1 + μ),
                                         (1 - ξ) * (1 + η) * (1 + μ)]
    return values
end
