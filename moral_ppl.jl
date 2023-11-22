module MoralPPL

include("damage_value.jl")
include("monetary_value.jl")
include("decision_function.jl")
include("generative_model.jl")

export main

function main(amount::Real, damage::Real)
    return generative_model(amount, Damage(damage))
end

end

print(MoralPPL.main(5, 4))
