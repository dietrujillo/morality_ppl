module MoralPPL

include("damage_value.jl")
include("monetary_value.jl")
include("decision_function.jl")
include("generative_model.jl")

export main, VALID_DAMAGE_TYPES

main(amount::Real, damage::String) = main(amount, DamageType(damage))
function main(amount::Real, damage::DamageType)
    return generative_model(amount, damage)
end

end

print(MoralPPL.main(5001, "bluehouse"))
