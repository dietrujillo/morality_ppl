module MoralPPL

include("damage_value.jl")
include("generative_model.jl")

using Gen
using .GenerativeModel: model_acceptance

export main, VALID_DAMAGE_TYPES, COMPENSATION_DEMANDED_TABLE

function main()
    choicemap = Gen.get_choices(Gen.simulate(model_acceptance, ([10000., 1001.], [100., 900.])))
    return choicemap
end

end

choicemap = MoralPPL.main()
