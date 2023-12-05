module MoralPPL

include("damage_value.jl")
include("generative_model.jl")

using Gen
using .GenerativeModel: model_acceptance

export main, VALID_DAMAGE_TYPES, COMPENSATION_DEMANDED_TABLE

function model_inference(model, data)
    
    observations = Gen.choicemap()
    for (i, y) in enumerate(data[:bargain_accepted])
        observations[(i, :acceptance)] = y
    end
    
    num_samples = 10000
    (trace, _) = Gen.importance_resampling(model, (data[:amount_offered], data[:damage_type]), observations, num_samples);
    return trace
end

function main()
    choicemap = Gen.get_choices(Gen.simulate(model_acceptance, ([10000., 1001.], [:razehouse, :bluemailbox])))
    return choicemap
end

end

choicemap = MoralPPL.main()
