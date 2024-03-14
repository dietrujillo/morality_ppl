module MoralPPL

include("damage_type.jl")
include("generative_model.jl")
include("dataloading.jl")

using DataFrames
using Gen
using Statistics: mean
using StatsBase

using .GenerativeModel
using .DataLoading: data_to_dict

export model_acceptance, load_dataset, fit, predict, COMPENSATION_DEMANDED_TABLE, get_parameter_addresses

function fit(model, data, num_samples::Int = 1000)
    observations = Gen.choicemap()

    amounts_offered_dict = data_to_dict(data, :responseID, :amount_offered)
    damage_types_dict = data_to_dict(data, :responseID, :damage_type)
    bargain_accepted_dict = data_to_dict(data, :responseID, :bargain_accepted)

    for (responseID, acceptances) in bargain_accepted_dict
        for (i, accept) in enumerate(acceptances)
            observations[(:individual, responseID) => (:acceptance, i) => :accept] = accept
        end
    end
    
    (trace, lml_est) = Gen.importance_resampling(model, (amounts_offered_dict, damage_types_dict, collect(keys(bargain_accepted_dict))),
                                           observations, num_samples, verbose=true)
    return trace, lml_est
end

function predict(model, trace, data, parameter_addresses::Vector, num_predict_rounds::Int = 10)
    
    constraints = Gen.choicemap()
    for addr in parameter_addresses
        try
            constraints[addr] = trace[addr]
        catch e
            println(addr)
            throw(e)
        end
    end

    predictions = Dict()
    amounts_offered_dict = data_to_dict(data, :responseID, :amount_offered)
    damage_types_dict = data_to_dict(data, :responseID, :damage_type)
    response_ids = collect(keys(amounts_offered_dict))

    for round in 1:num_predict_rounds
        (new_trace, _) = Gen.generate(model, (amounts_offered_dict, damage_types_dict, response_ids), constraints)
        for responseID in response_ids
            predictions[responseID] = [new_trace[(:individual, responseID) => (:acceptance, i) => :accept] for i=1:length(amounts_offered_dict[responseID])]
        end
    end
    
    return predictions
end

end  # Module MoralPPL
