module MoralPPL

include("damage_type.jl")
include("generative_model.jl")

using DataFrames
using Gen
using Statistics: mean

using .GenerativeModel

export model_acceptance, load_dataset, fit, predict, COMPENSATION_DEMANDED_TABLE, PARAMETER_ADDRESSES

function fit(model, data, num_samples::Int = 1000)
    
    observations = Gen.choicemap()
    for (i, y) in enumerate(data[:, :bargain_accepted])
        observations[(:acceptance, i) => :accept] = y
    end
    
    (trace, _) = Gen.importance_resampling(model, (data[:, :amount_offered], data[:, :damage_type]),
                                           observations, num_samples, verbose=true)
    return trace
end

function predict(model, trace, test_data::Union{DataFrame, SubDataFrame}, parameter_addresses::Vector, num_predict_rounds::Int = 10)
    
    constraints = Gen.choicemap()
    for addr in parameter_addresses
        try
            constraints[addr] = trace[addr]
        catch e
            println(addr)
            throw(e)
        end
    end

    predictions = []
    for round in 1:num_predict_rounds
        (new_trace, _) = Gen.generate(model, (test_data[:, :amount_offered], test_data[:, :damage_type]), constraints)
        push!(predictions, [new_trace[(:acceptance, i) => :accept] for i=1:nrow(test_data)])
    end
    
    predictions_df = DataFrame(convert.(Vector{Float64}, predictions), :auto)
    model_predictions = mean.(eachrow(predictions_df))

    return model_predictions
end

end  # Module MoralPPL