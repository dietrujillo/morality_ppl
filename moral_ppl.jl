module MoralPPL

include("generative_model.jl")

using CSV
using DataFrames
using Gen

using .GenerativeModel

export model_acceptance, load_dataset, fit, predict, VALID_DAMAGE_TYPES, COMPENSATION_DEMANDED_TABLE, PARAMETER_ADDRESSES

const OFFER_AS_INT_DICT = Dict(
    "hundred" => 100,
    "thousand" => 1000,
    "tenthousand" => 10000,
    "hunthousand" => 100000,
    "million" => 1000000,
)

function load_dataset(data_path::String)
    table = CSV.read(data_path, DataFrame)
    table = stack(table, GenerativeModel.VALID_DAMAGE_TYPES)
    rename!(table, [:variable, :value] .=> [:damage_type, :bargain_accepted])
    table = filter(row -> row[:condition] in keys(OFFER_AS_INT_DICT), table)

    table[:, :amount_offered] = map(x -> OFFER_AS_INT_DICT[x], table[:, :condition])


    table[!,:damage_type] = Symbol.(table[:,:damage_type])
    table[!,:amount_offered] = convert.(Float64, table[:,:amount_offered])
    table[!,:bargain_accepted] = convert.(Bool, table[:,:bargain_accepted])

    return table[:, [:damage_type, :amount_offered, :bargain_accepted]]
end

function fit(model, data, num_samples::Int = 1000)
    
    observations = Gen.choicemap()
    for (i, y) in enumerate(data[:, :bargain_accepted])
        observations[(:acceptance, i) => :accept] = y
    end
    
    (trace, _) = Gen.importance_resampling(model, (data[:, :amount_offered], data[:, :damage_type]),
                                           observations, num_samples, verbose=true)
    return trace
end

function predict(model, trace, test_data::Union{DataFrame, SubDataFrame}, parameter_addresses::Vector)
    
    constraints = Gen.choicemap()
    for addr in parameter_addresses
        try
            constraints[addr] = trace[addr]
        catch e
            println(addr)
            throw(e)
        end
    end
    
    (new_trace, _) = Gen.generate(model, (test_data[:, :amount_offered], test_data[:, :damage_type]), constraints)
    
    predictions = [new_trace[(:acceptance, i) => :accept] for i=1:nrow(test_data)]
    return predictions
end

end  # Module MoralPPL