using Random: seed!

using DataFrames

include("dataloading.jl")
using .DataLoading: load_dataset, data_to_dict, DATA_PATH
include("probit.jl")
using .Probit: acceptance_model, acceptance_inference
include("compensation_demanded.jl")
using .CompensationDemanded

seed!(42)

data = load_dataset(DATA_PATH, true)

damage_means = Dict(VALID_DAMAGE_TYPES .=> [COMPENSATION_DEMANDED_TABLE[:, damage_type][1] for damage_type in VALID_DAMAGE_TYPES])
damage_stds = Dict(VALID_DAMAGE_TYPES .=> [COMPENSATION_DEMANDED_TABLE[:, damage_type][5] for damage_type in VALID_DAMAGE_TYPES]) 

acceptances = data_to_dict(data, :responseID, :bargain_accepted)
amounts = data_to_dict(data, :responseID, :amount_offered)
damages = data_to_dict(data, :responseID, :damage_type)

results = Dict()
for individual in unique(data[:, :responseID])
    results[individual] = acceptance_inference(
        acceptances[individual],
        amounts[individual],
        damages[individual],
        damage_means,
        damage_stds, 
        [0.5, 0.0, 0.5]
    )
end

model_likelihood = sum([v.lml for v in values(results)])

results