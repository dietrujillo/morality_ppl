using Random: seed!

using DataFrames
using StatsBase
using Combinatorics
using GMT
using LinearAlgebra
using ProgressBars

include("dataloading.jl")
using .DataLoading: load_dataset, data_to_dict, DATA_PATH
include("probit.jl")
using .Probit: acceptance_model, acceptance_inference, acceptance_prediction
include("compensation_demanded.jl")
using .CompensationDemanded
include("data_visualization.jl")

seed!(42)

# Prepare data
data = load_dataset(DATA_PATH, true)

damage_means = Dict(VALID_DAMAGE_TYPES .=> [COMPENSATION_DEMANDED_TABLE[:, damage_type][1] for damage_type in VALID_DAMAGE_TYPES])
damage_stds = Dict(VALID_DAMAGE_TYPES .=> [COMPENSATION_DEMANDED_TABLE[:, damage_type][5] for damage_type in VALID_DAMAGE_TYPES]) 

acceptances = data_to_dict(data, :responseID, :bargain_accepted)
amounts = data_to_dict(data, :responseID, :amount_offered)
damages = data_to_dict(data, :responseID, :damage_type)

# Helper function to find simplex points for type prior grid search
function simplex_grid(num_outcomes, points_per_dim)
    num_points = multinomial(num_outcomes-1, points_per_dim)
    points = Array{Float64,2}(undef, num_outcomes, num_points)
    for (p,comb) in enumerate(with_replacement_combinations(1:num_outcomes, points_per_dim))
        distr = counts(comb, 1:num_outcomes) ./ points_per_dim
        points[:,p] = distr
    end
    return [points[:,i] for i in 1:size(points,2)]
end
type_priors = simplex_grid(3, 20)

# Run Bayesian inference and predict on the dataset
final_results = Dict()
final_predictions = Dict()
for type_prior in ProgressBar(type_priors)
    model_results = Dict()
    model_predictions = Dict()
    for individual in unique(data[:, :responseID])
        model_results[individual] = acceptance_inference(
            acceptances[individual],
            amounts[individual],
            damages[individual],
            damage_means,
            damage_stds, 
            type_prior
        )
        model_predictions[individual] = acceptance_prediction(
            model_results[individual],
            amounts[individual],
            damages[individual],
            damage_means,
            damage_stds,
            10
        )
    end
    final_results[type_prior] = model_results
    final_predictions[type_prior] = model_predictions
end

# Helper functions to analyze results

## Retrieves the likelihood of the model by taking the product of likelihoods for each individual (sum of log likelihoods)
get_model_likelihood(model_results) = sum([v.lml for v in values(model_results)])

## Removes not valid individuals (likelihood of 0) from the results
get_valid_results(model_results) = Dict(filter(p -> last(p).lml != -Inf, collect(model_results)))

## Retrieves the value (or key) of the results dict when the key is not exactly right
## (as it is a continuous vector) by taking the closest key to the vector provided.
function get_fuzzy_value(results, key, return_key=false)
    closest = [1, 0, 0]
    minnorm = Inf
    for candidate in first.(collect(results))
        if norm(key - candidate) < minnorm
            minnorm = norm(key - candidate)
            closest = candidate
        end
    end
    if return_key
        return closest
    end
    return Dict(results)[closest]
end

# Remove individuals that could not be modeled
# Also manually remove the type priors where p(rule_based) ∈ {0,1}, as it messes up the math due to the deterministic nature of rule-based people.
keys_to_delete = filter(x -> x == [1, 0, 0] || x[1] == 0, type_priors)
valid_results = [k => get_valid_results(v) for (k, v) in final_results if k ∉ keys_to_delete]

# Study how many people are valid and whether they are the same across models
validkeys = [collect(keys(x)) for x in last.(valid_results)]
all(x == validkeys[1] for x in validkeys)

# Plots

## Priors heatmap
prior_likelihoods = Dict([k => get_model_likelihood(v) for (k, v) in valid_results])
prior_likelihoods = sort(collect(prior_likelihoods), by=last, rev=true)
plot_priors_heatmap(first.(prior_likelihoods), last.(prior_likelihoods))

## Prediction boxplots
model_key = first(prior_likelihoods[1])
#model_key = get_fuzzy_value(valid_results, [1//3, 1//3, 1//3], true)
plot_prediction_boxplots(
    :bluehouse,
    final_predictions[model_key],
    amounts,
    damages,
    acceptances,
    Dict(valid_results)[model_key]
)
