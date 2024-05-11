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
offers = data_to_dict(data, :responseID, :amount_offered)

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
            type_prior
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
keys_to_delete = filter(x -> x[1] ∈ [0] || x[2] ∈ [0] || x[3] ∈ [0], type_priors)
valid_results = [k => get_valid_results(v) for (k, v) in final_results if k ∉ keys_to_delete];

# Study how many people are valid and whether they are the same across models
validkeys = [collect(keys(x)) for x in last.(valid_results)]
all(x == validkeys[1] for x in validkeys)
length.(validkeys)

# Get model likelihoods in a sorted list
prior_likelihoods = Dict([k => get_model_likelihood(v) for (k, v) in valid_results])
prior_likelihoods = sort(collect(prior_likelihoods), by=last, rev=true)

#best_likelihood_without_flexible = last(first(filter(x -> first(x)[2] == 0, prior_likelihoods)))
worst_likelihood_with_flexible = last(last(filter(x -> first(x)[2] != 0, prior_likelihoods)))

# Analyze results after removing rule-based invididuals
without_rulebased_results = Dict()
without_rulebased_type_priors = [x for x in simplex_grid(2, 20) if x[1] != 1]
individual_types = [x => argmax(y.type_probs) for (x,y) in collect(Dict(valid_results)[first(prior_likelihoods[1])])]
non_rulebased_individuals = first.(filter(x -> last(x) != 1, individual_types))
for type_prior in ProgressBar(without_rulebased_type_priors)
    model_results = Dict()
    model_predictions = Dict()
    for individual in non_rulebased_individuals
        model_results[individual] = acceptance_inference(
            acceptances[individual],
            amounts[individual],
            damages[individual],
            damage_means,
            damage_stds,
            [0, type_prior...]
        )
    end
    without_rulebased_results[type_prior] = model_results
end
without_rulebased_likelihoods = sort([x => get_model_likelihood(y) for (x,y) in without_rulebased_results], by=last, rev=true)

# Plots

## Priors heatmap
plot_priors_heatmap(first.(prior_likelihoods), last.(prior_likelihoods))

pretty_prior_likelihoods = [k => exp(v - (max(last.(prior_likelihoods)...))) for (k, v) in prior_likelihoods]
plot_priors_heatmap(first.(pretty_prior_likelihoods), last.(pretty_prior_likelihoods), 0, "Probability")

## Prediction boxplots
model_key = first(prior_likelihoods[1])
#model_key = get_fuzzy_value(valid_results, [1//3, 1//3, 1//3], true)
plot_types(
    :bluehouse,
    final_predictions[model_key],
    amounts,
    damages,
    acceptances,
    Dict(valid_results)[model_key]
)

plot_thresholds(
    :bluehouse,
    final_predictions[model_key],
    amounts,
    damages,
    acceptances,
    Dict(valid_results)[model_key]
)

flexible_lineplot_basepriors = [x for x in simplex_grid(2, 20) if x[1] != 1 && x[2] != 1]
flexible_lineplot_priors = [[round((1 - y) * x[1], digits=4), y, round((1 - y) * x[2], digits=4)] for x in flexible_lineplot_basepriors for y in 0:0.1:0.9]
flexible_lineplot_likelihoods = Dict()
for type_prior in ProgressBar(flexible_lineplot_priors)
    model_results = Dict()
    for individual in unique(data[:, :responseID])
        model_results[individual] = acceptance_inference(
            acceptances[individual],
            amounts[individual],
            damages[individual],
            damage_means,
            damage_stds,
            type_prior
        )
    end
    flexible_lineplot_likelihoods[type_prior] = get_model_likelihood(model_results)
end
flexible_lineplot_likelihoods = sort(collect(flexible_lineplot_likelihoods), by=last, rev=true)
plot1 = plot_flexible_lineplot(filter(x -> first(x)[2] != 0, collect(flexible_lineplot_likelihoods)))
title!(plot1, "Model likelihood over ratio of rule-based to agreement-based for a fixed resource-rational prior")
plot2 = plot_flexible_lineplot(filter(x -> first(x)[2] == 0., collect(flexible_lineplot_likelihoods)), ["gray"])
xlabel!(plot2, "Ratio of rule-based to agreement-based")
ylabel!(plot2, "Log Probability of the model with the chosen priors")
xticks!(plot1, 0:0.1:1)
xticks!(plot2, -0.1:0.1:1)
out = Plots.plot(plot1, plot2, layout=@layout([a; b]))
#savefig(out, "flexible_lineplot.svg")

output_df = DataFrame()
for participant in keys(final_predictions[model_key])
    participant_predictions = final_predictions[model_key][participant]
    participant_labels = acceptances[participant]
    participant_damages = damages[participant]
    participant_offers = offers[participant]
    for (damage, offer, label, pred) in zip(participant_damages, participant_offers, participant_labels, participant_predictions)
        append!(output_df, [
            :responseID => participant,
            :damage_type => damage, 
            :amount_offered => offer,
            :label => label,
            :pred => pred
        ])
    end
end

individual_type_predictions = Dict([(x => argmax(y.type_probs)) for (x, y) in collect(Dict(valid_results)[model_key])])
plot_prediction_scatterplot(output_df, individual_type_predictions)
#savefig("results_scatterplot.png")
