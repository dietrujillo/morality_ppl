using Random: seed!
using Statistics: mean

using EvalMetrics: binary_eval_report
using DataFrames
using Gen
using StatsBase: countmap

include("dataloading.jl")
include("moral_ppl.jl")
using .MoralPPL

function run_ppl_inference(model, train_data, test_data, num_samples)
    trace = fit(model, train_data, num_samples)
    predictions = predict(model, trace, test_data, PARAMETER_ADDRESSES)
    return predictions, trace
end

function main(train_data, test_data, n_runs::Int = 3, num_samples::Int = 1000)
    @assert n_runs >= 3

    participants = unique(train_data[:, :responseID])

    traces = Vector{Vector{Gen.DynamicDSLTrace{DynamicDSLFunction{Any}}}}(undef, length(participants))

    predictions = []
    labels = []
    ids = []
    damage_types = []
    amounts_offered = []

    for (index, participant) in enumerate(participants)

        println("Running simulations for participant $index.")

        participant_predictions = Vector{Vector{Float64}}(undef, n_runs)
        participant_traces = Vector{Gen.DynamicDSLTrace{DynamicDSLFunction{Any}}}(undef, n_runs)

        participant_train_data = filter(:responseID => (x -> x == participant), train_data)
        participant_test_data = filter(:responseID => (x -> x == participant), test_data)
        Threads.@threads for run in 1:n_runs
            run_predictions, trace = run_ppl_inference(model_acceptance, participant_train_data, participant_test_data, num_samples)
            participant_predictions[run] = run_predictions
            participant_traces[run] = trace
        end
        traces[index] = participant_traces

        participant_predictions_df = DataFrame(convert.(Vector{Float64}, participant_predictions), :auto)
        participant_ensemble_predictions = mean.(eachrow(participant_predictions_df))
        
        predictions = vcat(predictions, participant_ensemble_predictions)
        labels = vcat(labels, participant_test_data[:, :bargain_accepted])
        ids = vcat(ids, repeat([participant], outer=length(participant_ensemble_predictions)))
        damage_types = vcat(damage_types, participant_test_data[:, :damage_type])
        amounts_offered = vcat(amounts_offered, participant_test_data[:, :amount_offered])
    end

    results_df = DataFrame(
        :predictions => convert(Vector{Float64}, predictions),
        :labels => convert(Vector{Float64}, labels),
        :responseID => convert(Vector{String}, ids),
        :damage_type => convert(Vector{Symbol}, damage_types),
        :amount_offered => convert(Vector{Float64}, amounts_offered)
    )
    results_df[:, :final_pred] = map((x) -> round(x), results_df[:, :predictions])

    report = binary_eval_report(results_df[:, :labels], results_df[:, :predictions])

    return report, traces, results_df

end

seed!(42)
train_data, test_data = load_and_split(DATA_PATH, true)
report, traces, results_df = main(train_data, test_data, 8, 1000)
println(report)
