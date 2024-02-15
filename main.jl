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

    println("Running $n_runs simulations in parallel...")

    participants = unique(train_data[:, :responseID])

    predictions = Vector{Vector{Vector{Float64}}}(undef, length(participants))
    traces = Vector{Vector{Gen.DynamicDSLTrace{DynamicDSLFunction{Any}}}}(undef, length(participants))

    individual_predictions = []
    individual_labels = []
    for (index, participant) in enumerate(participants)

        println("Running simulations for participant $index.")

        participant_predictions = Vector{Vector{Float64}}(undef, n_runs)
        participant_traces = Vector{Gen.DynamicDSLTrace{DynamicDSLFunction{Any}}}(undef, n_runs)

        participant_train_data = filter(:responseID => (x -> x  == participant), train_data)
        participant_test_data = filter(:responseID => (x -> x  == participant), test_data)
        Threads.@threads for run in 1:n_runs
            run_predictions, trace = run_ppl_inference(model_acceptance, participant_train_data, participant_test_data, num_samples)
            participant_predictions[run] = run_predictions
            participant_traces[run] = trace
        end
        predictions[index] = participant_predictions
        traces[index] = participant_traces

        participant_predictions_df = DataFrame(convert.(Vector{Float64}, participant_predictions), :auto)
        participant_ensemble_predictions = mean.(eachrow(participant_predictions_df))

        individual_predictions = vcat(individual_predictions, participant_ensemble_predictions)
        individual_labels = vcat(individual_labels, participant_test_data[:, :bargain_accepted])
    end

    report = binary_eval_report(convert(Vector{Float64}, individual_labels), convert(Vector{Float64}, individual_predictions))

    return report, traces

end

seed!(42)
train_data, test_data = load_and_split(DATA_PATH, true)
report, traces = main(train_data, test_data, 8, 1000)
println(report)
