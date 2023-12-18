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

    predictions = Vector{Vector{Float64}}(undef, n_runs)
    traces = Vector{Gen.DynamicDSLTrace{DynamicDSLFunction{Any}}}(undef, n_runs)
    Threads.@threads for run in 1:n_runs
        run_predictions, trace = run_ppl_inference(model_acceptance, train_data, test_data, num_samples)
        predictions[run] = run_predictions
        traces[run] = trace
    end

    predictions_df = DataFrame(convert.(Vector{Float64}, predictions), :auto)
    ensemble_predictions = mean.(eachrow(predictions_df))

    report = binary_eval_report(convert(Vector{Float64}, test_data[:, :bargain_accepted]), ensemble_predictions)

    return report, traces, predictions_df

end

seed!(42)
train_data, test_data = load_and_split(DATA_PATH)
report, traces, predictions_df = main(train_data, test_data, 8, 1000)
println(report)
