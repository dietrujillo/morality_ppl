using Random: shuffle, seed!
using Statistics: mean

using EvalMetrics: binary_eval_report
using DataFrames
using StatsBase: countmap

include("dataloading.jl")
include("moral_ppl.jl")
using .MoralPPL

DATA_PATH = "data/data_wide_bargain.csv"

function run_model(model, train_data, test_data, num_samples)
    trace = fit(model, train_data, num_samples)
    predictions = predict(model, trace, test_data, PARAMETER_ADDRESSES)
    return predictions, trace
end

function main(n_runs::Int = 3, num_samples::Int = 1000)
    @assert n_runs >= 3

    train_data, test_data = load_and_split(DATA_PATH)

    println("Running $n_runs simulations in parallel...")

    predictions = Vector{Vector{Float64}}(undef, n_runs)
    traces = Vector{Gen.DynamicDSLTrace{DynamicDSLFunction{Any}}}(undef, n_runs)
    Threads.@threads for run in 1:n_runs
        run_predictions, trace = run_model(model_acceptance, train_data, test_data, num_samples)
        predictions[run] = run_predictions
        traces[run] = trace
    end

    predictions_df = DataFrame(convert.(Vector{Float64}, predictions), :auto)
    ensemble_predictions = mean.(eachrow(predictions_df))

    report = binary_eval_report(convert(Vector{Float64}, test_data[:, :bargain_accepted]), ensemble_predictions)

    return report, traces, predictions_df, train_data, test_data

end

using Gen
Gen.get_choices(Gen.simulate(model_acceptance, ([1000., 10.], [:bluemailbox, :razehouse])))

report, traces, predictions_df, train_data, test_data = main(3, 100)
println(report)
predictions_dict = countmap(mean.(eachrow(predictions_df)))
println(predictions_dict)

# What percentage of true labels in the data / subsets of the data?
test_labels = convert(Vector{Float64}, test_data[:, :bargain_accepted])
full_data = test_data[:, :bargain_accepted]
wrong_data = test_data[Bool.(abs.(round.(mean.(eachrow(predictions_df))) - test_labels)), :bargain_accepted]
correct_data = test_data[.!Bool.(abs.(round.(mean.(eachrow(predictions_df))) - test_labels)), :bargain_accepted]
println(sum(full_data) / length(full_data))
println(sum(correct_data) / length(correct_data))
println(sum(wrong_data) / length(wrong_data))
