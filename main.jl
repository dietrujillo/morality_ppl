using Random: shuffle, seed!
using Statistics: mean

using EvalMetrics: binary_eval_report
using DataFrames
using HypothesisTests: confint, OneSampleTTest

include("moral_ppl.jl")
using .MoralPPL

DATA_PATH = "data/data_wide_bargain.csv"

function splitdf(df, pct)
    @assert 0 <= pct <= 1
    ids = shuffle(collect(axes(df, 1)))
    sel = ids .<= nrow(df) .* pct
    return view(df, sel, :), view(df, .!sel, :)
end

function run_model(model, train_data, test_data, num_samples)
    trace = fit(model, train_data, num_samples)
    predictions = predict(model, trace, test_data, PARAMETER_ADDRESSES)
    return predictions, trace
end

function main(n_runs::Int = 3, num_samples::Int = 1000, seed::Int = 42)
    @assert n_runs >= 3
    seed!(seed)
    table = load_dataset(DATA_PATH)
    train_data, test_data = splitdf(table, 0.7)

    println("Running $n_runs simulations in parallel...")

    predictions = []
    traces = []
    Threads.@threads for run in 1:n_runs
        run_predictions, trace = run_model(model_acceptance, train_data, test_data, num_samples)
        push!(predictions, run_predictions)
        push!(traces, trace)
    end

    predictions_df = DataFrame(convert.(Vector{Float64}, predictions), :auto)
    ensemble_predictions = round.(mean.(eachrow(predictions_df)))

    report = binary_eval_report(ensemble_predictions, convert(Vector{Float64}, test_data[:, :bargain_accepted]))

    return report, traces

end

report, traces = main(16, 1000)
println(report)