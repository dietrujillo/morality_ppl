using Random: shuffle, seed!
using Statistics: mean

using EvalMetrics: binary_eval_report
using DataFrames

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

function load_and_split(data_path, seed::Int = 42)
    seed!(seed)
    table = load_dataset(data_path)
    train_data, test_data = splitdf(table, 0.7)
    return train_data, test_data
end

function main(n_runs::Int = 3, num_samples::Int = 1000)
    @assert n_runs >= 3

    train_data, test_data = load_and_split(DATA_PATH)

    println("Running $n_runs simulations in parallel...")

    predictions = []
    traces = []
    Threads.@threads for run in 1:n_runs
        run_predictions, trace = run_model(model_acceptance, train_data, test_data, num_samples)
        push!(predictions, run_predictions)
        push!(traces, trace)
    end

    predictions_df = DataFrame(convert.(Vector{Float64}, predictions), :auto)
    ensemble_predictions = mean.(eachrow(predictions_df))

    report = binary_eval_report(convert(Vector{Float64}, test_data[:, :bargain_accepted]), ensemble_predictions)

    return report, traces

end

using Gen
Gen.get_choices(Gen.simulate(model_acceptance, ([1000., 10.], [:bluemailbox, :razehouse])))

report, traces = main(64, 1000)
println(report)