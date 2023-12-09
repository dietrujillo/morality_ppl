using Random: shuffle
using DataFrames

using Statistics: mean
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

function main(n_runs::Int = 3, num_samples::Int = 1000)
    table = load_dataset(DATA_PATH)
    train_data, test_data = splitdf(table, 0.7)
    accuracies = zeros(n_runs)

    println("Running $n_runs simulations in parallel...")

    Threads.@threads for run in 1:n_runs
        predictions, trace = run_model(model_acceptance, train_data, test_data, num_samples)
        accuracy = (1 - sum(broadcast(abs, test_data[:, :bargain_accepted] - predictions)) / length(predictions))
        accuracies[run] = accuracy
    end

    if length(accuracies) >= 3
        x̄ = mean(accuracies)
        ci = confint(OneSampleTTest(accuracies))
        println("μ = $x̄ ± $(ci[2] - x̄)")
    else
        println(accuracies)
    end

end

main(16, 5000)