using Random: seed!
using Statistics: mean, median

using EvalMetrics: binary_eval_report
using DataFrames
using Gen
using StatsBase: countmap, mode

include("dataloading.jl")
include("moral_ppl.jl")
using .MoralPPL
using .DataLoading: load_and_split, data_to_dict, DATA_PATH

function run_ppl_inference(model, train_data, test_data, num_samples)
    trace, lml_est = fit(model, train_data, num_samples)
    predictions = predict(model, trace, test_data, get_parameter_addresses(unique(train_data[:, :responseID])))
    return predictions, trace, lml_est
end

function main(model, train_data, test_data, n_runs::Int = 3, num_samples::Int = 1000)
    @assert n_runs >= 3

    predictions = []
    traces = []
    estimates = []

    Threads.@threads for run in 1:n_runs
        println("Running simulation $run of $n_runs.")
        run_predictions, trace, lml_est = run_ppl_inference(model, train_data, test_data, num_samples)
        push!(predictions, run_predictions)
        push!(traces, trace)
        push!(estimates, lml_est)
    end

    final_predictions = Dict()
    true_labels = data_to_dict(test_data, :responseID, :bargain_accepted)
    for (responseID, labels) in true_labels
        final_predictions[responseID] = []
        for (i, label) in enumerate(labels)
            push!(final_predictions[responseID], mean([predictions[x][responseID][i] for x in 1:n_runs]))
        end
    end

    final_predictions_list = reduce(vcat, last.(sort(collect(final_predictions), by=x->x[1])))
    labels_list = reduce(vcat, last.(sort(collect(true_labels), by=x->x[1])))
    report = binary_eval_report(convert(Vector{Float64}, labels_list), convert(Vector{Float64}, final_predictions_list))

    return predictions, final_predictions, true_labels, traces, estimates, report
end

seed!(42)
model = model_acceptance
train_data, test_data = load_and_split(DATA_PATH, true)
predictions, final_predictions, true_labels, traces, estimates, report = main(model, train_data, test_data, 7, 10)
println(report)

#using JLD2
#results_dir = "final_15_1000"
#traces = load_object("results/$(results_dir)/traces.jld2")
#report = load_object("results/$(results_dir)/report.jld2")
#results_df = DataFrame(CSV.File("results/$(results_dir)/results_df.csv"))  
#CSV.write("results/$(results_dir)/results_df.csv", results_df)
#save_object("results/$(results_dir)/report.jld2", report)
#save_object("results/$(results_dir)/traces.jld2", traces)
