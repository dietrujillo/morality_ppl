using Random: seed!
using Statistics: mean, median

using EvalMetrics: binary_eval_report
using DataFrames
using Gen
using StatsBase: countmap, mode

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

        individual_types = [Gen.get_choices(participant_traces[x])[:individual_type] for x in 1:n_runs]
        individual_type_mode = mode(individual_types)

        valid_participant_predictions = []
        for col_index in 1:length(participant_predictions[1])
            predictions_row = []
            for row_index in 1:length(participant_predictions)
                if individual_types[row_index] == individual_type_mode  # Only use predictions from the chosen individual type, ignoring the rest
                    push!(predictions_row, participant_predictions[row_index][col_index])
                end
            end
            push!(valid_participant_predictions, predictions_row)
        end

        participant_ensemble_predictions = mean.(valid_participant_predictions)  # Average simulation predictions for every test case
        
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
report, traces, results_df = main(train_data, test_data, 31, 1000)
println(report)

#using JLD2
#results_dir = "final_15_1000"
#traces = load_object("results/$(results_dir)/traces.jld2")
#report = load_object("results/$(results_dir)/report.jld2")
#results_df = DataFrame(CSV.File("results/$(results_dir)/results_df.csv"))  
#CSV.write("results/$(results_dir)/results_df.csv", results_df)
#save_object("results/$(results_dir)/report.jld2", report)
#save_object("results/$(results_dir)/traces.jld2", traces)
