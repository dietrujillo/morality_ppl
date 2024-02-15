using Plots
using ROC
using StatsPlots: boxplot
using Statistics: mean

# ROC curve
final_predictions = mean.(eachrow(predictions_df))
labels = test_data[:, :bargain_accepted]
plot(roc(final_predictions, labels, true), legend=false)
xlabel!("False Positive Rate")
ylabel!("True Positive Rate")
savefig("ppl_roc_curve.png")

# Predicted scores
predictions_counts = sort(collect(countmap(mean.(eachrow(predictions_df)))))
histogram(mean.(eachrow(predictions_df)), bins=10, label=false, alpha=0.6)
plot!(first.(predictions_counts), last.(predictions_counts), label=false, color="black", linewidth=2)
xlabel!("Score")
ylabel!("Count")
title!("Histogram of Model Scores")
savefig("scores_hist.png")

# Visualizations of estimated parameters
addresses = PARAMETER_ADDRESSES
estimated_params = Dict([addr => [Gen.get_choices(traces[t])[addr] for t=1:8] for addr in addresses])

# ==> Rule-Based Individual Probability
boxplot(estimated_params[:rule_based_individual_p], color="orange", xlim=(-1, 3), legend=false, size=(400, 600))
scatter!(ones(length(estimated_params[:rule_based_individual_p])), 
        estimated_params[:rule_based_individual_p], color=:black)
xticks!(1:1, ["Rule-Based Individual"])
ylabel!("Probabilities by fitted generative models")
savefig("rule_based_p_result.png")

# ==> Unreasonable Neighbor
boxplot(estimated_params[:unreasonable_p], color="purple", xlim=(-1, 3), legend=false, size=(400, 600))
scatter!(ones(length(estimated_params[:unreasonable_p])), 
        estimated_params[:unreasonable_p], color=:black)
xticks!(1:1, ["Unreasonable Neighbor Probability"])
ylabel!("Probability by fitted generative models")
savefig("unreasonable_probability.png")

boxplot(estimated_params[:unreasonable_neighbor_λ], color="lightgreen", xlim=(-1, 3), legend=false, size=(400, 600))
scatter!(ones(length(estimated_params[:unreasonable_neighbor_λ])), 
        estimated_params[:unreasonable_neighbor_λ], color=:black)
xticks!(1:1, ["Unreasonable Neighbor λ"])
ylabel!("Rate of Unreasonable Neighbor Exponential Distribution")
savefig("unreasonable_lambda.png")

# ==> Thresholds for the rule-based setting
boxplot(estimated_params[:min_utility_threshold], color="lightblue", xlim=(-1, 3), legend=false, size=(400, 600))
scatter!(ones(length(estimated_params[:min_utility_threshold])), 
        estimated_params[:min_utility_threshold], color=:black)
xticks!(1:1, ["Minimum Utility Threshold"])
ylabel!("")
savefig("min_utility_threshold_boxplot.png")

boxplot(estimated_params[:max_cost_threshold], color="lightsalmon", xlim=(-1, 3), legend=false, size=(500, 600))
scatter!(ones(length(estimated_params[:max_cost_threshold])), 
        estimated_params[:max_cost_threshold], color=:black)
xticks!(1:1, ["Maximum Cost Threshold"])
ylabel!("")
savefig("max_utility_threshold_boxplot.png")



