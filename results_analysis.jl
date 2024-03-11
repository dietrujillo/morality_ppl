using Plots
using StatsPlots, CSV, DataFrames
using StatsBase
using DataFrames
using Measures

num_simulations = 31
participants = unique(train_data[:, :responseID])
simulation_participants = [(participant, x) for participant in participants for x in 1:num_simulations]

function is_rule_based(participant::Tuple{String, Int})
    responseID, simulationID = participant
    response_index = findfirst((x -> x == responseID), participants)
    return Gen.get_choices(traces[response_index][simulationID])[:individual_type] == 1
end

function is_flexible(participant::Tuple{String, Int})
    responseID, simulationID = participant
    response_index = findfirst((x -> x == responseID), participants)
    return Gen.get_choices(traces[response_index][simulationID])[:individual_type] == 2
end

for response_id in filter((x -> is_rule_based(x) && is_flexible(x)), simulation_participants)
    println(filter(:responseID => (x -> x == response_id), results_df))
end

length(filter((x -> is_rule_based(x)), simulation_participants))
length(filter((x -> is_flexible(x)), simulation_participants))
length(filter((x -> !is_rule_based(x) && !is_flexible(x)), simulation_participants))

OFFERS = sort(unique(results_df[:, :amount_offered]), rev=false)
for offer in OFFERS
    offer_df = filter(:amount_offered => (x -> x == offer), results_df)
    println("Filtering by offer $offer - $(nrow(offer_df)) entries.")

    rule_based = length(filter((x -> is_rule_based(x)), simulation_participants))
    flexible = length(filter((x -> is_flexible(x)), simulation_participants))
    agreement = length(filter((x -> !is_rule_based(x) && !is_flexible(x)), simulation_participants))

    rule_based_frac = rule_based / length(simulation_participants)
    flexible_frac = flexible / length(simulation_participants)
    agreement_frac = agreement / length(simulation_participants)

    println("Rule-based inflexible: $rule_based - $(round(rule_based_frac * 100, digits=2))%")
    println("Rule-based flexible: $flexible - $(round(flexible_frac * 100, digits=2))%")
    println("Agreement-based: $agreement - $(round(agreement_frac * 100, digits=2))%")
    println()
end

gr()
horizontal_layout = @layout([a b c])
vertical_layout = @layout([a; b; c; d; e; f; g; h; i; j])

function build_damage_boxplots(damage_type::Symbol, results_df::DataFrame, size::Tuple = (1200, 300))
    if damage_type == :all
        damage_type_df = results_df
    else
        damage_type_df = filter(:damage_type => (x -> x == damage_type), results_df)
    end
    rule_based_df = vcat([filter(:responseID => (x -> is_rule_based((x, y))), damage_type_df) for y in num_simulations]...)
    flexible_df = vcat([filter(:responseID => (x -> is_flexible((x, y))), damage_type_df) for y in num_simulations]...)
    agreement_df = vcat([filter(:responseID => (x -> !is_rule_based((x, y)) && !is_flexible((x, y))), damage_type_df) for y in num_simulations]...)

    damage_plots = []
    for (name, df) in zip(["Rule-Based", "Flexible", "Agreement-Based"], [rule_based_df, flexible_df, agreement_df])
        plot_data = replace([filter(:amount_offered => (x -> x == offer), df)[:, :predictions] for offer in OFFERS], [] => [0.])
        plt = boxplot(plot_data, xticks=(1:5, string.(OFFERS)), legend=false, title=name, ylim=(0., 1.))
        labels_plot_data = replace([filter(:amount_offered => (x -> x == offer), df)[:, :labels] for offer in OFFERS], [] => [0.])
        labels_average_data = mean.(labels_plot_data)
        plt = plot!(1:5, labels_average_data, xticks=(1:5, string.(OFFERS)), legend=false, title=name, ylim=(0., 1.), linewidth=5, color="steelblue")
        push!(damage_plots, plt)
    end

    damage_plot = plot(damage_plots..., layout=horizontal_layout, size=size, plot_title=String(damage_type), plot_titlevspan=0.1)
    return damage_plot
end

damage_type = :razehouse
damage_boxplots = build_damage_boxplots(damage_type, results_df)
savefig(damage_boxplots, "$(String(damage_type))_plot.png")

function build_proportion_barplots(damage_type::Symbol, results_df::DataFrame, size::Tuple = (1200, 300))
    if damage_type == :all
        damage_type_df = results_df
    else
        damage_type_df = filter(:damage_type => (x -> x == damage_type), results_df)
    end

    rule_based_frac = []
    flexible_frac = []
    agreement_frac = []
    for offer in OFFERS
        offer_df = filter(:amount_offered => (x -> x == offer), damage_type_df)    

        rule_based = length(filter((x -> is_rule_based(x)), simulation_participants))
        flexible = length(filter((x -> is_flexible(x)), simulation_participants))
        agreement = length(filter((x -> !is_rule_based(x) && !is_flexible(x)), simulation_participants))

        push!(rule_based_frac, rule_based / nrow(offer_df))
        push!(flexible_frac, flexible / nrow(offer_df))
        push!(agreement_frac, agreement / nrow(offer_df))
    end

    barplot = groupedbar(
        [rule_based_frac flexible_frac agreement_frac],
        bar_position=:stack, bar_width=0.7, size=size, 
        xticks=(1:5, string.(Int.(OFFERS))),
        label=["rule-based" "flexible" "agreement"]
    )
    return barplot
end

build_proportion_barplots(:bluemailbox, results_df)

# Build megaplot
results_plots = []
for damage_type in VALID_DAMAGE_TYPES
    damage_plot = build_damage_boxplots(damage_type, results_df)
    push!(results_plots, damage_plot)
end
results_plots
final_plot = plot(results_plots..., layout=vertical_layout, size=(1600, 3600))
savefig(final_plot, "final_plot.png")

# Get individual type predictions
individual_predictions = [[Gen.get_choices(traces[x][y])[:individual_type] for y in 1:num_simulations] for x in 1:198]

function entropy(data::Vector{Int64})
    n = length(unique(data))
    p = zeros(n)
    for i in unique(data)
        c = count(x -> x == i, data)
        p[findfirst(==(i), unique(data))] = c / length(data)
    end
    -sum(p .* log2.(p))
end

countmap(entropy.(individual_predictions))
for i in countmap.(filter((x -> entropy(x) != 0.), individual_predictions))
    println(i)
end