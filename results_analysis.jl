using Plots
using StatsPlots, CSV, DataFrames

participants = unique(train_data[:, :responseID])

function is_rule_based(responseID)
    response_index = findfirst((x -> x == responseID), participants)
    return mean([Gen.get_choices(traces[response_index][x])[:is_rule_based] for x in 1:8]) > 0.5
end

function is_flexible(responseID)
    response_index = findfirst((x -> x == responseID), participants)
    return mean([Gen.get_choices(traces[response_index][x])[:is_flexible] for x in 1:8]) > 0.5
end

for response_id in filter((x -> is_rule_based(x) && is_flexible(x)), unique(results_df[:, :responseID]))
    println(filter(:responseID => (x -> x == response_id), results_df))
end

length(filter((x -> is_rule_based(x) && !is_flexible(x)), unique(results_df[:, :responseID])))
length(filter((x -> is_rule_based(x) && is_flexible(x)), unique(results_df[:, :responseID])))
length(filter((x -> !is_rule_based(x)), unique(results_df[:, :responseID])))

OFFERS = sort(unique(results_df[:, :amount_offered]), rev=false)
for offer in OFFERS
    offer_df = filter(:amount_offered => (x -> x == offer), results_df)
    println("Filtering by offer $offer - $(nrow(offer_df)) entries.")

    rule_based = length(filter((x -> is_rule_based(x) && !is_flexible(x)), unique(offer_df[:, :responseID])))
    flexible = length(filter((x -> is_rule_based(x) && is_flexible(x)), unique(offer_df[:, :responseID])))
    agreement = length(filter((x -> !is_rule_based(x)), unique(offer_df[:, :responseID])))

    rule_based_frac = rule_based / nrow(offer_df) * 3
    flexible_frac = flexible / nrow(offer_df) * 3
    agreement_frac = agreement / nrow(offer_df) * 3

    println("Rule-based inflexible: $rule_based - $(round(rule_based_frac * 100, digits=2))%")
    println("Rule-based flexible: $flexible - $(round(flexible_frac * 100, digits=2))%")
    println("Agreement-based: $agreement - $(round(agreement_frac * 100, digits=2))%")
    println()
end

function build_damage_boxplots(damage_type::Symbol, results_df::DataFrame, size::Tuple = (1200, 300))
    if damage_type == :all
        damage_type_df = results_df
    else
        damage_type_df = filter(:damage_type => (x -> x == damage_type), results_df)
    end
    rule_based_df = filter(:responseID => (x -> is_rule_based(x) && !is_flexible(x)), damage_type_df)
    flexible_df = filter(:responseID => (x -> is_rule_based(x) && is_flexible(x)), damage_type_df)
    agreement_df = filter(:responseID => (x -> !is_rule_based(x)), damage_type_df)

    damage_plots = []
    for (name, df) in zip(["Rule-Based", "Flexible", "Agreement-Based"], [rule_based_df, flexible_df, agreement_df])
        plot_data = replace([filter(:amount_offered => (x -> x == offer), df)[:, :predictions] for offer in OFFERS], [] => [0.])
        plt = boxplot(plot_data, xticks=(1:5, string.(OFFERS)), legend=false, title=name)
        push!(damage_plots, plt)
    end

    damage_plot = plot(damage_plots..., layout=horizontal_layout, size=size)
    return damage_plot
end

build_damage_boxplots(:bluemailbox, results_df)

function build_true_label_plot(damage_type::Symbol, results_df::DataFrame, size::Tuple = (1200, 300))
    if damage_type == :all
        damage_type_df = results_df
    else
        damage_type_df = filter(:damage_type => (x -> x == damage_type), results_df)
    end

    rule_based_df = filter(:responseID => (x -> is_rule_based(x) && !is_flexible(x)), damage_type_df)
    flexible_df = filter(:responseID => (x -> is_rule_based(x) && is_flexible(x)), damage_type_df)
    agreement_df = filter(:responseID => (x -> !is_rule_based(x)), damage_type_df)

    damage_plots = []
    for (name, df) in zip(["Rule-Based", "Flexible", "Agreement-Based"], [rule_based_df, flexible_df, agreement_df])
        plot_data = replace([filter(:amount_offered => (x -> x == offer), df)[:, :labels] for offer in OFFERS], [] => [0.])
        average_data = mean.(plot_data)
        plt = plot(1:5, average_data, xticks=(1:5, string.(OFFERS)), legend=false, title=name)
        push!(damage_plots, plt)
    end

    damage_plot = plot(damage_plots..., layout=horizontal_layout, size=size)
    return damage_plot
end

build_true_label_plot(:bluemailbox, results_df)

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

        rule_based = length(filter((x -> is_rule_based(x) && !is_flexible(x)), unique(offer_df[:, :responseID])))
        flexible = length(filter((x -> is_rule_based(x) && is_flexible(x)), unique(offer_df[:, :responseID])))
        agreement = length(filter((x -> !is_rule_based(x)), unique(offer_df[:, :responseID])))

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
gr()
horizontal_layout = @layout([a b c])
vertical_layout = @layout([a; b; c; d; e; f; g; h; i; j])
for damage_type in VALID_DAMAGE_TYPES
    damage_plot = build_damage_plot(damage_type, results_df)
    push!(results_plots, damage_plot)
end
final_plot = plot(results_plots..., layout=vertical_layout, size=(1600, 2400))
savefig(final_plot, "final_plot.png")
