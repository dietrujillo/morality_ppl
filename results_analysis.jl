participants = unique(train_data[:, :responseID])

function is_rule_based(responseID)
    response_index = findfirst((x -> x == responseID), participants)
    return mean([Gen.get_choices(traces[response_index][x])[:rule_based_individual_p] for x in 1:8]) > 0.5
end

function is_flexible(responseID)
    response_index = findfirst((x -> x == responseID), participants)
    return mean([Gen.get_choices(traces[response_index][x])[:flexible_p] for x in 1:8]) > 0.5
end

for response_id in filter((x -> is_rule_based(x) && is_flexible(x)), unique(results_df[:, :responseID]))
    println(filter(:responseID => (x -> x == response_id), results_df))
end

length(filter((x -> is_rule_based(x) && !is_flexible(x)), unique(results_df[:, :responseID])))
length(filter((x -> is_rule_based(x) && is_flexible(x)), unique(results_df[:, :responseID])))
length(filter((x -> !is_rule_based(x)), unique(results_df[:, :responseID])))

OFFERS = unique(results_df[:, :amount_offered])
for offer in sort(OFFERS, rev=false)
    offer_df = filter(:amount_offered => (x -> x == offer), results_df)
    println("Filtering by offer $offer - $(nrow(offer_df)) entries.")

    rule_based = length(filter((x -> is_rule_based(x) && !is_flexible(x)), unique(offer_df[:, :responseID])))
    flexible = length(filter((x -> is_rule_based(x) && is_flexible(x)), unique(offer_df[:, :responseID])))
    agreement = length(filter((x -> !is_rule_based(x)), unique(offer_df[:, :responseID])))

    println("Rule-based inflexible: $rule_based - $(round(rule_based / nrow(offer_df) * 3 * 100, digits=2))%")
    println("Rule-based flexible: $flexible - $(round(flexible / nrow(offer_df) * 3* 100, digits=2))%")
    println("Agreement-based: $agreement - $(round(agreement / nrow(offer_df) * 3* 100, digits=2))%")
    println()
end