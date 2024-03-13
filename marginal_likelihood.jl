

using Gen
using StatsBase

function is_flexible(responseID)
    response_index = findfirst((x -> x == responseID), participants)
    return mode([Gen.get_choices(traces[response_index][x])[:individual_type] for x in 1:5]) == 2
end

function is_rule_based(responseID)
    response_index = findfirst((x -> x == responseID), participants)
    return mode([Gen.get_choices(traces[response_index][x])[:individual_type] for x in 1:5]) == 1
end

flexible_people = unique(filter(:responseID => (x -> is_rule_based(x)),results_df)[:,:responseID])

# need to find indices of flexible people for trace

participants = unique(results_df[:, :responseID])
flex_indices = [] # 15 flexible people
for index in 1:length(participants)
    id = participants[index]
    if is_rule_based(id)
        push!(flex_indices, index)
    end
end

#######################################

results = []
for (i,id) in enumerate(flexible_people)
    #for run in 1:5
    run=1
        # make new choicemap so that it is mutable (we can force individual type)
        observations = Gen.choicemap() # constraints
        # add in the observations besides individual_type 
        for choice in get_values_shallow(Gen.get_choices(traces[flex_indices[i]][run]))
            if choice[1] != :individual_type
                observations[choice[1]] = choice[2]
            end
        end   
        for choice in get_submaps_shallow(Gen.get_choices(traces[flex_indices[i]][run]))
            set_submap!(observations, choice[1], Gen.get_submap(get_choices(traces[flex_indices[i]][run]), choice[1]))
        end   
        #######################################################
        id_df = filter(:responseID => ==(id), results_df)

        amounts_offered = id_df[:,:amount_offered]
        damage_types = id_df[:,:damage_type]

        ests = []
        for ind_type in 1:3
            set_value!(observations, :individual_type, ind_type)
            trace, lml_est = Gen.importance_resampling(model_acceptance, (amounts_offered, damage_types), observations, 20)
            push!(ests, lml_est)
        end
        push!(results, ests)
    #end
end

println(results) #results is an array of marginal likelihood for each flexible person










