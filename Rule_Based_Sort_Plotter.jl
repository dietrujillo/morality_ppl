
using CairoMakie, DataFrames, AlgebraOfGraphics

rule_based = []
not_rule_based = []

participants = unique(train_data[:, :responseID])

#sort the participants into rule based and not rule based catergory based on threshold = 0.5
for (index, participant) in enumerate(participants)
    rule_based_p = mean(trace[:rule_based_individual_p] for trace in traces[index]) #average over runs
    if rule_based_p >= 0.5
        push!(rule_based, participant)
    else
        push!(not_rule_based, participant)
    end
end

all_human_data = load_dataset(DATA_PATH)

all_rule_based = DataFrame()
all_not_rule_based = DataFrame()
#sort human data by rule based
for id in rule_based
    global all_rule_based = vcat(all_rule_based, filter(:responseID => ==(id), all_human_data))
end

for id in not_rule_based
    global all_not_rule_based = vcat(all_not_rule_based, filter(:responseID => ==(id), all_human_data))
end

#############################################################################################################

#PLOT CREATING FUNCTION

function plot_creator(offer, rule_based::Bool=true)

    if rule_based
        plot = filter(:amount_offered => ==(offer),all_rule_based)
    else
        plot = filter(:amount_offered => ==(offer),all_not_rule_based)
    end
    data(plot) * mapping(:damage_type, :bargain_accepted;) *
                       visual(RainClouds; markersize=10, jitter_width=0.75, alpha=0.25, clouds=nothing, plot_boxplots=false) |> draw
end

plot_creator(1000)






