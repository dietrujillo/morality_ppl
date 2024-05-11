
using Measures

function get_amount_index(amount, offers)
    for (i, offer) in enumerate(offers)
        if offer == amount
            return i
        end
    end
    return -1
end 

# GRAPH VISUAL PARAMS:
legend_position = :topleft #false for no legend
ylims = (-0.1,1.1)
model_line_opacity = 0.4
line_width = 5
margin = 6mm
model_line_style = :dash
rule_based_legend_pos = :topleft
offerlabels = ["10²", "10³", "10⁴", "10⁵", "10⁶"]
colors_list = ["steelblue", "crimson"]

function plot_prediction_boxplots(name, key, value, results, amounts, offers, predictions, damage_types, damages, acceptances)
    plt = nothing
    color_index = 1
    for damage_type in damage_types
        x, y, = [], []
        labels = [[] for _ in 1:5]
        n_samples = 0
        for (k, v) in results # for each id and its info
            amounts_indices = [get_amount_index(amount, offers) for amount in amounts[k]] # [1,1,1,...5,5,5]
            if argmax(v[key]) == value # if threshold of person is threshold plot is for (if individual is of this type)
                n_samples += 1
                push!(x, amounts_indices[damages[k] .== damage_type]...) 
                push!(y, predictions[k][damages[k] .== damage_type]...)
                for i in 1:5 # for each offer amount, find what if this id accepted or not
                    push!(
                        labels[i],
                        acceptances[k][(damages[k] .== damage_type) .& (amounts_indices .== i)]...
                    )
                end
            end
        end
        # get averages and error bar value to plot from box plot code
        meanlabel=[]
        for i in 1:5
            valid = []
            for j in 1:length(x)
                if x[j] == i
                    push!(valid,y[j])
                end
            end
            push!(meanlabel,mean(valid))
        end

        if value == 1
            legend_pos = rule_based_legend_pos
        else
            legend_pos = false
        end

        if isnothing(plt)
            plt = Plots.plot(1:5, meanlabel, xticks=(1:5, offerlabels), legend=legend_pos, 
                                label=[string(damage_type)*"- Model"],
                                title="$name - N=$n_samples", ylim=ylims, linewidth=line_width, 
                                color=colors_list[color_index], linealpha=model_line_opacity, linestyle = model_line_style)
            plt = plot_mean_line(plt, labels, offers, color_index, damage_type, value)
        else
            plt = Plots.plot!(1:5, meanlabel, xticks=(1:5, offerlabels), legend=legend_pos, 
                                label=[string(damage_type)*" - Model"],
                                title="$name - N=$n_samples", ylim=ylims, linewidth=line_width, 
                                color=colors_list[color_index], linealpha=model_line_opacity, linestyle = model_line_style)
            plt = plot_mean_line(plt, labels, offers, color_index, damage_type, value)
        end
        color_index += 1
    end
    return plt
end

function plot_mean_line(plt, labels, offers, color_index, damage_type, value)
    meanlabel = []
    for v in labels
        if length(v) > 0
            push!(meanlabel, mean(v))
        else
            push!(meanlabel, 0)
        end
    end
    if value == 1
        legend_pos = rule_based_legend_pos
    else
        legend_pos = false
    end
    plt = Plots.plot!(plt, 1:5, meanlabel, xlabel = "Offer Amount", ylabel= "Probability of Acceptance",
            label = [string(damage_type)*" - Observed"],
            xticks=(1:5, offerlabels), legend=legend_pos, ylim=ylims, linewidth=line_width, color=colors_list[color_index], margin = margin)
    return plt
end

function plot_thresholds(
    damage_types, # now a list of damage types
    predictions::Dict,
    amounts::Dict,
    damages::Dict,
    acceptances::Dict,
    results::Dict,
    offers::Vector{Float64} = [100., 1000., 10000., 100000., 1e6],
    plot_size::Tuple{Int64, Int64} = (1600, 800)
)  
    damage_plots = []    
    for (name, threshold) in zip(
        [
            "Rule-Based", "Agreement-Based",
            "Resource Rational: 10²", "Resource Rational: 10³", "Resource Rational: 10⁴", "Resource Rational: 10⁵",
        ], [1, 6, 2, 3, 4, 5])
        plt = plot_prediction_boxplots(name, :threshold_probs, threshold, results, amounts, offers, predictions, damage_types, damages, acceptances)
        push!(damage_plots, plt)
    end
    
    horizontal_layout = @layout([_ a b _ ; c d e f])
    out = Plots.plot(damage_plots..., layout=horizontal_layout, size=plot_size)
    return out
end

plot = plot_thresholds(
        [:bluemailbox,:breakwindows],
        final_predictions[model_key],
        amounts,
        damages,
        acceptances,
        Dict(valid_results)[model_key])
display(plot)
savefig(plot, "MeanLinePlots_bluemailbox_breakwindow") 

#= VALID_DAMAGE_TYPES= [
    :bluemailbox,
    :blueoutsidedoor,
    :bluehouse,
    :cuttree,
    :breakwindows,
    :razehouse,
    :bleachlawn, 
    :blueinsidedoor,
    :erasemural,
    :smearpoop
] =#