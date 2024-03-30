using DataFrames
using Plots
using StatsPlots: boxplot

function show_responses(data::DataFrame, responseID::String = "")
    if (responseID !== "")
        data = filter(:responseID => (x -> x == responseID), data)
    end
    p = Plots.plot()
    for damage_type in unique(data[:, :damage_type])
        damage_type_data = sort(filter(:damage_type => (x -> x == damage_type), data), order(:amount_offered))
        p = Plots.plot!(damage_type_data[:, :amount_offered], damage_type_data[:, :bargain_accepted])
    end
    return p      
end

function plot_priors_heatmap(simplex_points, y)
    m = Matrix{Float64}(undef, length(simplex_points), 4)
    m[:,1:3] .= permutedims(hcat(simplex_points...))
    m[:,4] = y
    ternary(m, show=true, image=true, region=(0,1,0,1,0,1), frame=(grid=0, ticks=0.5, annot=0), vertex_labels="Rule-Based/Flexible/Agreement-Based")
end

function get_amount_index(amount, offers)
    for (i, offer) in enumerate(offers)
        if offer == amount
            return i
        end
    end
    return -1
end

function plot_prediction_boxplots(
    damage_type::Symbol,
    predictions::Dict,
    amounts::Dict,
    damages::Dict,
    acceptances::Dict,
    results::Dict,
    offers::Vector{Float64} = [100., 1000., 10000., 100000., 1e6],
    plot_size::Tuple{Int64, Int64} = (1200, 300)
)  
    damage_plots = []    
    for (name, individual_type) in zip(["Rule-Based", "Flexible", "Agreement-Based"], 1:3)
        x, y, = [], []
        labels = [[] for _ in 1:5]

        for (k, v) in results
            amounts_indices = [get_amount_index(amount, offers) for amount in amounts[k]]
            if argmax(v.type_probs) == individual_type
                push!(x, amounts_indices[damages[k] .== damage_type]...)
                push!(y, predictions[k][damages[k] .== damage_type]...)
                for i in 1:5
                    push!(
                        labels[i],
                        acceptances[k][(damages[k] .== damage_type) .& (amounts_indices .== i)]...
                    )
                end
            end
        end
        plt = boxplot(x, y, xticks=(1:5, string.(offers)), legend=false, title=name, ylim=(0., 1.))
    
        meanlabel = []
        for v in labels
            if length(v) > 0
                push!(meanlabel, mean(v))
            else
                push!(meanlabel, 0)
            end
        end
        plt = Plots.plot!(1:5, meanlabel, xticks=(1:5, string.(offers)), legend=false, title=name, ylim=(0., 1.), linewidth=5, color="steelblue")
        push!(damage_plots, plt)
    end
    horizontal_layout = @layout([a b c])
    out = Plots.plot(damage_plots..., layout=horizontal_layout, size=plot_size, plot_title=String(damage_type), plot_titlevspan=0.1)
    return out
end
