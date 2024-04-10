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

function plot_priors_heatmap(simplex_points, y, cont=10, ylabel="Log Likelihood")
    m = Matrix{Float64}(undef, length(simplex_points), 4)
    m[:,1:3] .= permutedims(hcat(simplex_points...))
    m[:,4] = y

    ternary(
        m, image=true, 
        region=(
            min(first.(simplex_points)...),max(first.(simplex_points)...),
            min([x[2] for x in simplex_points]...), max([x[2] for x in simplex_points]...),
            min(last.(simplex_points)...),max(last.(simplex_points)...),
        ), 
        frame=(grid=0, ticks=0, annot=0), 
        labels=("|", "|", "|"), 
        vertex_labels="Rule-Based/Resource-Rational/Agreement-Based",
        contour=if cont != 0 (cont=cont, annot=1) else nothing end,
        log=true
    )
    colorbar!(
        show=true,
        frame=(grid=0, ticks=0, annot=:auto),
        pos=(paper=true, anchor=(7, -1), size=(12,0.5), justify=:TC, horizontal=true),
        ylabel=ylabel
    )
end

function get_amount_index(amount, offers)
    for (i, offer) in enumerate(offers)
        if offer == amount
            return i
        end
    end
    return -1
end

function plot_prediction_boxplots(name, key, value, results, amounts, offers, predictions, damage_type, damages, acceptances)
    x, y, = [], []
    labels = [[] for _ in 1:5]
    n_samples = 0
    for (k, v) in results
        amounts_indices = [get_amount_index(amount, offers) for amount in amounts[k]]
        if argmax(v[key]) == value
            n_samples += 1
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
    plt = boxplot(x, y, xticks=(1:5, string.(offers)), legend=false, title="$name - N=$n_samples", ylim=(0., 1.))
    return plt, labels
end

function plot_mean_line(plt, labels, offers)
    meanlabel = []
    for v in labels
        if length(v) > 0
            push!(meanlabel, mean(v))
        else
            push!(meanlabel, 0)
        end
    end
    plt = Plots.plot!(plt, 1:5, meanlabel, xticks=(1:5, string.(offers)), legend=false, ylim=(0., 1.), linewidth=5, color="steelblue")
    return plt
end

function plot_types(
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
    for (name, individual_type) in zip(["Rule-Based", "Resource-Rational", "Agreement-Based"], 1:3)
        plt, labels = plot_prediction_boxplots(name, :type_probs, individual_type, results, amounts, offers, predictions, damage_type, damages, acceptances)
        plt = plot_mean_line(plt, labels, offers)
        push!(damage_plots, plt)
    end
    horizontal_layout = @layout([a b c])
    out = Plots.plot(damage_plots..., layout=horizontal_layout, size=plot_size, plot_title=String(damage_type), plot_titlevspan=0.1)
    return out
end

function plot_thresholds(
    damage_type::Symbol,
    predictions::Dict,
    amounts::Dict,
    damages::Dict,
    acceptances::Dict,
    results::Dict,
    offers::Vector{Float64} = [100., 1000., 10000., 100000., 1e6],
    plot_size::Tuple{Int64, Int64} = (800, 1200)
)  
    damage_plots = []    
    for (name, threshold) in zip(
        [
            "Rule-Based", "Agreement-Based",
            "Resource-Rational - 10²", "Resource-Rational - 10³", "Resource-Rational - 10⁴", "Resource-Rational - 10⁵",
        ], [1, 6, 2, 3, 4, 5])
        plt, labels = plot_prediction_boxplots(name, :threshold_probs, threshold, results, amounts, offers, predictions, damage_type, damages, acceptances)
        plt = plot_mean_line(plt, labels, offers)
        push!(damage_plots, plt)
    end
    
    horizontal_layout = @layout([a b ; c d ; e f])
    out = Plots.plot(damage_plots..., layout=horizontal_layout, size=plot_size, plot_title=String(damage_type))
    return out
end