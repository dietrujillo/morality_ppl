

using CairoMakie, DataFrames, AlgebraOfGraphics

function plot_creator(offer, train_data, results_df)
    train = filter(:amount_offered => ==(offer), train_data)
    test = filter(:amount_offered => ==(offer), results_df)

    train[!, :bargain_accepted] = convert.(Float64, train[:, :bargain_accepted])

    rename!(test,:predictions => :bargain_accepted)
    select!(test, Not([:labels]))
    insertcols!(train, :type => "train")
    insertcols!(test, :type => "test")

    plot=vcat(train,test)

    data(plot) * mapping(:damage_type, :bargain_accepted; color=:type => nonnumeric) *
                       visual(RainClouds; markersize=10, jitter_width=0.75, alpha=0.25, clouds=nothing, plot_boxplots=false) |> draw
end

plot_creator(10000,train_data, results_df)
