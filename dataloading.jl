using Random: shuffle
using CSV
using DataFrames

include("damage_type.jl")

DATA_PATH = "data/data_wide_bargain.csv"

const OFFER_AS_INT_DICT = Dict(
    "hundred" => 100,
    "thousand" => 1000,
    "tenthousand" => 10000,
    "hunthousand" => 100000,
    "million" => 1000000,
)

function splitdf(df, pct)
    @assert 0 <= pct <= 1
    ids = shuffle(collect(axes(df, 1)))
    sel = ids .<= nrow(df) .* pct
    return view(df, sel, :), view(df, .!sel, :)
end

function load_dataset(data_path::String)
    table = CSV.read(data_path, DataFrame)
    table = stack(table, VALID_DAMAGE_TYPES)
    rename!(table, [:variable, :value] .=> [:damage_type, :bargain_accepted])
    table = filter(row -> row[:condition] in keys(OFFER_AS_INT_DICT), table)

    table[:, :amount_offered] = map(x -> OFFER_AS_INT_DICT[x], table[:, :condition])


    table[!,:damage_type] = Symbol.(table[:,:damage_type])
    table[!,:amount_offered] = convert.(Float64, table[:,:amount_offered])
    table[!,:bargain_accepted] = convert.(Bool, table[:,:bargain_accepted])

    return table[:, [:damage_type, :amount_offered, :bargain_accepted]]
end

function load_and_split(data_path)
    table = load_dataset(data_path)
    train_data, test_data = splitdf(table, 0.7)
    return train_data, test_data
end

function onehot_encode(df, column)
    categories = sort(unique(df[:, column]))
    return select(transform(df, @. column => ByRow(isequal(categories)) .=> Symbol.(:ohe_, categories)), Not(column))
end;
