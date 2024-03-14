module DataLoading

export splitdf, load_dataset, load_and_split, data_to_dict

using Random: shuffle
using CSV
using DataFrames
using StatsBase: sample

include("damage_type.jl")

DATA_PATH = "data/within-data.csv"
EXCLUDE_DATA_PATH = "data/exclusions.csv"

const OFFER_AS_INT_DICT = Dict(
    "hundred" => 100,
    "thousand" => 1000,
    "tenthousand" => 10000,
    "hunthousand" => 100000,
    "million" => 1000000,
)

function splitdf(df, pct, individual_analysis::Bool = true)
    @assert 0 <= pct <= 1

    if individual_analysis
        n = Int(pct*10)
        sel = Vector{Bool}(undef, nrow(df))
        fill!(sel, false)
        all_response_ids = [responseid for responseid in unique(df, [:responseID])[!,:responseID]] #305 total responders
        for responder in all_response_ids
            train_types = sample(VALID_DAMAGE_TYPES, n; replace=false)
            for damage in train_types
                indxs = findall((df.damage_type .== damage) .& (df.responseID .== responder))
                for indx in indxs
                sel[indx] = 1 
                end
            end
        end
    else
        ids = shuffle(collect(axes(df, 1)))
        sel = ids .<= nrow(df) .* pct
    end
    return view(df, sel, :), view(df, .!sel, :)
end

function load_dataset(data_path::String, exclusions::Bool=true)
    table = CSV.read(data_path, DataFrame)
    rename!(table, [:subjectcode, :answer, :question, :context] .=> [:responseID, :bargain_accepted, :amount_offered, :damage_type])

    #exclusions
    if exclusions
        exclusion_table = DataFrame(CSV.File(EXCLUDE_DATA_PATH))
        delete!(exclusion_table, [1,2,3,4,nrow(exclusion_table)])
        rename!(exclusion_table, [:Column2] .=> [:responseID])
        select!(exclusion_table, [:excluded, :responseID])
        excludeIDs = [id for id in dropmissing(exclusion_table, disallowmissing=true)[:,:responseID]]
        for id in excludeIDs
            table = filter!(:responseID => !=(id), table)
        end
    end

    table[!,:responseID] = convert.(String, table[:,:responseID])
    table[!,:damage_type] = Symbol.(table[:,:damage_type])
    table[!,:amount_offered] = convert.(Float64, table[:,:amount_offered])
    table[!,:bargain_accepted] = convert.(Bool, table[:,:bargain_accepted])

    return table
end

function load_and_split(data_path, individual_analysis::Bool = true)
    table = load_dataset(data_path)
    train_data, test_data = splitdf(table, 0.7, individual_analysis)
    return train_data, test_data
end

function onehot_encode(df, column)
    categories = sort(unique(df[:, column]))
    return select(transform(df, @. column => ByRow(isequal(categories)) .=> Symbol.(:ohe_, categories)), Not(column))
end

function data_to_dict(df, key_column, value_column)
    out = Dict()
    for key in unique(df[:, key_column])
        key_df = filter(key_column => (x -> x == key), df)
        value = convert(Vector, key_df[:, value_column])
        out[key] = value
    end
    return out
end

end # Module DataLoading
