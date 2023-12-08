module CompensationDemanded

using CSV
using DataFrames
using StatsBase: iqr, median

export DamageType, VALID_DAMAGE_TYPES, COMPENSATION_DEMANDED_TABLE

const DamageType = Symbol
const VALID_DAMAGE_TYPES::Vector{DamageType} = [
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
]

_combine_median(df::DataFrame)::DataFrameRow = combine(df, names(df) .=> median, renamecols=false)[1, :]
_combine_iqr(df::DataFrame)::DataFrameRow = combine(df, names(df) .=> iqr, renamecols=false)[1, :]
function _build_damage_table(compensation_demanded::DataFrame)::DataFrame
    out = DataFrame()
    push!(out, _combine_median(compensation_demanded))
    push!(out, _combine_iqr(compensation_demanded))
    return out
end

const COMPENSATION_DEMANDED_FILEPATH = "data/data_wide_willing.csv"
const COMPENSATION_DEMANDED_TABLE = _build_damage_table(
    CSV.read(COMPENSATION_DEMANDED_FILEPATH, DataFrame)[:, VALID_DAMAGE_TYPES]
)

end