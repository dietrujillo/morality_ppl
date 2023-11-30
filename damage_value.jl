using CSV
using DataFrames: DataFrame, DataFrameRow, combine, median

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
_build_damage_table(compensation_demanded::DataFrame; 
                    combine_function=_combine_median)::DataFrameRow = combine_function(compensation_demanded)

const COMPENSATION_DEMANDED_FILEPATH = "data/data_wide_willing.csv"
const COMPENSATION_DEMANDED_TABLE = _build_damage_table(
    CSV.read(COMPENSATION_DEMANDED_FILEPATH, DataFrame)[:, VALID_DAMAGE_TYPES]
)

"""
    estimate_damage_value(damage::DamageType)

Estimate the utility or value of a given damage. The damage must be in the VALID_DAMAGE_TYPES 
and the return value is obtained from data collected from Levine et al. "When rules are over-ruled: Virtual bargaining as a contractualist
method of moral judgment." (2022).
"""
function estimate_damage_value(damage::DamageType)
    try
        return COMPENSATION_DEMANDED_TABLE[damage]
    catch e
        e isa ArgumentError && throw(ArgumentError("Invalid damage type provided. Allowed types are $VALID_DAMAGE_TYPES.")) || throw(e)
    end
end




