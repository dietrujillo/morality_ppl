using CSV
using DataFrames: DataFrame, DataFrameRow, combine, median, Not, select, All, nrow, subset

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


##########################################################

const SplitDamage = Symbol
const SPLIT_DAMAGE_TYPES::Vector{SplitDamage} = [

    :bluemailbox,
    :blueoutsidedoor,
    :bluehouse,
    :cuttree,
    :breakwindows,
    :razehouse,
    :bleachlawn, 
    :blueinsidedoor,
    :erasemural,
    :smearpoop,

    :bluemailbox2,
    :blueoutsidedoor2,
    :bluehouse2,
    :cuttree2,
    :breakwindows2,
    :razehouse2,
    :bleachlawn2, 
    :blueinsidedoor2,
    :erasemural2,
    :smearpoop2,

    :condition
]
# To account for how cols are damagetype followed by a 2 (e.g. cuttree2), just making 
# a dict so we can still pass in the damage type without a 2. 
const COL_DICT = Dict(
    :bluemailbox => :bluemailbox2,
    :blueoutsidedoor => :blueoutsidedoor2,
    :bluehouse => :bluehouse2,
    :cuttree => :cuttree2,
    :breakwindows => :breakwindows2,
    :razehouse => :razehouse2,
    :bleachlawn => :bleachlawn2, 
    :blueinsidedoor => :blueinsidedoor2,
    :erasemural => :erasemural2,
    :smearpoop => :smearpoop2,
)
# dict to map offer amount (String) to int
const OFFER_AS_INT_DICT = Dict(
    "hundred" => 100,
    "thousand" => 1000,
    "tenthousand" => 10000,
    "hunthousand" => 100000,
    "million" => 1000000
)

SPLIT_TABLE = DataFrame(CSV.read("downloads/morality_ppl/data_wide_bargain.csv", DataFrame)[:,SPLIT_DAMAGE_TYPES])

"""
get_split(damage_type::string, amount_offered::string)

based on the damage type and the amount offered, returns median amount of 
money (based on wide bargain data file) to be given to the neighbor.

NOTE: cols are damagetype followed by a 2. e.g. cuttree2

"""
function get_split(damage::DamageType, amount_offered::String)
    damage_key = COL_DICT[damage]
    sorted_by_offer = filter(:condition => ==(amount_offered), SPLIT_TABLE)
    sorted_by_damage = select(sorted_by_offer, [damage,damage_key,:condition])
    # get rid of rows where side payment is greater than offer amount
    subset(sorted_by_damage, damage_key => x -> x .<= OFFER_AS_INT_DICT[amount_offered])

    # bernoulli on yes/no "Acceptable to do the damage?" probability yes = fraction of 1's to 0's in above DF

end

