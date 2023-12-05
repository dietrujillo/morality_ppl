module GenerativeModel

using Gen
using CSV
using DataFrames
using StatsBase: iqr, median

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

function _kahneman_tversky_utility(x::Real, α::Real = 0.25; λ::Real = 2.25)::Real
    if x < 0
        return -λ * (-x)^α
    end
    return x^α
end

logistic(x::Real) = inv(exp(-x) + one(x))

@gen function estimate_damage_value(damage_type::DamageType)
    median, iqr = COMPENSATION_DEMANDED_TABLE[:, damage_type]
    damage_value ~ HomogeneousMixture(normal, [0, 0])(
        dirichlet([1., 1., 1.]),
        [normal(median, 1), normal(median, 1), normal(median, 1)],
        [normal(iqr, 1), normal(iqr, 1), normal(iqr, 1)]
    )
    return damage_value
end

@gen function estimate_side_payment_fraction(amount_offered::Float64)
    fraction ~ HomogeneousMixture(normal, [0, 0])(
        dirichlet([1., 1., 1.]),
        [uniform(0, .5), 0.5, uniform(.5, 1)], 
        [uniform(0, 1), uniform(0, 1), uniform(0, 1)]
    )
    return fraction
end

@dist multiplier_exponential(λ) = exponential(λ) + 1
@gen function unreasonable_multiplier()
    multiplier ~ multiplier_exponential(uniform(0, 1))
    return multiplier
end

@gen function model_acceptance(amounts_offered::Vector{Float64}, damage_types::Vector{DamageType})

    min_utility_threshold ~ uniform(0, 10)
    max_cost_threshold ~ uniform(1000, 100000)
    unreasonable_neighbor_multiplier ~ unreasonable_multiplier()

    function accept_probability(money_amount, damage_amount)
        if damage_amount > max_cost_threshold
            return 0
        end

        utility = _kahneman_tversky_utility(money_amount - damage_amount)
        if utility < (min_utility_threshold * unreasonable_neighbor_multiplier)
            return 0
        end

        return logistic(utility)
    end

    for (i, (amount_offered, damage_type)) in enumerate(zip(amounts_offered, damage_types))
        damage_amount = ({(:damage_value, i)} ~ estimate_damage_value(damage_type))
        side_payment_fraction = {(i, :side_payment_fraction)} ~ estimate_side_payment_fraction(amount_offered)
        money_amount = amount_offered * min(1, max(0, side_payment_fraction))
        {(i, :acceptance)} ~ bernoulli(accept_probability(money_amount, damage_amount))
    end
end

export model_acceptance

end