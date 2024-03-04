module GenerativeModel

using Gen

include("compensation_demanded.jl")
using .CompensationDemanded

export model_acceptance, PARAMETER_ADDRESSES, COMPENSATION_DEMANDED_TABLE

const PARAMETER_ADDRESSES = [
    :individual_type,
    :high_stakes_threshold,
    [((:damage_value, damage_type) => :damage_value) for damage_type in VALID_DAMAGE_TYPES]...
]

function _kahneman_tversky_utility(x::Real, α::Real = 0.25; λ::Real = 2.25)::Real
    if x < 0
        return -λ * (-x)^α
    end
    return x^α
end

logistic(x::Real) = inv(exp(-x) + one(x))
logistic(x::Real, bias::Real) = logistic(x + bias)

@gen function estimate_damage_value(damage_type::DamageType)
    mean, q90, median, iqr = COMPENSATION_DEMANDED_TABLE[:, damage_type]
    damage_value ~ normal(median, iqr)
    return damage_value
end

@gen function estimate_side_payment_fraction(amount_offered::Float64, damage_value::Float64)
    if amount_offered < damage_value
        fraction ~ normal(0.9, 0.1)
    elseif amount_offered - damage_value > 10
        fraction ~ normal(0.2, 0.5)
    else
        fraction ~ normal(0.5, 2)
    end
    return fraction
end

@dist multiplier_exponential(λ) = exponential(λ) + 1

const RULEBASED = 1
const FLEXIBLE = 2
const AGREEMENT = 3

@gen function model_acceptance(amounts_offered::Vector{Float64}, damage_types::Vector{DamageType})

    rule_based_prior ~ normal(1//3, 0.1)
    flexible_prior ~ normal(1//3, 0.1)
    agreement_based_prior ~ normal(1//3, 0.1)
    a = max(0,min(1,rule_based_prior))
    b = max(0,min(1,flexible_prior))
    c = max(0,min(1,agreement_based_prior))
    priors_vector = (1/(a+b+c))*[a, b, c]
    individual_type ~ categorical(priors_vector)
    is_rule_based = individual_type == 1
    is_flexible = individual_type == 2

    high_stakes_threshold ~ categorical([2//10, 3//10, 4//10, 1//10])
    threshold_values = [2, 10, 100, 1000]
    high_stakes_threshold_value = threshold_values[high_stakes_threshold]

    damage_values = Dict()
    for damage_type in VALID_DAMAGE_TYPES
        damage_values[damage_type] = {(:damage_value, damage_type)} ~ estimate_damage_value(damage_type)
    end

    function high_stakes(amounts_offered)
        return amounts_offered > high_stakes_threshold_value
    end

    @gen function accept_probability(amount_offered, damage_type)
        damage_value = damage_values[damage_type]
        money_value = amount_offered * max(0, min(1, estimate_side_payment_fraction(amount_offered, damage_value)))

        utility = _kahneman_tversky_utility(money_value - damage_value)

        if individual_type == RULEBASED || (individual_type == FLEXIBLE && !high_stakes(money_value))
            return {:accept} ~ bernoulli(0.)
        end

        return {:accept} ~ bernoulli(round(logistic(utility)))
    end

    for (i, (amount_offered, damage_type)) in enumerate(zip(amounts_offered, damage_types))
        {(:acceptance, i)} ~ accept_probability(amount_offered, damage_type)
    end
end

end