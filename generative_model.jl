module GenerativeModel

using Gen

include("compensation_demanded.jl")
using .CompensationDemanded

export model_acceptance, PARAMETER_ADDRESSES, COMPENSATION_DEMANDED_TABLE

const PARAMETER_ADDRESSES = [
    :min_utility_threshold,
    :max_cost_threshold,
    :unreasonable_p,
    :unreasonable_neighbor_λ,
    :rule_based_individual_p,
    [((:damage_value, damage_type) => :damage_value) for damage_type in VALID_DAMAGE_TYPES]...
]

function _kahneman_tversky_utility(x::Real, α::Real = 0.25; λ::Real = 2.25)::Real
    if x < 0
        return -λ * (-x)^α
    end
    return x^α
end

logistic(x::Real) = inv(exp(-x) + one(x))

@gen function estimate_damage_value(damage_type::DamageType)
    mean, q90, median, iqr = COMPENSATION_DEMANDED_TABLE[:, damage_type]
    damage_value ~ HomogeneousMixture(normal, [0, 0])(
        dirichlet([1., 1., 1.]),
        [normal(q90, 1), normal(median, 1), normal(mean, 1)],
        [normal(iqr, 1), normal(iqr, 1), normal(iqr, 1)]
    )
    return damage_value
end

@gen function estimate_side_payment_fraction(amount_offered::Float64, damage_value::Float64)
    fraction ~ HomogeneousMixture(normal, [0, 0])(
        [1, 1, 1],
        [0.2, 0.5, 0.8], 
        [0.5, 2, 0.5]
    )
    return fraction
end

@dist multiplier_exponential(λ) = exponential(λ) + 1

@gen function model_acceptance(amounts_offered::Vector{Float64}, damage_types::Vector{DamageType})

    rule_based_individual_p ~ uniform(0, 1)
    min_utility_threshold ~ uniform(0, 5)
    max_cost_threshold ~ uniform(1000, 100000)

    unreasonable_p ~ uniform(0, 1)
    unreasonable_neighbor_λ ~ uniform(0, 1)

    damage_values = Dict()
    for damage_type in VALID_DAMAGE_TYPES
        damage_values[damage_type] = {(:damage_value, damage_type)} ~ estimate_damage_value(damage_type)
    end

    @gen function accept_probability(amount_offered, damage_type)
        damage_value = damage_values[damage_type]
        side_payment_fraction ~ estimate_side_payment_fraction(amount_offered, damage_value)
        money_value = amount_offered * min(1, max(0.1, side_payment_fraction))
        
        is_rule_based ~ bernoulli(rule_based_individual_p)

        if is_rule_based && damage_value > max_cost_threshold
            return {:accept} ~ bernoulli(0.)
        end

        utility = _kahneman_tversky_utility(money_value - damage_value)
        unreasonable_neighbor ~ bernoulli(unreasonable_p)
        if unreasonable_neighbor
            multiplier ~ multiplier_exponential(unreasonable_neighbor_λ)
        else
            multiplier = 1
        end

        if is_rule_based && utility < (min_utility_threshold * multiplier)
            return {:accept} ~ bernoulli(0.)
        end

        return {:accept} ~ bernoulli(round(logistic(utility)))
    end

    for (i, (amount_offered, damage_type)) in enumerate(zip(amounts_offered, damage_types))
        {(:acceptance, i)} ~ accept_probability(amount_offered, damage_type)
    end
end

end