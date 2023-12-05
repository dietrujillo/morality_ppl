module GenerativeModel

using Gen

function _kahneman_tversky_utility(x::Real, α::Real = 0.25; λ::Real = 2.25)::Real
    if x < 0
        return -λ * (-x)^α
    end
    return x^α
end

logistic(x::Real) = inv(exp(-x) + one(x))

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

@gen function model_acceptance(amounts_offered::Vector{Float64}, damage_types::Vector{Float64})

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
        damage_amount = damage_type#({(:damage_amount, i)} ~ estimate_damage_amount(damage_type))
        side_payment_fraction = {(i, :side_payment_fraction)} ~ estimate_side_payment_fraction(amount_offered)
        money_amount = amount_offered * min(1, max(0, side_payment_fraction))
        {(i, :acceptance)} ~ bernoulli(accept_probability(money_amount, damage_amount))
    end
end

export model_acceptance

end