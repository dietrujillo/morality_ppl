module GenerativeModel

using Gen

include("compensation_demanded.jl")
using .CompensationDemanded

export model_acceptance, get_parameter_addresses, COMPENSATION_DEMANDED_TABLE, RULEBASED, FLEXIBLE, AGREEMENT

const RULEBASED = 1
const FLEXIBLE = 2
const AGREEMENT = 3

function get_parameter_addresses(responseIDs, damage_types = VALID_DAMAGE_TYPES) 
    return [
        :rule_based_prior,
        :flexible_prior,
        :agreement_prior,
        [((:individual, responseID) => :individual_type) for responseID in responseIDs]...,
        [((:individual, responseID) => :high_stakes_threshold) for responseID in responseIDs]...,
        [((:individual, responseID) => :logistic_bias) for responseID in responseIDs]...,
        Iterators.flatten([[((:individual, responseID) => (:damage_value, damage_type) => :damage_value) for responseID in responseIDs] for damage_type in damage_types])...
    ]
end

@gen function estimate_damage_value(damage_type::DamageType)
    mean, q90, median, iqr = COMPENSATION_DEMANDED_TABLE[:, damage_type]
    damage_value ~ normal(median, 0.)
    return damage_value
end

function _kahneman_tversky_utility(x::Real, α::Real = 0.25; λ::Real = 2.25)::Real
    if x < 0
        return -λ * (-x)^α
    end
    return x^α
end

logistic(x::Real) = inv(exp(-x) + one(x))
logistic(x::Real, bias::Real) = logistic(x + bias)

@gen function model_acceptance(amounts_offered, damage_types, response_ids)
    
    rule_based_prior ~ uniform(0, 1)
    flexible_prior ~ uniform(0, 1)
    agreement_prior ~ uniform(0, 1)
    priors_vector = [rule_based_prior, flexible_prior, agreement_prior]
    priors_vector = priors_vector / sum(priors_vector)

    @gen function individual_model(priors_vector, amounts_offered, damage_types)
        individual_type ~ categorical(priors_vector)

        high_stakes_threshold ~ categorical([1//4, 1//4, 1//4, 1//4])
        threshold_values = [100, 1000, 10000, 100000]
        high_stakes_threshold_value = threshold_values[high_stakes_threshold]

        logistic_bias ~ normal(0, 5)

        damage_values = Dict()
        for damage_type in VALID_DAMAGE_TYPES
            damage_values[damage_type] = {(:damage_value, damage_type)} ~ estimate_damage_value(damage_type)
        end

        function high_stakes(amount_offered)
            return amount_offered > high_stakes_threshold_value
        end

        @gen function accept_probability(amount_offered, damage_type)
            damage_value = damage_values[damage_type]

            utility = _kahneman_tversky_utility(amount_offered - damage_value)

            if individual_type == RULEBASED || (individual_type == FLEXIBLE && !high_stakes(amount_offered))
                return {:accept} ~ bernoulli(0.)
            end

            return {:accept} ~ bernoulli(logistic(utility, logistic_bias))
        end

        for (i, (amount_offered, damage_type)) in enumerate(zip(amounts_offered, damage_types))
            {(:acceptance, i)} ~ accept_probability(amount_offered, damage_type)
        end
    end

    for responseID in response_ids
        ind_amounts_offered = amounts_offered[responseID]
        ind_damage_types = damage_types[responseID]
        {(:individual, responseID)} ~ individual_model(priors_vector, ind_amounts_offered, ind_damage_types)
    end

end

end  # Module GenerativeModel