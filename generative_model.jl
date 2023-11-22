"""
    generativemodel(offered_amount, damage)

The generative model for the contractualist model for rule-breaking from Levine et al. "When rules are over-ruled: Virtual bargaining as a contractualist
method of moral judgment." (2022). At its core, it consists of a decision function that takes the estimated values
of the offered amount and the potential damage.
"""
function generative_model(offered_amount::Real, damage::Damage)
    amount_value = estimate_amount_value(offered_amount)
    damage_value = estimate_damage_value(damage)
    return decision_function(amount_value, damage_value)
end
