"""
    Damage(amount)

Creates a Damage object.

TODO: This implementation is a place-holder. Damage must have a damage type identifier and not
 a specific amount, as the amount must be inferred by the individuals.
"""
struct Damage
    amount::Real
end

"""
    estimate_damage_value(damage)

Estimate the value of a damage type.

TODO: This implementation is a place-holder. Damage estimation must consider factors such as 
cost of repairs, whether the damage is wanted, and sentimental value.
"""
function estimate_damage_value(damage::Damage)
    return damage.amount
end
