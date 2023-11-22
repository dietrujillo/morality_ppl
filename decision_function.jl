"""
    decision = decision_function(amount_value, damage_value)

Decision function. This function will take estimated values for the amount offered and the damage taken,
and output a true/false value determining whether the individual chooses to accept the damage.

TODO: This is a bare-bones implementation and requires further refinement.
"""
function decision_function(amount_value::Real, damage_value::Real)
    return amount_value > damage_value
end
