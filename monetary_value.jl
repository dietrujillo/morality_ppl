"""
    estimate_amount_value(amount)

Estimate the value of a monetary amount.

TODO: This implementation is naïve. The value of a monetary amount is not linear w.r.t. the amount,
but has diminishing returns.
"""
function estimate_amount_value(amount)
    diminishing_returns_function = identity
    return diminishing_returns_function(amount)
end
