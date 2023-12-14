module MoralPPL_LogisticRegression

include("generative_model.jl")
include("damage_value.jl")

using CSV
using DataFrames
using Gen
using OneHotArrays
using Flux
using Statistics
using MLDatasets
using EvalMetrics: binary_eval_report

using .GenerativeModel

export model_acceptance, load_dataset, fit, predict, VALID_DAMAGE_TYPES, COMPENSATION_DEMANDED_TABLE, PARAMETER_ADDRESSES

const OFFER_AS_INT_DICT = Dict(
    "hundred" => 100,
    "thousand" => 1000,
    "tenthousand" => 10000,
    "hunthousand" => 100000,
    "million" => 1000000,
)

table = make_bargain_table()
#make condition col numeric, sort by condition
table = sort(transform(table, :condition => ByRow(x -> OFFER_AS_INT_DICT[x]) => :condition), :condition)

# condition  amount of rows
# 100        381
# 1000       528
# 10k        561
# 100k       599
# 1 mil      619

flux_x_onehot = onehotbatch(table[:,4], VALID_DAMAGE_TYPES)
conds = transpose(table[:,3])

x = convert(Matrix{Int32},cat(flux_x_onehot,conds;dims=1)) # (11x2688)
y = onehotbatch(convert(Vector{Int32},table[:,1]),[1,0]) # one hot encoding of yes/no, (2x2688)

# split data for train and test

function partitionTrainTest(x,y,n=180) 
    idxs = cat(rand(1:381,n),rand(382:910,n),rand(910:1471,n),rand(1472:2070,n),rand(2071:2688,n);dims=1)
    train_x = x[:, setdiff(1:end, idxs)]
    train_y = y[:, setdiff(1:end, idxs)]
    test_x = x[:, idxs]
    test_y = y[:, idxs]
    return train_x, train_y, test_x, test_y 
end

train_x, train_y, test_x, test_y = partitionTrainTest(x,y)

classes = [1,0]
# use Flux.onecold(y, classes) for results as strings

flux_model = Chain(Dense(11 => 2), softmax)

# loss function
function flux_loss(flux_model, x, y)
    ŷ = flux_model(x)
    Flux.logitcrossentropy(ŷ, y)
end;

# accuracy function
flux_accuracy(x, y) = mean(Flux.onecold(flux_model(x), classes) .== y);

# Training
function train_flux_model()
    dLdm, _, _ = Flux.gradient(flux_loss, flux_model, train_x, train_y)
    @. flux_model[1].weight = flux_model[1].weight - 0.1 * dLdm[:layers][1][:weight]
    @. flux_model[1].bias = flux_model[1].bias - 0.1 * dLdm[:layers][1][:bias]
end;

# set num of epochs and epsilon
for i = 1:10000
    train_flux_model();
    flux_accuracy(train_x, train_y[1,:]) >= 0.98 && break
end

#Use model on test set
print("Accuracy:",flux_accuracy(test_x, test_y[1,:]))

report = binary_eval_report(test_y[1,:],Flux.onecold(flux_model(test_x), classes))

end  # Module MoralPPL_LogisticRegression