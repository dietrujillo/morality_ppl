module MoralPPL_LogisticRegression

include("damage_value.jl")

using CSV
using DataFrames
using OneHotArrays
using Flux
using Statistics
using MLDatasets
using EvalMetrics: binary_eval_report
using Random: seed!
#using ScikitLearn

function load_test_train_data()
    table = make_bargain_table()
    #make condition col numeric, sort by condition (for stratify purposes)
    table = sort(transform(table, :condition => ByRow(x -> OFFER_AS_INT_DICT[x]) => :condition), :condition)
    flux_x_onehot = onehotbatch(table[:,4], VALID_DAMAGE_TYPES)
    conds = transpose(table[:,3])
    x = convert(Matrix{Int32},cat(flux_x_onehot,conds;dims=1)) # (11x2688)
    y = onehotbatch(convert(Vector{Int32},table[:,1]),[1,0]) # one hot encoding of yes/no, (2x2688)
    # split data for train and test
    function partitionTrainTest(x, y, n=180, seed::Int = 42) #n = num samples for test
        seed!(seed)
        idxs = cat(rand(1:381,n),rand(382:910,n),rand(910:1471,n),rand(1472:2070,n),rand(2071:2688,n);dims=1)
        train_x = x[:, setdiff(1:end, idxs)]
        train_y = y[:, setdiff(1:end, idxs)]
        test_x = x[:, idxs]
        test_y = y[:, idxs]
        return train_x, train_y, test_x, test_y 
    end
    train_x, train_y, test_x, test_y = partitionTrainTest(x,y)
end

# use Flux.onecold(y, classes) for results as strings
classes = [1,0]
# Flux regression model
flux_model = Chain(Dense(11 => 2), softmax)
# Flux loss function
function flux_loss(flux_model, x, y)
    ŷ = flux_model(x)
    Flux.logitcrossentropy(ŷ, y)
end;
# Flux accuracy function
flux_accuracy(x, y) = mean(Flux.onecold(flux_model(x), classes) .== y);

function fit_by_training(train_x,train_y, iters::Int=10000, eps::Float64=0.98)
    function train_flux_model()
        dLdm, _, _ = Flux.gradient(flux_loss, flux_model, train_x, train_y)
        @. flux_model[1].weight = flux_model[1].weight - 0.1 * dLdm[:layers][1][:weight]
        @. flux_model[1].bias = flux_model[1].bias - 0.1 * dLdm[:layers][1][:bias]
    end;
    # train 
    for i = 1:iters
        train_flux_model();
        flux_accuracy(train_x, train_y[1,:]) >= eps && break
    end
end

###############################################################################################

train_x, train_y, test_x, test_y = load_test_train_data()
#train
fit_by_training(train_x, train_y)
predictions = flux_model(test_x)
#test on test set
report = binary_eval_report(test_y[1,:],flux_model(test_x)[1,:])
print(report)

end  # Module MoralPPL_LogisticRegression