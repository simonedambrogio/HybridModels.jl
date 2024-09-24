# using Pkg; Pkg.update(); Pkg.precompile()
println("Load Libraries and Model...")
path2root = dirname(Base.active_project());
include( joinpath(path2root, "scr", "Modelling", "utils.jl") );
include( joinpath(path2root, "scr", "Modelling", "analysis", "scr", "fit", "recover", "utils.jl") );
using LinearAlgebra, OneHotArrays, JLD2, DataStructures, MLDataUtils, Optim, JSON, CSV, DataFrames;
using GLMakie, Random; # Random.seed!(123);
using StatsFuns, FluxOptTools
using NNlib: softmax, tanh
using HybridModels
# list all folders in the model folder
[
    include( joinpath(path2modellingscr, "model", file) ) 
    for file in readdir( joinpath(path2modellingscr, "model") ) |> filter(x -> !occursin("._", x))
];

subject, input2remove = 1, "n-blocked__ρ-same__ρ-other";
finalPath = @__DIR__; path2ht = nothing; tune_hyperparameters = false;

# finalPath, path2ht, tune_hyperparameters = "/Volumes/PROJECTS/Ongoing/HybridModellingRobust/scr/analysis/data/fit/multiple-runs/test", nothing, false
# finalPath = joinpath("/Volumes/PROJECTS/Ongoing/HybridModellingRobust/scr/analysis/data/fit/multiple-runs", subject);
var2remove = string.(split(input2remove, "__"));
idxinput=getidx(var2remove);


println("\nLoading Data...")
ct1 = CSV.read( joinpath( path2data, "preprocessed", "binary", "ct1.csv"), DataFrame );
trials = vcat([
    begin
        sbjdf = filter(r -> r.subject==subject, ct1);
        trials = vcat(
            [ # Trials
                sbjdf |> 
                filter(r -> r.trial==trial && r.event in ["switch", "stay", "select"] && r.visit>0) |> 
                df -> Trial(df) for trial in sbjdf.trial |> unique
            ]
        );
        trials
    end
    for subject in 1:10
]...);
d = vcat([create_dataframe(trial) for trial in trials]...);



println("Set Parameters")
λ₀ = 0.99f0 |> logit;
ω  = 0.60f0 |> logit;
κ₁ = 0.25f0 |> logit;
λ₂ = 0.01f0 |> logit;
τ  = 0.08f0 |> logit;
β  = 1.2f0;
generative_model = Agent( 
    KDC(λ₀, ω, κ₁, λ₂, τ),
    VoiUCB(β)
);


println("Simulate data")
pact = generative_model(d) |> softmax;
d.act = [rand( Distributions.Categorical(p[:]) ) for p in eachcol(pact)];
data = getXy(d; batchdim=60);
shuffle!(data);
(train_data, test_data) = splitobs(data, at=0.7);
# X, y = train_data[1];
data = (; train_data, test_data);

println("Randomly Initialize Parameters")
cognitive_model = Agent(KDC(), VoiUCB());

println("Optimize Parameters")
loss(model, X, y) = Flux.logitcrossentropy(model(X), y);
pars = cognitive_model |> m -> Flux.params( m.kdc, m.ddc );
filename = joinpath(finalPath, "fit-cognitivemodel.json"); iter = 500;
trainBFGS!(cognitive_model, data, loss, iter, pars, filename)
