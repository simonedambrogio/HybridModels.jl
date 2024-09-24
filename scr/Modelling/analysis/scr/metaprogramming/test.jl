println("Load Libraries and Model...")
path2root = dirname(Base.active_project());
path2modellingscr = joinpath(path2root, "scr", "Modelling", "analysis", "scr");
include( joinpath(path2root, "scr", "Modelling", "utils.jl") );
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



cognitive_model = Agent(KDC(), VoiUCB());

function model(X)

    # --- Transform and Extract Parameters --- #
    @kdc λ₀ = 0.99f0;
    @kdc ω  = 0.60f0;
    @kdc κ₁ = 0.25f0;
    @kdc λ₂ = 0.01f0;
    @kdc τ  = 0.08f0;
    β  = 1.2f0;

    # --- Compute α and β to establish the shape of the beta posterior distribution --- #
    nL = n0L(X) .+ n1_0L(X, ω)      .+ n1_1L(X) .+ n2_0L(X, λ₂) .+ n2_1L(X)
    rL = r0L(X) .+ r1_0L(X, λ₀, nL) .+ r1_1L(X) .+ r2_0L(X, λ₂) .+ r2_1L(X)

    nR = n0R(X) .+ n1_0R(X, ω)      .+ n1_1R(X) .+ n2_0R(X, λ₂) .+ n2_1R(X)
    rR = r0R(X) .+ r1_0R(X, λ₀, nR) .+ r1_1R(X) .+ r2_0R(X, λ₂) .+ r2_1R(X)

    # --- Value of Select --- #    
    ρL = rL ./ nL
    ρR = rR ./ nR

    # --- Value of Information --- #    
    # voiL = m.ddc([nL nR]')[:]
    # voiR = m.ddc([nR nL]')[:]

    # --- Cost of Information --- #    
    coiL = (1f0 .- X.gL) .* c.κ₁
    coiR = (1f0 .- X.gR) .* c.κ₁

    return [(voiL .- coiL) (voiR .- coiR) (ρL .- ρR) (ρR .- ρL)]' ./ c.τ
end;


