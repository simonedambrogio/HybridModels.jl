path2root = dirname(Base.active_project());
include( joinpath(path2root, "scr", "scr", "utils.jl") );
using Functors, StatsFuns, JLD2, HybridModels;
include("../𝐷.jl"); include("../model.jl"); 
X, y = joinpath(path2root, "scr", "outcome", "test_data.jld2") |> load_object |> first;






@hybridmodel function m(X)
    # --- Transform and Extract Parameters --- #
    @kdc λ₀ = logit(0.99f0) ω = logit(0.60f0) κ₁ = logit(0.25f0) λ₂ = logit(0.01f0) τ = logit(0.08f0); 
    @ddc β = 1.2f0

    
    # --- Transform parameters --- #
    λ₀, ω, κ₁, λ₂, τ = σ(λ₀), σ(ω), σ(κ₁), σ(λ₂), σ(τ);

    # --- Compute α and β to establish the shape of the beta posterior distribution --- #
    nL = n0L(X) .+ n1_0L(X, ω)      .+ n1_1L(X) .+ n2_0L(X, λ₂) .+ n2_1L(X)
    rL = r0L(X) .+ r1_0L(X, λ₀, nL) .+ r1_1L(X) .+ r2_0L(X, λ₂) .+ r2_1L(X)

    nR = n0R(X) .+ n1_0R(X, ω)      .+ n1_1R(X) .+ n2_0R(X, λ₂) .+ n2_1R(X)
    rR = r0R(X) .+ r1_0R(X, λ₀, nR) .+ r1_1R(X) .+ r2_0R(X, λ₂) .+ r2_1R(X)

    # --- Value of Select --- #    
    ρL = rL ./ nL
    ρR = rR ./ nR

    # --- Value of Information --- #    
    voiL = ucb([nL nR]', β)'
    voiR = ucb([nR nL]', β)'

    # --- Cost of Information --- #    
    coiL = (1f0 .- X.gL) .* κ₁
    coiR = (1f0 .- X.gR) .* κ₁

    return [(voiL .- coiL) (voiR .- coiR) (ρL .- ρR) (ρR .- ρL)]' ./ τ
end;

m(X)


using Random, RobustNeuralNetworks
input_dim = 2;
rng = Xoshiro();
ny = 1
nh = fill(32,4)   # 4 hidden layers, each with 32 neurons
γ = 5  



@hybridmodel function mymodel(X)
    # --- Transform and Extract Parameters --- #
    @kdc λ₀ = 0.99f0 ω = 0.60f0 κ₁ = 0.25f0 λ₂ = 0.01f0 τ = 0.08f0 
    @ddc θ = DenseLBDNParams{Float32}(input_dim, nh, ny, γ; nl=Flux.tanh, learn_γ=true, rng)

    # --- Compute α and β to establish the shape of the beta posterior distribution --- #
    nL = n0L(X) .+ n1_0L(X, ω)      .+ n1_1L(X) .+ n2_0L(X, λ₂) .+ n2_1L(X)
    rL = r0L(X) .+ r1_0L(X, λ₀, nL) .+ r1_1L(X) .+ r2_0L(X, λ₂) .+ r2_1L(X)

    nR = n0R(X) .+ n1_0R(X, ω)      .+ n1_1R(X) .+ n2_0R(X, λ₂) .+ n2_1R(X)
    rR = r0R(X) .+ r1_0R(X, λ₀, nR) .+ r1_1R(X) .+ r2_0R(X, λ₂) .+ r2_1R(X)

    # --- Value of Select --- #    
    ρL = rL ./ nL
    ρR = rR ./ nR

    # --- Value of Information --- #    
    nn = LBDN(θ)
    voiL = nn([nL nR]')
    voiR = nn([nR nL]')

    # --- Cost of Information --- #    
    coiL = (1f0 .- X.gL) .* κ₁
    coiR = (1f0 .- X.gR) .* κ₁

    return [(voiL' .- coiL) (voiR' .- coiR) (ρL .- ρR) (ρR .- ρL)]' ./ τ
end;

mymodel(X)

mymodel.kdc
