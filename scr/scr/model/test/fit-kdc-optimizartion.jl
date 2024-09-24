using Optim, NNlib
using Flux: logitcrossentropy;

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

θinit = [
    -(rand(Float32) + 1), # κ₁
    randn(Float32) |> abs, # ω
    -(rand(Float32) - 1),  # τ
    -(rand(Float32) + 4), # λ₂
    randn(Float32) |> abs, # λ₀
    # randn(Float32) |> abs, # β
];

# Modify the loss function to take θ and return a scalar
function loss(θ)
    predictions = (m)(θ, X)
    return logitcrossentropy(predictions, y)
end



# Create an optimization problem
opt_prob = Optim.optimize(loss, θinit, LBFGS())

# Get the optimized parameters
θ_opt = Optim.minimizer(opt_prob)

# Print the results
println("Optimization results:")
println("Minimized loss: ", Optim.minimum(opt_prob))
println("Optimized parameters: ", σ.(θ_opt))



