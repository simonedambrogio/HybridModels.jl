using Random, RobustNeuralNetworks, Functors
using HybridModels


"""
    VoiNNParams{T}

Parameters for the Value of Information Neural Network (VoiNN) algorithm.

# Fields
- `θ::DenseLBDNParams`: Parameters for the Dense Lipschitz Bounded Deep Network (LBDN).
"""
struct VoiNNParams{T} <: HybridModels.AbstractParams
    θ::DenseLBDNParams{T}
end;


# Random Initialization
function VoiNNParams{T}(input_dim, nh, ny, γ; nl=Flux.tanh, learn_γ=true, rng) where T
    θ = DenseLBDNParams{T}(input_dim, nh, ny, γ; nl=nl, learn_γ=learn_γ, rng)
    VoiNNParams{T}(θ);
end;

# Add a constructor without type parameter
VoiNNParams(input_dim, nh, ny, γ; nl=Flux.tanh, learn_γ=true, rng) = VoiNNParams{Float32}(
    DenseLBDNParams{Float32}(input_dim, nh, ny, γ; nl=nl, learn_γ=learn_γ, rng)
);



"""
    VoiNN{T}

A Value of Information Neural Network (VoiNN) algorithm.

# Fields
- `θ::VoiNNParams{T}`: The parameters of the algorithm.
"""
struct VoiNN{T} <: HybridModels.AbstractDataDrivenComponent
    θ::VoiNNParams{T}
end;


"""
How to use this:

input_dim = 2
rng = Xoshiro();
ny = 1
nh = fill(32,4)   # 4 hidden layers, each with 32 neurons
γ = 5             # Start with a Lipschitz bound of 5

θ = VoiNNParams(input_dim, nh, ny, γ; nl=Flux.tanh, learn_γ=true, rng);
m = VoiNN(θ);
"""
function (m::VoiNN)(x::AbstractMatrix{T}) where T
    # Compute Value of Information
    nn = RobustNeuralNetworks.LBDN(m.θ.θ);
    return nn(x)
end;

@functor VoiNN
@functor VoiNNParams




