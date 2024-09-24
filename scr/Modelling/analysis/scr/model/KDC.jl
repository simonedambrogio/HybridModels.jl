using Functors, HybridModels, StatsFuns, MacroTools
include("𝐷.jl")

struct KDCParams{T} <: HybridModels.AbstractParams
    λ₀::Vector{T}
    ω::Vector{T}
    κ₁::Vector{T}
    λ₂::Vector{T}
    τ::Vector{T}
end;

function Base.show(io::IO, m::KDCParams)
    # println(io, "Knowledge-Driven Component")
    println(io, "")
    println(io, "  λ₀: ", round(logistic(m.λ₀[1]), digits=3) )
    println(io, "  ω : ", round(logistic(m.ω[1]), digits=3) )
    println(io, "  κ₁: ", round(logistic(m.κ₁[1]), digits=3) )
    println(io, "  λ₂: ", round(logistic(m.λ₂[1]), digits=3) )
    println(io, "  τ:  ", round(logistic(m.τ[1]), digits=3) )
end


function extract(m::KDCParams)
    # println(io, "Knowledge-Driven Component")
    return Dict(
        "λ₀" => logistic(m.λ₀[1]),
        "ω" => logistic(m.ω[1]),
        "κ₁" => logistic(m.κ₁[1]),
        "λ₂" => logistic(m.λ₂[1]),
        "τ" => logistic(m.τ[1])
    )
end;

# Random Initialization
function KDCParams{T}() where T
    λ₀ =  abs.(randn(T, 1))
    κ₁ = -rand(T, 1) .- 1 
    ω  = abs.(randn(T, 1))
    λ₂ = -rand(T, 1) .- 4
    τ = -rand(T, 1) .- 1

    KDCParams{T}(λ₀, ω, κ₁, λ₂, τ)
end;

# Add a constructor without type parameter
KDCParams() = KDCParams{Float32}();

KDCParams{T}(λ₀::T, ω::T, κ₁::T, λ₂::T, τ::T) where T = KDCParams{T}([λ₀], [ω], [κ₁], [λ₂], [τ])

"""
    KDC{T}

A Knowledge-Driven Component (KDC) algorithm.

# Fields
- `θ::KDCParams{T}`: The parameters of the algorithm.
"""
struct KDC{T} <: HybridModels.AbstractKnowledgeDrivenComponent
    θ::KDCParams{T}
end;

# Add a constructor without type parameter

function KDC{T}() where T
    θ = KDCParams{T}()
    KDC{T}(θ);
end;

function KDC()
    p = KDCParams{Float32}()
    KDC{Float32}(p)
end


function KDC{T}(λ₀::Vector{T}, ω::Vector{T}, κ₁::Vector{T}, λ₂::Vector{T}, τ::Vector{T}) where T
    θ = KDCParams{T}(λ₀, ω, κ₁, λ₂, τ)
    KDC{T}(θ);
end;

function KDC{T}(λ₀::T, ω::T, κ₁::T, λ₂::T, τ::T) where T
    θ = KDCParams{T}(λ₀, ω, κ₁, λ₂, τ)
    KDC{T}(θ);
end;

KDC(λ₀, ω, κ₁, λ₂, τ) = KDC{Float32}(λ₀, ω, κ₁, λ₂, τ);

@functor KDCParams
@functor KDC

# f = VoiUCB(Float32[1.2]);
# m = Agent(f)

# m = KDC()

# p = KDCParams{Float32}() # Parameters
# m = KDC{Float32}(p)      # Voi


transformpars(m::KDC{T}) where T = KDCParams{T}(
    logistic(m.θ.λ₀[1]), 
    logistic(m.θ.ω[1]), 
    logistic(m.θ.κ₁[1]), 
    logistic(m.θ.λ₂[1]), 
    logistic(m.θ.τ[1])
);

# c = transformpars(kdc)


# utils --------------------------------
using DataFrames
f⁻(x::T, k::T, t::T) where T = x * exp(-k*t);

# --- Left --- #
# N
function n0L(m::Agent, X::𝐷{N}, c::KDCParams) where N 
    (2f0 .+ X.nL) .* X.initial_visit;
end;
function n1_0L(m::Agent, X::𝐷{N}, c::KDCParams) where N 
    (2f0 .+ (c.ω .* X.nL) .+ ((1f0.-c.ω) .* X.nB)) .* X.gR .* X.first_visit;
end;
function n1_1L(m::Agent, X::𝐷{N}, c::KDCParams) where N 
    (2f0 .+ X.nL) .* X.gL .* X.first_visit;
end;
function n2_0L(m::Agent, X::𝐷{N}, c::KDCParams) where N 
    (2f0 .+ f⁻.(X.nL, c.λ₂, X.t)) .* X.gR .* X.after_first_visit;
end;
function n2_1L(m::Agent, X::𝐷{N}, c::KDCParams) where N 
    (2f0 .+ X.nL) .* X.gL .* X.after_first_visit;
end;

# r
function r0L(m::Agent, X::𝐷{N}, c::KDCParams) where N 
    (2f0 .+ X.nL) .* 0.5f0  .* X.initial_visit;
end;
function r1_0L(m::Agent, X::𝐷{N}, c::KDCParams, nL::SVector{N, Float32}) where N 
    (nL .* (0.5f0 .+ (X.μR .- 0.5f0) .* c.λ₀)) .* X.gR .* X.first_visit
end;
function r1_0L(m::Agent, X::𝐷{N}, c::KDCParams, nL::Vector{Float32}) where N 
    (nL .* (0.5f0 .+ (X.μR .- 0.5f0) .* c.λ₀)) .* X.gR .* X.first_visit
end;
function r1_1L(m::Agent, X::𝐷{N}, c::KDCParams) where N 
    (1f0 .+ X.rL) .* X.gL .* X.first_visit
end;
function r2_0L(m::Agent, X::𝐷{N}, c::KDCParams) where N 
    (1f0 .+ f⁻.(X.rL,c.λ₂,X.t)) .* X.gR .* X.after_first_visit
end;
function r2_1L(m::Agent, X::𝐷{N}, c::KDCParams) where N 
    (1f0 .+ X.rL) .* X.gL .* X.after_first_visit
end;

# --- Right --- #
# N
function n0R(m::Agent, X::𝐷{N}, c::KDCParams) where N 
    (2f0 .+ X.nR) .* X.initial_visit;
end;

function n1_0R(m::Agent, X::𝐷{N}, c::KDCParams) where N 
    (2f0 .+ (c.ω .* X.nR) .+ (1f0.-c.ω) .* X.nB) .* X.gL .* X.first_visit
end;
function n1_1R(m::Agent, X::𝐷{N}, c::KDCParams) where N 
    (2f0 .+ X.nR) .* X.gR .* X.first_visit;
end;
function n2_0R(m::Agent, X::𝐷{N}, c::KDCParams) where N 
    (2f0 .+ f⁻.(X.nR, c.λ₂, X.t)) .* X.gL .* X.after_first_visit;
end;
function n2_1R(m::Agent, X::𝐷{N}, c::KDCParams) where N 
    (2f0 .+ X.nR) .* X.gR .* X.after_first_visit;
end;

# r
function r0R(m::Agent, X::𝐷{N}, c::KDCParams) where N 
    (2f0 .+ X.nR) .* 0.5f0  .* X.initial_visit;
end;
function r1_0R(m::Agent, X::𝐷{N}, c::KDCParams, nR::SVector{N, Float32}) where N 
    (nR .* (0.5f0 .+ (X.μL .- 0.5f0) .* c.λ₀)) .* X.gL .* X.first_visit
end;
function r1_0R(m::Agent, X::𝐷{N}, c::KDCParams, nR::Vector{Float32}) where N 
    (nR .* (0.5f0 .+ (X.μL .- 0.5f0) .* c.λ₀)) .* X.gL .* X.first_visit
end;
function r1_1R(m::Agent, X::𝐷{N}, c::KDCParams) where N 
    (1f0 .+ X.rR) .* X.gR .* X.first_visit
end;
function r2_0R(m::Agent, X::𝐷{N}, c::KDCParams) where N 
    (1f0 .+ f⁻.(X.rR,c.λ₂,X.t)) .* X.gL .* X.after_first_visit
end;
function r2_1R(m::Agent, X::𝐷{N}, c::KDCParams) where N 
    (1f0 .+ X.rR) .* X.gR .* X.after_first_visit
end;





# Function to Make Model Predictions and not to train ---
# --- Left --- #
# N
function n0L(m::Agent, X::DataFrame, c::KDCParams)
    (2f0 .+ X.nL) .* X.initial_visit;
end;
function n1_0L(m::Agent, X::DataFrame, c::KDCParams)
    (2f0 .+ (c.ω .* X.nL) .+ ((1f0.-c.ω) .* X.nB)) .* X.gR .* X.first_visit;
end;
function n1_1L(m::Agent, X::DataFrame, c::KDCParams)
    (2f0 .+ X.nL) .* X.gL .* X.first_visit;
end;
function n2_0L(m::Agent, X::DataFrame, c::KDCParams)
    (2f0 .+ f⁻.(X.nL, c.λ₂, X.t)) .* X.gR .* X.after_first_visit;
end;
function n2_1L(m::Agent, X::DataFrame, c::KDCParams)
    (2f0 .+ X.nL) .* X.gL .* X.after_first_visit;
end;

# r
function r0L(m::Agent, X::DataFrame, c::KDCParams)
    (2f0 .+ X.nL) .* 0.5f0  .* X.initial_visit;
end;
function r1_0L(m::Agent, X::DataFrame, c::KDCParams, nL::Vector{Float32})
    (nL .* (0.5f0 .+ (X.μR .- 0.5f0) .* c.λ₀)) .* X.gR .* X.first_visit
end;
function r1_1L(m::Agent, X::DataFrame, c::KDCParams)
    (1f0 .+ X.rL) .* X.gL .* X.first_visit
end;
function r2_0L(m::Agent, X::DataFrame, c::KDCParams)
    (1f0 .+ f⁻.(X.rL,c.λ₂,X.t)) .* X.gR .* X.after_first_visit
end;
function r2_1L(m::Agent, X::DataFrame, c::KDCParams)
    (1f0 .+ X.rL) .* X.gL .* X.after_first_visit
end;

# --- Right --- #
# N
function n0R(m::Agent, X::DataFrame, c::KDCParams)
    (2f0 .+ X.nR) .* X.initial_visit;
end;
function n1_0R(m::Agent, X::DataFrame, c::KDCParams)
    (2f0 .+ (c.ω .* X.nR) .+ (1f0.-c.ω) .* X.nB) .* X.gL .* X.first_visit
end;
function n1_1R(m::Agent, X::DataFrame, c::KDCParams)
    (2f0 .+ X.nR) .* X.gR .* X.first_visit;
end;
function n2_0R(m::Agent, X::DataFrame, c::KDCParams)
    (2f0 .+ f⁻.(X.nR, c.λ₂, X.t)) .* X.gL .* X.after_first_visit;
end;
function n2_1R(m::Agent, X::DataFrame, c::KDCParams)
    (2f0 .+ X.nR) .* X.gR .* X.after_first_visit;
end;

# r
function r0R(m::Agent, X::DataFrame, c::KDCParams)
    (2f0 .+ X.nR) .* 0.5f0  .* X.initial_visit;
end;
function r1_0R(m::Agent, X::DataFrame, c::KDCParams, nR::Vector{Float32})
    (nR .* (0.5f0 .+ (X.μL .- 0.5f0) .* c.λ₀)) .* X.gL .* X.first_visit
end;
function r1_1R(m::Agent, X::DataFrame, c::KDCParams)
    (1f0 .+ X.rR) .* X.gR .* X.first_visit
end;
function r2_0R(m::Agent, X::DataFrame, c::KDCParams)
    (1f0 .+ f⁻.(X.rR,c.λ₂,X.t)) .* X.gL .* X.after_first_visit
end;
function r2_1R(m::Agent, X::DataFrame, c::KDCParams)
    (1f0 .+ X.rR) .* X.gR .* X.after_first_visit
end;



# Function to Make Model Predictions and not to train ---
# --- Left --- #
# N
function n0L(m::Agent, X::NamedTuple, c::KDCParams)
    (2f0 .+ X.nL) .* X.initial_visit;
end;
function n1_0L(m::Agent, X::NamedTuple, c::KDCParams)
    (2f0 .+ (c.ω .* X.nL) .+ ((1f0.-c.ω) .* X.nB)) .* X.gR .* X.first_visit;
end;
function n1_1L(m::Agent, X::NamedTuple, c::KDCParams)
    (2f0 .+ X.nL) .* X.gL .* X.first_visit;
end;
function n2_0L(m::Agent, X::NamedTuple, c::KDCParams)
    (2f0 .+ f⁻.(X.nL, c.λ₂, X.t)) .* X.gR .* X.after_first_visit;
end;
function n2_1L(m::Agent, X::NamedTuple, c::KDCParams)
    (2f0 .+ X.nL) .* X.gL .* X.after_first_visit;
end;

# r
function r0L(m::Agent, X::NamedTuple, c::KDCParams)
    (2f0 .+ X.nL) .* 0.5f0  .* X.initial_visit;
end;
function r1_0L(m::Agent, X::NamedTuple, c::KDCParams, nL::Vector{Float32})
    (nL .* (0.5f0 .+ (X.μR .- 0.5f0) .* c.λ₀)) .* X.gR .* X.first_visit
end;
function r1_0L(m::Agent, X::NamedTuple, c::KDCParams, nL::Float32)
    (nL .* (0.5f0 .+ (X.μR .- 0.5f0) .* c.λ₀)) .* X.gR .* X.first_visit
end;
function r1_1L(m::Agent, X::NamedTuple, c::KDCParams)
    (1f0 .+ X.rL) .* X.gL .* X.first_visit
end;
function r2_0L(m::Agent, X::NamedTuple, c::KDCParams)
    (1f0 .+ f⁻.(X.rL,c.λ₂,X.t)) .* X.gR .* X.after_first_visit
end;
function r2_1L(m::Agent, X::NamedTuple, c::KDCParams)
    (1f0 .+ X.rL) .* X.gL .* X.after_first_visit
end;

# --- Right --- #
# N
function n0R(m::Agent, X::NamedTuple, c::KDCParams)
    (2f0 .+ X.nR) .* X.initial_visit;
end;
function n1_0R(m::Agent, X::NamedTuple, c::KDCParams)
    (2f0 .+ (c.ω .* X.nR) .+ (1f0.-c.ω) .* X.nB) .* X.gL .* X.first_visit
end;
function n1_1R(m::Agent, X::NamedTuple, c::KDCParams)
    (2f0 .+ X.nR) .* X.gR .* X.first_visit;
end;
function n2_0R(m::Agent, X::NamedTuple, c::KDCParams)
    (2f0 .+ f⁻.(X.nR, c.λ₂, X.t)) .* X.gL .* X.after_first_visit;
end;
function n2_1R(m::Agent, X::NamedTuple, c::KDCParams)
    (2f0 .+ X.nR) .* X.gR .* X.after_first_visit;
end;

# r
function r0R(m::Agent, X::NamedTuple, c::KDCParams)
    (2f0 .+ X.nR) .* 0.5f0  .* X.initial_visit;
end;
function r1_0R(m::Agent, X::NamedTuple, c::KDCParams, nR::Vector{Float32})
    (nR .* (0.5f0 .+ (X.μL .- 0.5f0) .* c.λ₀)) .* X.gL .* X.first_visit
end;
function r1_0R(m::Agent, X::NamedTuple, c::KDCParams, nR::Float32)
    (nR .* (0.5f0 .+ (X.μL .- 0.5f0) .* c.λ₀)) .* X.gL .* X.first_visit
end;
function r1_1R(m::Agent, X::NamedTuple, c::KDCParams)
    (1f0 .+ X.rR) .* X.gR .* X.first_visit
end;
function r2_0R(m::Agent, X::NamedTuple, c::KDCParams)
    (1f0 .+ f⁻.(X.rR,c.λ₂,X.t)) .* X.gL .* X.after_first_visit
end;
function r2_1R(m::Agent, X::NamedTuple, c::KDCParams)
    (1f0 .+ X.rR) .* X.gR .* X.after_first_visit
end;







# Function to Make Model Predictions and not to train ---
# --- Left --- #
# N
function n0L(X::DataFrame)
    (2f0 .+ X.nL) .* X.initial_visit;
end;
function n1_0L(X::DataFrame, ω)
    (2f0 .+ (ω .* X.nL) .+ ((1f0.-ω) .* X.nB)) .* X.gR .* X.first_visit;
end;
function n1_1L(X::DataFrame)
    (2f0 .+ X.nL) .* X.gL .* X.first_visit;
end;
function n2_0L(X::DataFrame, λ₂)
    (2f0 .+ f⁻.(X.nL, λ₂, X.t)) .* X.gR .* X.after_first_visit;
end;
function n2_1L(X::DataFrame)
    (2f0 .+ X.nL) .* X.gL .* X.after_first_visit;
end;

# r
function r0L(X::DataFrame)
    (2f0 .+ X.nL) .* 0.5f0  .* X.initial_visit;
end;
function r1_0L(X::DataFrame, λ₀, nL::Vector{T}) where T
    (nL .* (0.5f0 .+ (X.μR .- 0.5f0) .* λ₀)) .* X.gR .* X.first_visit
end;
function r1_1L(X::DataFrame)
    (1f0 .+ X.rL) .* X.gL .* X.first_visit
end;
function r2_0L(X::DataFrame, λ₂)
    (1f0 .+ f⁻.(X.rL,λ₂,X.t)) .* X.gR .* X.after_first_visit
end;
function r2_1L(X::DataFrame)
    (1f0 .+ X.rL) .* X.gL .* X.after_first_visit
end;

# --- Right --- #
# N
function n0R(X::DataFrame)
    (2f0 .+ X.nR) .* X.initial_visit;
end;
function n1_0R(X::DataFrame, ω)
    (2f0 .+ (ω .* X.nR) .+ (1f0.-ω) .* X.nB) .* X.gL .* X.first_visit
end;
function n1_1R(X::DataFrame)
    (2f0 .+ X.nR) .* X.gR .* X.first_visit;
end;
function n2_0R(X::DataFrame, λ₂)
    (2f0 .+ f⁻.(X.nR, λ₂, X.t)) .* X.gL .* X.after_first_visit;
end;
function n2_1R(X::DataFrame)
    (2f0 .+ X.nR) .* X.gR .* X.after_first_visit;
end;

# r
function r0R(X::DataFrame)
    (2f0 .+ X.nR) .* 0.5f0  .* X.initial_visit;
end;
function r1_0R(X::DataFrame, λ₀, nR::Vector{T}) where T
    (nR .* (0.5f0 .+ (X.μL .- 0.5f0) .* λ₀)) .* X.gL .* X.first_visit
end;
function r1_1R(X::DataFrame)
    (1f0 .+ X.rR) .* X.gR .* X.first_visit
end;
function r2_0R(X::DataFrame, λ₂)
    (1f0 .+ f⁻.(X.rR,λ₂,X.t)) .* X.gL .* X.after_first_visit
end;
function r2_1R(X::DataFrame)
    (1f0 .+ X.rR) .* X.gR .* X.after_first_visit
end;




function n0L(X::𝐷)
    (2f0 .+ X.nL) .* X.initial_visit;
end;
function n1_0L(X::𝐷, ω)
    (2f0 .+ (ω .* X.nL) .+ ((1f0.-ω) .* X.nB)) .* X.gR .* X.first_visit;
end;
function n1_1L(X::𝐷)
    (2f0 .+ X.nL) .* X.gL .* X.first_visit;
end;
function n2_0L(X::𝐷, λ₂)
    (2f0 .+ f⁻.(X.nL, λ₂, X.t)) .* X.gR .* X.after_first_visit;
end;
function n2_1L(X::𝐷)
    (2f0 .+ X.nL) .* X.gL .* X.after_first_visit;
end;

# r
function r0L(X::𝐷)
    (2f0 .+ X.nL) .* 0.5f0  .* X.initial_visit;
end;
function r1_0L(X::𝐷, λ₀, nL::Vector{T}) where T
    (nL .* (0.5f0 .+ (X.μR .- 0.5f0) .* λ₀)) .* X.gR .* X.first_visit
end;
function r1_1L(X::𝐷)
    (1f0 .+ X.rL) .* X.gL .* X.first_visit
end;
function r2_0L(X::𝐷, λ₂)
    (1f0 .+ f⁻.(X.rL,λ₂,X.t)) .* X.gR .* X.after_first_visit
end;
function r2_1L(X::𝐷)
    (1f0 .+ X.rL) .* X.gL .* X.after_first_visit
end;

# --- Right --- #
# N
function n0R(X::𝐷)
    (2f0 .+ X.nR) .* X.initial_visit;
end;
function n1_0R(X::𝐷, ω)
    (2f0 .+ (ω .* X.nR) .+ (1f0.-ω) .* X.nB) .* X.gL .* X.first_visit
end;
function n1_1R(X::𝐷)
    (2f0 .+ X.nR) .* X.gR .* X.first_visit;
end;
function n2_0R(X::𝐷, λ₂)
    (2f0 .+ f⁻.(X.nR, λ₂, X.t)) .* X.gL .* X.after_first_visit;
end;
function n2_1R(X::𝐷)
    (2f0 .+ X.nR) .* X.gR .* X.after_first_visit;
end;

# r
function r0R(X::𝐷)
    (2f0 .+ X.nR) .* 0.5f0  .* X.initial_visit;
end;
function r1_0R(X::𝐷, λ₀, nR::Vector{T}) where T
    (nR .* (0.5f0 .+ (X.μL .- 0.5f0) .* λ₀)) .* X.gL .* X.first_visit
end;
function r1_1R(X::𝐷)
    (1f0 .+ X.rR) .* X.gR .* X.first_visit
end;
function r2_0R(X::𝐷, λ₂)
    (1f0 .+ f⁻.(X.rR,λ₂,X.t)) .* X.gL .* X.after_first_visit
end;
function r2_1R(X::𝐷)
    (1f0 .+ X.rR) .* X.gR .* X.after_first_visit
end;

