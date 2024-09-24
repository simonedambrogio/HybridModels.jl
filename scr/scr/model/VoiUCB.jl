using Functors
using HybridModels
"""
    VoiUCBParams{T}

Parameters for the Value of Information Upper Confidence Bound (UCB) algorithm.

# Fields
- `β::Vector{T}`: A vector of parameters used in the calculation of the UCB.
"""
struct VoiUCBParams{T} <: HybridModels.AbstractParams
    β::Vector{T}
end;

# Random Initialization
function VoiUCBParams{T}() where T
    β = rand(T, 1) .+ 0.5
    VoiUCBParams{T}(β);
end;

function Base.show(io::IO, m::VoiUCBParams)
    # println(io, "Knowledge-Driven Component")
    println(io, "")
    println(io, "  β: ", round(m.β[1], digits=3) )
end


# Add a constructor without type parameter
VoiUCBParams() = VoiUCBParams{Float32}();



"""
    VoiUCB{T}

A Value of Information Upper Confidence Bound (UCB) algorithm.

# Fields
- `θ::VoiUCBParams{T}`: The parameters of the algorithm.
"""
struct VoiUCB{T} <: HybridModels.AbstractDataDrivenComponent
    θ::VoiUCBParams{T}
end;


# Add a constructor without type parameter
VoiUCB() = VoiUCB{Float32}();

function VoiUCB{T}() where T
    θ = VoiUCBParams{T}()
    VoiUCB{T}(θ);
end;

function VoiUCB{T}(β::Vector{T}) where T
    θ = VoiUCBParams{T}(β)
    VoiUCB{T}(θ);
end;

# Add this new constructor
VoiUCB(β::Vector{T}) where T = VoiUCB{T}(β)
VoiUCB(β::Real) = VoiUCB([β])



"""
How to use this:

m = VoiUCB()
or
p = VoiUCBParams{Float32}() # Parameters
m = VoiUCB{Float32}(p)      # Voi
or
m = VoiUCB(Float32[1.2])

x = rand(Float32, 2, 10) .* 100
m(x)

"""
function (m::VoiUCB)(x::AbstractMatrix{T}) where T
    # Check if x has exactly two rows
    if size(x, 1) != 2
        throw(ArgumentError("Input matrix x must have exactly 2 rows, got $(size(x, 1))"))
    end
    
    # Compute Value of Information
    N  = x[1,:]' + x[2,:]'
    Na = x[1,:]'
    return sqrt.( log.(N) ./ Na ) .* m.θ.β
end;

@functor VoiUCB
@functor VoiUCBParams



