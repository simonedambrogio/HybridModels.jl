

function ucb(x::AbstractMatrix{T}, β::T) where T
    # Check if x has exactly two rows
    if size(x, 1) != 2
        throw(ArgumentError("Input matrix x must have exactly 2 rows, got $(size(x, 1))"))
    end
    
    # Compute Value of Information
    N  = x[1,:]' + x[2,:]'
    Na = x[1,:]'
    return sqrt.( log.(N) ./ Na ) .* β
end;

# utils --------------------------------
using DataFrames
f⁻(x::T, k::T, t::T) where T = x * exp(-k*t);

# --- Left --- #
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

