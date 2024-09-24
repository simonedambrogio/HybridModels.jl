# ----------- Define MDP ----------- #
using Base: @kwdef
@kwdef struct MDP
    cost_stay::Float32 = 0.01f0
    cost_switch::Float32 = 0.1f0
    max_steps::Int = 10
    max_dots::Int = 10
    green_dots::Vector = [0.1, 0.1] # Left Right
    prior::NamedTuple = (; α=1, β=1)
end;
