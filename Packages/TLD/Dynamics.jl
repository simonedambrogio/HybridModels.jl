include("Transition.jl")

# It stores the Dynamical Variables change when Transitioning fro s to sʼ 
struct Dynamics
    transition::Transition
    # Dynamics stores 8 values:
    # 4 dynamical variables associated to state s
    # 4 dynamical variables that change when transitioning to sʼ
    dynamics::Vector
end

get_transition(dy::Dynamics) = dy.transition
get_dynamics(dy::Dynamics) = dy.dynamics

function Base.show(io::IO, dy::Dynamics)
    print(io, string(dy.transition))
    print(io, "\n\n")
    
    for i in 1:4
        str = string(round(dy.dynamics[i] .* 1000)./1000)
        print(io, " " ^ (7-length(str)), str)
    end
    print(io, " " ^ 5, "(Dynamical Variables | s)")
    print(io, "\n")
    for i in 5:8
        str = string(round(dy.dynamics[i] .* 1000)./1000)
        print(io, " " ^ (7-length(str)), str)
    end
    print(io, " " ^ 5, "(Dynamical Variables | sʼ) - (Dynamical Variables | s)")
end

function Dynamics(transition::Transition, m)
    Dynamics(
        transition,
        values(preference_change(transition, m)) |> collect
    )
end;