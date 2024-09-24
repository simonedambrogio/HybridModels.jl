using DataFrames, Statistics, Distributions
include("State.jl")
include("TrialMDP.jl")
include("Transition.jl")
# Action = Int64;
# Action = Int32;
# Terminal State
# Terminal = Int64; ⊥ = -1;

# ----------- Define Trial ----------- #
struct Trial{T}
    s::Vector{State{T}}
    a::Vector{T}
    t::Vector{Transition{State{T}, T}}
    visit::Vector{T}
    timeafterswitch::Vector{T}
    mdp::TrialMDP
end

function Base.show(io::IO, t::Trial)
    print(io, "\nNumber of Samples: ", length(t.s), "\nSelection: ", last(t.a), "\nNumber of Visits: ", last(t.visit))
end

Base.getindex(t::Trial, idx::BitVector) = Trial(t.s[idx], t.a[idx], t.t[idx], t.visit[idx], t.timeafterswitch[idx], t.mdp)

function Trial(df::DataFrame)
    length(unique(df.subject)) != 1 && error("The df input can only include 1 subject")
    length(unique(df.session)) != 1 && error("The df input can only include 1 session")
    length(unique(df.trial))   != 1 && error("The df input can only include 1 trial")

    statesVec     = State(df)
    actionsVec    = [(df.mouse_position[1:end-1] .== "right") .+ 1; df.choice[1]+3]
    transitionVec = [[Transition(statesVec[i], actionsVec[i], statesVec[i+1]) for i in 1:(length(statesVec)-1)]; Transition(statesVec[end], actionsVec[end], statesVec[end])]
    visitedVec    = df.visit
    timeafterswitch = s2time(statesVec)
    
    return Trial(statesVec, actionsVec, transitionVec, visitedVec, timeafterswitch, TrialMDP(df[1,:]))
end

function DataFrames.DataFrame(trial::Trial)
    states = trial.s;
    t = trial.timeafterswitch .|> Float32;
    
    ln = length(states);
    nL, nR, nB, N, rL, rR, gL, gR, initial_visit, first_visit, after_first_visit, μL, μR = [zeros(Float32, ln) for _ in 1:13];
    
    for (i, s) in enumerate(states)
        rL[i], rR[i] = Float32.(s.red);
        nL[i], nR[i], nB[i], N[i] = (Float32.(s.colored)..., sum(Float32, s.colored));
        gL[i], gR[i] = s.gaze==0 ? (0f0, 0f0) : s.gaze==1 ? (1f0, 0f0) : (0f0, 1f0);
        vL, vR = Float32.(s.visited);
        initial_visit[i] = Float32(vL==0 && vR==0);
        first_visit[i] = Float32(vL+vR == 1);
        after_first_visit[i] = Float32(vL+vR > 1);
        μL[i] = mean( Beta(first(s.red)+1, first(s.colored[1:2] .- s.red) * first(s.visited)+1 ) );
        μR[i] = mean( Beta(last(s.red)+1, last(s.colored[1:2] .- s.red) * last(s.visited)+1 ) );
    
    end
    
    X = (;
        nL, nR, nB, N, rL, rR, gL, gR, initial_visit, first_visit,
        after_first_visit, μL, μR, t, act=trial.a
    ) |> DataFrame;

    return X
end
