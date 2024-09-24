using Distributions, Flux, StaticArrays
path2root = dirname(Base.active_project());
path2data = joinpath(path2root, "data");
function update_global_directories(path2root)

    # Get all folders in the MyStructs directory
    mystruct_path = joinpath(path2root, "MyStructs")
    mystruct_folders = filter(isdir, readdir(mystruct_path, join=true))

    # Add each folder in MyStructs to LOAD_PATH if it doesn't already exist
    for folder in mystruct_folders
        if !(folder in LOAD_PATH)
            push!(LOAD_PATH, folder)
        end
    end

end;
update_global_directories(path2root)
using TLD


using Base.Filesystem: isdir, mkdir
"""
    create_unique_path(base_path::String)

Create a unique directory path. If the base path exists, appends '+' to the path name.
"""
function create_unique_path(base_path::String)
    new_path = base_path
    suffix = ""

    while isdir(new_path)
        suffix *= "+"
        new_path = base_path * suffix
    end

    mkpath(new_path)
    return new_path
end


vars2test = [
    "none", "n-same", "n-other", "n-blocked", "ρ-same", "ρ-other", "gaze", "visit", 
    "n-blocked__ρ-same__ρ-other", 
    "n-blocked__ρ-same__ρ-other__n-same", "n-blocked__ρ-same__ρ-other__n-other",
    "n-blocked__ρ-same__ρ-other__gaze", "n-blocked__ρ-same__ρ-other__visit", 
    "n-blocked__ρ-same__ρ-other__gaze__visit"
];

function getidx(var2remove::String)
    inputs = ["n-same", "n-other", "n-blocked", "ρ-same", "ρ-other", "gaze", "visit"];
    return .!(var2remove  .== inputs)
end;

function getidx(var2remove::Vector{String})
    inputs = ["n-same", "n-other", "n-blocked", "ρ-same", "ρ-other", "gaze", "visit"];
    return .![(ip in var2remove ? true : false) for ip in inputs]
end;

getdim(var2remove::String) = sum(getidx(var2remove));
getdim(var2remove::Vector{String}) = sum(getidx(var2remove));


# State(tmdp::TrialMDP) = State(0, [0,0], tmdp.green_dots, 0, [0,0], 1);


# ----- Generate UCB trials ----- #
# Action = Signed;
# isfirstvisit(s::State, a::Action) = iszero(s.visited[a]);
# isstay(s::State, a::Action) = s.gaze == a;
# isselect(a::Action) = a>2;

# function green2colored(s::State, a::Action, tmdp::TrialMDP)
    
#     red, visited = s.red, s.visited;
#     red[a] = rand(Binomial(tmdp.green_dots[a], tmdp.μ[a]))
#     visited[a] = 1 
#     State(
#         s.time_step+1,
#         red,
#         s.colored,
#         a,
#         visited,
#         1
#     )

# end;

# function transition_to_sʼ(s::State, a::Action, tmdp::TrialMDP) 

#     if isfirstvisit(s, a) # First Visit -> From Green to Colored the Meaning of "a" is Sample Left vs Sample Right
        
#         green2colored(s, a, tmdp)

#     else

#         if isstay(s, a)
#             red = copy(s.red)
#             red[a] += rand(Bernoulli(tmdp.μ[a]))
#             State(s.time_step+1, red, [s.colored[1:2]+onehot(a, 1:2); s.colored[3]], a, s.visited, 1)
#         else
#             State(s.time_step+1, s.red, s.colored, a, s.visited, 1)
#         end

#     end

# end;

# 𝑆ʼ(s::State{N}, a::Action, tmdp::TrialMDP) where N = isselect(a) ? s : transition_to_sʼ(s,a,tmdp)


# function simulate(m, s0::State{N}, tmdp::TrialMDP) where N
#     actions=1:4
#     s = deepcopy(s0)
#     a, v, tm = 0, 0, 0 # State, Action, Visit, time after switch
#     S, A, V, T, TAS = Vector{State}(undef,0), Vector{Action}(undef,0), Vector{Int64}(undef,0), Vector{Transition}(undef,0), Vector{Int64}(undef,0)
#     while true
        
#         X = (; s2nt(s)..., t=tm);
#         prob = ProbabilityWeights( softmax( m(X) )[:] )
#         a = StatsBase.sample(actions, prob) |> Action
#         push!(S, deepcopy(s)); push!(A, a); 
#         a>2 && (push!(T, Transition( deepcopy(s), a, deepcopy(s))); push!(V, v); push!(TAS, tm); break)
#         is_switch = s.gaze!=a;
#         v+=is_switch; push!(V, v)
#         push!(TAS, tm)
#         tm = (tm==0 || !is_switch) ? tm+1 : 0;
#         sʼ =  𝑆ʼ(s, a, tmdp)
#         t = Transition(deepcopy(s), a, deepcopy(sʼ));
#         push!(T, deepcopy(t));
#         s = deepcopy(sʼ)
#     end

#     return Trial(S, A, T, V, TAS, tmdp)
# end

# function simulate(m, f::Function, s0::State{N}, tmdp::TrialMDP) where N
#     actions=1:4
#     s = deepcopy(s0)
#     a, v, tm = 0, 0, 0 # State, Action, Visit, time after switch
#     S, A, V, T, TAS = Vector{State}(undef,0), Vector{Action}(undef,0), Vector{Int64}(undef,0), Vector{Transition}(undef,0), Vector{Int64}(undef,0)
#     while true
        
#         X = (; s2nt(s)..., t=tm);
#         prob = ProbabilityWeights( softmax( m(X, f) )[:] )
#         a = StatsBase.sample(actions, prob) |> Action
#         push!(S, deepcopy(s)); push!(A, a); 
#         a>2 && (push!(T, Transition( deepcopy(s), a, deepcopy(s))); push!(V, v); push!(TAS, tm); break)
#         is_switch = s.gaze!=a;
#         v+=is_switch; push!(V, v)
#         push!(TAS, tm)
#         tm = (tm==0 || !is_switch) ? tm+1 : 0;
#         sʼ =  𝑆ʼ(s, a, tmdp)
#         t = Transition(deepcopy(s), a, deepcopy(sʼ));
#         push!(T, deepcopy(t));
#         s = deepcopy(sʼ)
#     end

#     return Trial(S, A, T, V, TAS, tmdp)
# end

# ------- Transition Function ------ #
# "Generate s' given state and action"
# function Base.rand(mdp::TrialMDP, s::State, a::Action)
    
#     (a>2||a<1) && return println("Action must be 1 or 2")

#     if s.type==2
        
#         # If First Visit
#         if s.time_step==0
#             return 𝑆ʼ(s, a, lookuptable.mdp) |> rand
#         elseif iszero(s.visited[a])
#             n_green = Int(mdp.green_dots[a])
#             red = copy(s.red)
#             red[a] = sum( rand(Bernoulli(mdp.μ[a]), n_green) )
#             return State(s.time_step+1, red, s.colored, a, s.visited+gaze, s.type)
#         else 
#             o = zeros(2); o[a] = rand(Bernoulli(mdp.μ[a]))
#             delta_colored = length(s.colored)==2 ? gaze : vcat(gaze,0);
#             return State(s.time_step+1, s.red+o, s.colored+delta_colored, a, s.visited, s.type)
#         end

#     else
#         # -- Sample -- #
#         gaze = zeros(2); gaze[a]=1 # left: (1,0); right: (0,1)
        
#         # If First Visit
#         if iszero(s.visited[a])
#             n_green = Int(mdp.green_dots[a])
#             red = copy(s.red)
#             red[a] = sum( rand(Bernoulli(mdp.μ[a]), n_green) )
#             return State(s.time_step+1, red, s.colored, a, s.visited+gaze, s.type)
#         else 
#             o = zeros(2); o[a] = rand(Bernoulli(mdp.μ[a]))
#             delta_colored = length(s.colored)==2 ? gaze : vcat(gaze,0);
#             return State(s.time_step+1, s.red+o, s.colored+delta_colored, a, s.visited, s.type)
#         end
#     end

# end;

# # tmdp = TrialMDP(max_dots, [0.1, 0.9], [0, 0, 0], (α = 1, β = 1));
# # s = State(0, zeros(2), tmdp.green_dots[1:2], 0, zeros(2), 3);
# # mdp = MDP(0.01f0, 0.1f0, max_steps, max_dots, [[0., 0.], [0., 0.]], (α=1, β=1));
# "Generate s' given state and action"
# function Base.rand(mdp::MDP, tmdp::TrialMDP, s::State, a::Action)
    
#     (a>2||a<1) && return println("Action must be 1 or 2")
        
#     if s.type==2
        
#         # If First Visit
#         if s.time_step==0
#             return 𝑆ʼ(s, a, mdp) |> rand
#         elseif iszero(s.visited[a])
#             n_green = Int(tmdp.green_dots[a])
#             red = copy(s.red)
#             red[a] = sum( rand(Bernoulli(tmdp.μ[a]), n_green) )
#             return State(s.time_step+1, au(red, a), au(s.colored,a), 1, au(update_visit(s.visited, a), a), 2)
#         else 
#             o = zeros(2); o[1] = rand(Bernoulli(tmdp.μ[a]))

#             return State(s.time_step+1, au(s.red,a)+o, au(s.colored,a)+[1, 0], 1, update_visit(s.visited,a), 2)
#         end

#     else
#         # -- Sample -- #
#         gaze = zeros(2); gaze[a]=1 # left: (1,0); right: (0,1)
        
#         # If First Visit
#         if iszero(s.visited[a])
#             n_green = Int(mdp.green_dots[a])
#             red = copy(s.red)
#             red[a] = sum( rand(Bernoulli(mdp.μ[a]), n_green) )
#             return State(s.time_step+1, red, s.colored, a, s.visited+gaze, s.type)
#         else 
#             o = zeros(2); o[a] = rand(Bernoulli(mdp.μ[a]))
#             delta_colored = length(s.colored)==2 ? gaze : vcat(gaze,0);
#             return State(s.time_step+1, s.red+o, s.colored+delta_colored, a, s.visited, s.type)
#         end
#     end

# end;

# # === Model === #
# function s2nt(s::State) # Covert State in NamedTuple
#     rL, rR = s.red;
#     bL, bR = (s.colored[1:2]-s.red) .* s.visited;
#     nL, nR, nB, N = (s.colored..., sum(s.colored));
#     gL, gR = s.gaze==0 ? (0, 0) : s.gaze==1 ? (1, 0) : (0, 1);
#     vL, vR = s.visited;
#     initial_visit = Float32(vL==0 && vR==0);
#     first_visit = Float32(vL+vR == 1);
#     after_first_visit = Float32(vL+vR > 1);
#     greenL = s.colored[1] * Int(sum(s.visited)==0);
#     greenR = s.colored[2] * Int(sum(s.visited)==0);
#     μL = mean( Beta(first(s.red)+1, first(s.colored[1:2] .- s.red) * first(s.visited)+1 ) );
#     μR = mean( Beta(last(s.red)+1, last(s.colored[1:2] .- s.red) * last(s.visited)+1 ) );
#     (; 
#         greenL=Float32(greenL), greenR=Float32(greenR),
#         nL=Float32(nL), nR=Float32(nR), nB=Float32(nB), N=Float32(N), 
#         rL=Float32(rL), rR=Float32(rR), 
#         bL=Float32(bL), bR=Float32(bR), 
#         gL=Float32(gL), gR=Float32(gR), 
#         vL=Float32(vL), vR=Float32(vR), initial_visit=Float32(initial_visit), 
#         first_visit, after_first_visit,
#         μL=Float32(μL), μR=Float32(μR)
#     )
# end

# observation(t::Transition) = Float32(t.sʼ.red[t.a] - t.s.red[t.a]);

# function s02h(s::State)
#     s.time_step!=0 && @error "State is not s0"
#     nL, nR, nB = s.colored .|> Float32;
#     [nL 0f0 1f0; nR 0f0 1f0; nB -1f0 0f0]
# end;

# function t2x(t::Transition)
    
#     (t.s.time_step==0) && return zeros32(3, 3); # All green
    
#     s = t.s;
#     rL, rR = s.red;
#     nL, nR, _ = s.colored;
#     gL, gR = t.sʼ.gaze==1 ? (1f0, 0f0) : (0f0, 1f0);
#     v = Float32(sum(t.sʼ.visited)-1);
#     [nL*gL rL*gL gL; nR*gR rR*gR gR; 0f0 v observation(t)]


# end;

# function s2x(s::State)
    
#     rL, rR = s.red .+ ones(Float32,2);
#     nL, nR, nB = s.colored .+ 2 .*ones(Float32,3);
#     gL, gR = s.gaze==0 ? (0f0, 0f0) : s.gaze==1 ? (1f0, 0f0) : (0f0, 1f0);
#     v = Float32(sum(s.visited)-1);
#     [nL rL gL; nR rR gR; nB v 0f0]

# end

# ----------- Define MDP ----------- #
# mutable struct Evidence
#     N::Float32  # Total Colored dots
#     α::Float32  # Success (Red dots)
#     v::Bool     # Visited 
# end;

# function t2dict(t::Transition) # Covert State in NamedTuple

#     s = t.s;
#     rL, rR = s.red;
#     # bL, bR = (s.colored[1:2]-s.red) .* s.visited;
#     nL, nR, nB = s.colored;
#     gaze = s.gaze==0 ? "initial" : s.gaze==1 ? "left" : "right";
#     vL, vR = s.visited;
    
#     Dict(
#             "left"    => Evidence(Float32(nL), Float32(rL), Bool(vL)),
#             "right"   => Evidence(Float32(nR), Float32(rR), Bool(vR)),
#             "blocked" => (; N=Float32(nB)),
#             "gaze"    => gaze,
#             "o"       => observation(t),
#             "first sample" => true
#         )

# end;

# function t2dict(t::Transition, x::Dict) # Covert State in NamedTuple
    
#     current_gaze = t.s.gaze==1 ? "left" : "right";
#     previous_gaze = x["gaze"];
    
#     if previous_gaze != current_gaze # Switch
        
#         s = t.s;
#         rL, rR = s.red;
#         nL, nR, _ = s.colored;

#         if current_gaze=="left"
#             Dict(
#                 "left"    => Evidence(Float32(nL), Float32(rL), true),
#                 "right"   => x["right"],
#                 "blocked" => x["blocked"],
#                 "gaze"    => current_gaze,
#                 "o"       => observation(t),
#                 "first sample" => false
#             )
#         else 
#             Dict(
#                 "left"    => x["left"],
#                 "right"   => Evidence(Float32(nR), Float32(rR), true),
#                 "blocked" => x["blocked"],
#                 "gaze"    => current_gaze,
#                 "o"       => observation(t),
#                 "first sample" => false
#             )
#         end
#     else

#         Dict(
#                 "left"    => x["left"],
#                 "right"   => x["right"],
#                 "blocked" => x["blocked"],
#                 "gaze"    => current_gaze,
#                 "o"       => observation(t),
#                 "first sample" => false
#             )

#     end


# end;

# 𝑄(s::State, m, 𝛩) = m(s, 𝛩);


# function sim(model, mdp)

#     s, a, v, tas = State(mdp), 0, 0, 0 # State, Action, Visit, time after switch
#     S, A, V, T, TAS = Vector{State}(undef,0), Vector{Action}(undef,0), Vector{Int64}(undef,0), Vector{Transition}(undef,0), Vector{Int64}(undef,0)

#     while true
        
#         X = s2𝐷(s, tas); 
#         prob = ProbabilityWeights( vec(softmax( MArray(model(X)) )) )
#         a = StatsBase.sample(actions, prob)
#         push!(S, s); push!(A, a); 
#         a>2 && (push!(T, Transition(s, a, s)); push!(V, v); push!(TAS, tas+1); break)
#         is_switch = s.gaze!=a;
#         v+=is_switch; push!(V, v)
#         tas = is_switch ? 0 : tas+1;
#         sʼ = rand(mdp, s, a)
#         t = Transition(s, a, sʼ);
#         push!(T, t);
#         push!(TAS, tas)
#         s=sʼ

#     end
#     return Trial(S, A, T, V, TAS, mdp)
# end

# simulate(model, trial::Trial) = sim(model, trial.mdp)
# simulate(model) = sim(model, TrialMDP())

# Operations on NamedTuple s with the same keys

# mean
# function Statistics.mean(tuples::AbstractVector{<:NamedTuple})::NamedTuple
#     @assert !isempty(tuples) "Array of NamedTuples should not be empty"
#     reference_keys = keys(first(tuples))
#     for t in tuples
#         @assert keys(t) == reference_keys "Field names of all NamedTuples should match"
#     end
    
#     means = (; [(k => mean(getfield(t, k) for t in tuples)) for k in reference_keys]...)
#     return means
# end

# function merge(v::Vector)
#     if isempty(v)
#         return NamedTuple()
#     end

#     keys_first = keys(v[1])
#     # for t in v[2:end]
#     #     if sort(keys(t)) != sort(keys_first)
#     #         throw(ArgumentError("All NamedTuples must have the same keys"))
#     #     end
#     # end

#     merged_values = Dict(k => [getproperty(t, k) for t in v] for k in keys_first)
#     return NamedTuple(merged_values)
# end




# function get_t(X)
#     # Compute t. t represents the number of samples of the attended option 
#     # after the last switch.
#     t = zeros(length(first(X)));
#     # gL==gR means that it is the the start of the trial. So t can stay 0.
#     for i in axes(t,1)
#         if X.gL[i] != X.gR[i] 
#             dwell = Int(X.gL[i-1] == X.gL[i]);
#             t[i] = t[i-1] * dwell + 1 
#         end
#     end
#     return t
# end;

# function s2𝐷(s::State, time::Int)
#     nL, nR, nB, N = (Float32.(s.colored)..., sum(Float32, s.colored));
#     rL, rR = Float32.(s.red);
#     gL, gR = s.gaze==0 ? (0f0, 0f0) : s.gaze==1 ? (1f0, 0f0) : (0f0, 1f0);
#     vL, vR = Float32.(s.visited);
#     initial_visit = Float32(vL==0 && vR==0);
#     first_visit = Float32(vL+vR == 1);
#     after_first_visit = Float32(vL+vR > 1);
#     μL = mean(Beta(first(s.red)+1, first(s.colored[1:2] .- s.red) * first(s.visited)+1 ) ) |> Float32 ;
#     μR = mean(Beta(last(s.red)+1, last(s.colored[1:2] .- s.red) * last(s.visited)+1 ) ) |> Float32;
#     𝐷(SA[nL], SA[nR], SA[nB], SA[N], SA[rL], SA[rR], SA[gL], SA[gR], SA[initial_visit], SA[first_visit], SA[after_first_visit], SA[μL], SA[μR], SA[Float32(time)])
# end

# function trials2dataframe(trials::Vector{Trial})
#     states = vcat([trial.s for trial in trials]...);
#     t = vcat([trial.timeafterswitch for trial in trials]...) .|> Float32;
    
#     ln = length(states);
#     nL, nR, nB, N, rL, rR, gL, gR, initial_visit, first_visit, after_first_visit, μL, μR = [zeros(Float32, ln) for _ in 1:13];
    
#     for (i, s) in enumerate(states)
#         rL[i], rR[i] = Float32.(s.red);
#         nL[i], nR[i], nB[i], N[i] = (Float32.(s.colored)..., sum(Float32, s.colored));
#         gL[i], gR[i] = s.gaze==0 ? (0f0, 0f0) : s.gaze==1 ? (1f0, 0f0) : (0f0, 1f0);
#         vL, vR = Float32.(s.visited);
#         initial_visit[i] = Float32(vL==0 && vR==0);
#         first_visit[i] = Float32(vL+vR == 1);
#         after_first_visit[i] = Float32(vL+vR > 1);
#         μL[i] = mean( Beta(first(s.red)+1, first(s.colored[1:2] .- s.red) * first(s.visited)+1 ) );
#         μR[i] = mean( Beta(last(s.red)+1, last(s.colored[1:2] .- s.red) * last(s.visited)+1 ) );
    
#     end
    
#     X = (;
#         nL, nR, nB, N, rL, rR, gL, gR, initial_visit, first_visit,
#         after_first_visit, μL, μR, t
#     ) |> DataFrame;

#     return X
# end

