using DataFrames, Statistics, OneHotArrays

struct 𝐷{T}
    nL::Vector{T}
    nR::Vector{T}
    nB::Vector{T}
    rL::Vector{T}
    rR::Vector{T}
    gL::Vector{T}
    gR::Vector{T}
    initial_visit::Vector{T}
    first_visit::Vector{T}
    after_first_visit::Vector{T}
    μL::Vector{T}
    μR::Vector{T}
    t::Vector{T}
end

function 𝐷(df::DataFrame, batchdim::Int)
    return 𝐷([Vector{Float32}(df[:,f]) for f in fieldnames(𝐷)]...)
end

function DataFrame(x::𝐷{T}) where T
    DataFrame(NamedTuple{fieldnames(𝐷)}( [getfield(x, f) for f in fieldnames(𝐷)] ))
end

function 𝐷2df(x, idx=nothing)

    if isnothing(idx)
        DataFrame(NamedTuple{fieldnames(𝐷)}( [getfield(x, f) for f in fieldnames(𝐷)] ))
    else
        DataFrame(NamedTuple{fieldnames(𝐷)}( [getfield(x, f)[idx] for f in fieldnames(𝐷)] ))
    end
end

Base.length(d::𝐷) = length(d.nL);
function Base.getindex(d::𝐷, idx::Int)
    fields = typeof(d) |> fieldnames;
    𝐷( [[getfield(d, f)[idx]] for f in fields]... )
end

function Base.getindex(d::𝐷, idxs::UnitRange)
    fields = typeof(d) |> fieldnames;
    𝐷( [getfield(d, f)[idxs] for f in fields]... )
end

function Base.show(io::IO, d::𝐷)
    println(io, "Number of Samples: $( length(d) )")    
end


function trial2𝐷y(trials::Vector{Trial{T}}; batchdim::Int) where T

    a = vcat([trial.a for trial in trials]...);
    states = vcat([trial.s for trial in trials]...);
    timeafterswitch = vcat([trial.timeafterswitch for trial in trials]...) .|> Float32;
    
    n_samples = length(states);
    sam2rm = n_samples % batchdim;
    
    for _ in 1:sam2rm
        idx2rm = rand(2:length(a))
        deleteat!(a, idx2rm)
        deleteat!(states, idx2rm)
        deleteat!(timeafterswitch, idx2rm)
    end
    
    ln = length(states);
    nL, nR, nB, N, rL, rR, gL, gR, initial_visit, frst_visit, after_first_visit, μL, μR = [zeros(Float32, ln) for _ in 1:13];
    
    for (i, s) in enumerate(states)
        rL[i], rR[i] = Float32.(s.red);
        nL[i], nR[i], nB[i], N[i] = (Float32.(s.colored)..., sum(Float32, s.colored));
        gL[i], gR[i] = s.gaze==0 ? (0f0, 0f0) : s.gaze==1 ? (1f0, 0f0) : (0f0, 1f0);
        vL, vR = Float32.(s.visited);
        initial_visit[i] = Float32(vL==0 && vR==0);
        frst_visit[i] = Float32(vL+vR == 1);
        after_first_visit[i] = Float32(vL+vR > 1);
        μL[i] = mean( Beta(first(s.red)+1, first(s.colored[1:2] .- s.red) * first(s.visited)+1 ) );
        μR[i] = mean( Beta(last(s.red)+1, last(s.colored[1:2] .- s.red) * last(s.visited)+1 ) );
    
    end
    
    X = (;
        nL, nR, nB, rL, rR, gL, gR, initial_visit, frst_visit,
        after_first_visit, μL, μR, timeafterswitch
    );
    y = onehotbatch(a, 1:4);
    
    train_set = Flux.DataLoader((X, y), batchsize=batchdim);
    
    # input, target = first(train_set)
    out = NamedTuple[]
    for (input, target) in train_set
        
        push!(
            out, 
            (; 
                X = 𝐷([Vector{Float32}(getfield(input, f)) for f in fieldnames(typeof(input))]...), 
                y = target
            )
        )
    end
    
    return out

end;
