function create_dataframe(trial::Trial)
    
    extract_red(s::State) = Float32.(s.red);
    extract_colored(s::State) = Float32.(s.colored);
    extract_gaze(s::State) = s.gaze==0 ? [0f0, 0f0] : s.gaze==1 ? [1f0, 0f0] : [0f0, 1f0];
    extract_visit(s::State) = Float32.(s.visited);

    s = copy(trial.s)
    act = copy(trial.a);
    t = copy(trial.timeafterswitch) .|> Float32
    red = hcat(extract_red.(s)...);
    rL, rR = [r[:] for r in eachrow(red)];
    
    colored = hcat(extract_colored.(s)...);
    nL, nR, nB = [n[:] for n in eachrow(colored)];
    gaze = hcat(extract_gaze.(s)...);
    gL, gR = [g[:] for g in eachrow(gaze)];
    visit = hcat(extract_visit.(s)...);
    vL, vR = [v[:] for v in eachrow(visit)];
    initial_visit = Float32.(vL .== 0 .&& vR .== 0) ;
    first_visit = Float32.(vL+vR .== 1);
    after_first_visit = Float32.(vL+vR .> 1);
    
    μL = mean.( Beta.(rL .+ 1, (nL - rL) .* vL .+ 1 ) );
    μR = mean.( Beta.(rR .+ 1, (nR - rR) .* vR .+ 1 ) );

    return DataFrame((;
        nL, nR, nB, rL, rR, gL, gR, initial_visit, first_visit, after_first_visit, μL, μR, t, act
    ) )
    
end;

function getXy(d::AbstractDataFrame; batchdim = nothing)

    batchdim = isnothing(batchdim) ? nrow(d) : batchdim;
        
    row2rm = nrow(d) % batchdim
    newrow = nrow(d) - row2rm
    ngroup = newrow / batchdim |> Int
    d = shuffle(d)[1:newrow, :]
    d.batchid = vcat([fill(i, batchdim) for i in 1:ngroup]...)

    [
        begin
            (; 
                X = 𝐷( DataFrame(d), batchdim),
                y = onehotbatch( d.act, 1:4 )
            )
        end for d in groupby(d, :batchid)
    ]
end;
