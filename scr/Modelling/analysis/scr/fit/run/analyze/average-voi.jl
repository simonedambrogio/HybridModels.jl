path2root = dirname(Base.active_project());
include( joinpath(path2root, "scr", "Modelling", "utils.jl") );
using GLMakie, JSON, RData, DataFrames, CSV, JLD2, ProgressBars;
function plotvois(ex::DataFrame)
    fig = Figure(size=(500*2, 500))
    
    ax = Axis3(fig[1,1], xlabel="N. Current", ylabel="N. Other", zlabel="VOI");
    ex |> filter(r -> r.fv && r.gA) |> x -> scatter!(ax, x.nA, x.nB, x.voi)
    ex |> filter(r -> r.fv && .!r.gA) |> x -> scatter!(ax, x.nA, x.nB, x.voi)

    ax = Axis3(fig[1,2], xlabel="N. Current", ylabel="N. Other", zlabel="VOI");
    ex |> filter(r -> .!r.fv && r.gA) |> x -> scatter!(ax, x.nA, x.nB, x.voi, label="Attended")
    ex |> filter(r -> .!r.fv && .!r.gA) |> x -> scatter!(ax, x.nA, x.nB, x.voi, label="Unattended")
    
    axislegend(ax)

    display(fig)

    return fig, ax
end

GLMakie.activate!()
path2output = joinpath(path2root, "scr", "Modelling", "analysis", "output");


println("\nGet DataFrame with [subject, nA, nB, gA, fv, voi] for each subject...")
dfs = DataFrame[];
for s in ProgressBar( 1:25 )
    model = load_object( joinpath(path2output, string(s), "hybridmodel", "model.jld2") );
    idxinput = JSON.parsefile(joinpath(path2output, string(s), "inputs.json"))["idxinput"] .|> Bool;
    df = CSV.read( joinpath(path2data, "preprocessed", "binary", "allD.csv"), DataFrame ) |> 
    x -> select(x, Not(:act)) |> unique |> 
    x -> mapcols(col -> Float32.(col), x);
    ex = extract(model, df, idxinput) |> DataFrame |> 
    x -> transform(x, :voi => (x -> fill(s, length(x))) => :subject) |> 
    x -> select(x, [:subject, :nA, :nB, :gA, :fv, :voi]);

    push!(dfs, ex)
end
df = vcat(dfs...);
plotvois(df |> filter(r -> r.subject==1))


println("\nCreate a DataFrame with the average value of voi...")
begin
    println("Weight the influence of voi based on the sample size")
    
    w = CSV.read( joinpath(path2data, "preprocessed", "binary", "ct1.csv"), DataFrame ) |> 
    x -> groupby(x, :subject) |> x -> [size(x, 1) for x in x] |> 
    x -> x ./ sum(x);

    nsubjects = size(w, 1);


    println("Atended - First Visit")
    numsamples = df |> filter(r -> r.subject == 1 && r.gA && r.fv) |> x -> size(x, 1)
    vois = zeros(Float32, nsubjects, numsamples);
    Threads.@threads for s in ProgressBar( 1:nsubjects )
        vois[s,:] = df |> filter(r -> r.subject == s && r.gA && r.fv) |> 
            x -> select(x, :voi) |> Matrix |> vec |> permutedims
    end
    !all([sum(r)!=0 for r in eachrow(vois)]) && error("Number of rows is not 20")

    dfmean = df |> filter(r -> r.subject==1 && r.gA && r.fv) |> 
    x -> select(x, [:nA, :nB, :gA, :fv]) |> 
    x -> DataFrames.transform(x, :nA => (x -> sum(vois .* w; dims=1)[:]) => :voi) |> 
    unique
    
    
    println("Unatended - First Visit")
    numsamples = df |> filter(r -> r.subject == 1 && .!r.gA && r.fv) |> x -> size(x, 1)
    vois = zeros(Float32, nsubjects, numsamples);
    Threads.@threads for s in ProgressBar( 1:nsubjects )
        vois[s,:] = df |> filter(r -> r.subject == s && .!r.gA && r.fv) |> 
            x -> select(x, :voi) |> Matrix |> vec |> permutedims
    end
    !all([sum(r)!=0 for r in eachrow(vois)]) && error("Number of rows is not 20")
    
    dfmean = vcat( 
        dfmean,
        df |> filter(r -> r.subject==1 && .!r.gA && r.fv) |> 
        x -> select(x, [:nA, :nB, :gA, :fv]) |> 
        x -> DataFrames.transform(x, :nA => (x -> sum(vois .* w; dims=1)[:]) => :voi) |> unique
    );
    
    println("Atended - After First Visit")
    numsamples = df |> filter(r -> r.subject == 1 && r.gA && .!r.fv) |> x -> size(x, 1)
    vois = zeros(Float32, nsubjects, numsamples);
    Threads.@threads for s in ProgressBar( 1:nsubjects )
        vois[s,:] = df |> filter(r -> r.subject == s && r.gA && .!r.fv) |> 
            x -> select(x, :voi) |> Matrix |> vec |> permutedims
    end
    !all([sum(r)!=0 for r in eachrow(vois)]) && error("Number of rows is not 20")

    dfmean = vcat( 
        dfmean,
        df |> filter(r -> r.subject==1 && r.gA && .!r.fv) |> 
        x -> select(x, [:nA, :nB, :gA, :fv]) |> 
        x -> DataFrames.transform(x, :nA => (x -> sum(vois .* w; dims=1)[:]) => :voi) |> unique
    )
    
    println("Unatended - After First Visit")
    numsamples = df |> filter(r -> r.subject == 1 && .!r.gA && .!r.fv) |> x -> size(x, 1)
    vois = zeros(Float32, nsubjects, numsamples);
    Threads.@threads for s in ProgressBar( 1:nsubjects )
        vois[s,:] = df |> filter(r -> r.subject == s && .!r.gA && .!r.fv) |> 
            x -> select(x, :voi) |> Matrix |> vec |> permutedims
    end
    !all([sum(r)!=0 for r in eachrow(vois)]) && error("Number of rows is not 20")

    dfmean = vcat( 
        dfmean,
        df |> filter(r -> r.subject==1 && .!r.gA && .!r.fv) |> 
        x -> select(x, [:nA, :nB, :gA, :fv]) |> 
        x -> DataFrames.transform(x, :nA => (x -> sum(vois .* w; dims=1)[:]) => :voi) |> unique
    );
end;
fig, _ = plotvois(dfmean);


save(joinpath(@__DIR__, "average-voi.png"), fig, px_per_unit=2)
