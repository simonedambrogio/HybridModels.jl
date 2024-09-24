using JLD2, OneHotArrays, DataFrames;
using Clustering, LinearAlgebra, RData
include( joinpath( dirname(Base.active_project()), "scr", "utils.jl" ) );
include( joinpath(path2root, "scr", "model", "UCB2.jl") );
path2data = joinpath(path2root, "scr", "analysis", "data");
using GLMakie; GLMakie.activate!(); 
println("\nLoading Data...")

dictionary_dataframe = RData.load( joinpath( path2data,  "data_list.rds") );

input2remove = "n-blocked__ρ-same__ρ-other";
folders = filter(s -> occursin(string("n-blocked__ρ-same__ρ-other", join(fill("+", 11))), s) && !occursin("._", s), readdir( joinpath(path2data, "fit", "1") ));


# Loss on Multiple Fits ---
info = [
    begin
        println(subject)
        vcat([ joinpath( joinpath(path2data, "fit", subject, folder, "training-info.jld2") ) |> load_object for folder in folders]...)
    end for subject in string.(1:20)
];

f=Figure(size=(900, 600), fontsize=25);
ax=GLMakie.Axis(f[1,1], xlabel="Subject", ylabel="Loss on Test Set");
[scatter!(ax, (rand(nrow(inf)) .- 0.5) .* 0.3 .+ i, inf.test_loss) for (i, inf) in enumerate(info)];
display(f)

begin # Save Figure
    CairoMakie.activate!()
    save( joinpath(@__DIR__, "figure.pdf"), f, pdf_version="1.4")
    GLMakie.activate!()
end


n = [
    dictionary_dataframe["ct1"] |> 
    filter(r -> r.subject==sbj && r.event in ["start switch", "switch", "stay", "select"]) |> 
    nrow for sbj in string.(1:20)
];

scatter(n, [mean(inf.test_loss) for inf in info])
scatter(n, [std(inf.test_loss) for inf in info])
scatter(1:20, n, markersize=[mean(inf.test_loss) for inf in info] .* 70, color=[mean(inf.test_loss) for inf in info])
scatter(1:20, n, markersize=[std(inf.test_loss) for inf in info] .* 1_700, color=[mean(inf.test_loss) for inf in info])


var2remove = string.(split(input2remove, "__"));
vois = [
    begin
        println(subject)
        # Load data ---
        df = vcat(
            joinpath(path2data, "fit", subject, folders[1], "train_data.jld2") |> load_object |> 
            data -> [DataFrame(d.X) for d in data] |> vdf -> vcat(vdf...),
            joinpath(path2data, "fit", subject, folders[1], "test_data.jld2") |> load_object |> 
            data -> [DataFrame(d.X) for d in data] |> vdf -> vcat(vdf...),
        );
        hcat([
            begin
                # Load model ---
                model = joinpath(path2data, "fit", subject, folder, "out.jld2") |> load_object;        
                extract(model, df, var2remove).bonus
            end for folder in folders
        ]...);        
    end for subject in string.(1:20)
];

inputs = [
    begin
        println(subject)
        # Load data ---
        df = vcat(
            joinpath(path2data, "fit", subject, folders[1], "train_data.jld2") |> load_object |> 
            data -> [DataFrame(d.X) for d in data] |> vdf -> vcat(vdf...),
            joinpath(path2data, "fit", subject, folders[1], "test_data.jld2") |> load_object |> 
            data -> [DataFrame(d.X) for d in data] |> vdf -> vcat(vdf...),
        );
        model = joinpath(path2data, "fit", subject, folders[1], "out.jld2") |> load_object;
        extract(model, df, var2remove)
                
    end for subject in string.(1:20)
];

best_models = [argmax(i.test_loss) for i in info];

nns = [
    joinpath(path2data, "fit", subject, folders[best_models[i]], "out.jld2") |> load_object
    for (i, subject) in enumerate(string.(1:20))
];


outnn = Dict{String, Matrix{Float32}}();

# First Visit ------------------
f=Figure(size=(1_700, 1_000));
sbj = 0
for r in 1:4, c in 1:5
    ax=Axis3(f[r,c], xlabel="nA", ylabel="nB", zlabel="voi");
    sbj += 1
    inputs[sbj] |> DataFrame |> 
    df -> insertcols(df, :voi => vois[sbj][:, best_models[sbj]]) |> 
    df -> filter(r -> r.fv==1, df) |>
    df -> scatter!(ax, df.nA, df.nB, df.voi, color=df.gA)
end
display(f)

# Attended
df = vcat([inputs[sbj] |> DataFrame |> df -> filter(r -> r.fv==1 && r.gA==1, df) for sbj in 1:20]...) |> 
df -> select(df, [:nA, :nB]) |> 
df -> DataFrames.transform(df, :nA => (x -> Int.(round.(x))) => :nA, :nB => (x -> Int.(round.(x))) => :nB) |> 
unique;
outnn["fv-gA"] = hcat([nns[sbj].nn( [df.nA./100 df.nB./100 fill(1f0, nrow(df)) fill(1f0, nrow(df))]' )[:] for sbj in 1:20]...);

# f=Figure();
# ax=Axis3(f[1,1])
# scatter!(ax, df.nA, df.nB, outnn["fv-gA"][:,1])
# scatter!(ax, df.nA, df.nB, outnn["fv-gA"][:,19])
# display(f)
# fig, ax, hm = heatmap(cor(outnn["fv-gA"]))
# Colorbar(fig[:, end+1], hm)
# density( cor(outnn["fv-gA"]) |> c -> tril(c, -1) |> vec |> filter(x -> x!=0.)  )

f=Figure();ax=GLMakie.Axis(f[1,1], limits =(nothing, (-1, 1)), xticks=(1:20));
scatter!(ax,
    mean.([cor(outnn["fv-gA"])[sbj, :] |> filter(x -> x!=1.) for sbj in 1:20])
)
display(f)


# Unattended
df = vcat([inputs[sbj] |> DataFrame |> df -> filter(r -> r.fv==1 && r.gA==0, df) for sbj in 1:20]...) |> 
df -> select(df, [:nA, :nB]) |> 
df -> DataFrames.transform(df, :nA => (x -> Int.(round.(x))) => :nA, :nB => (x -> Int.(round.(x))) => :nB) |> 
unique;
outnn["fv-gB"] = hcat([nns[sbj].nn( [df.nA./100 df.nB./100 fill(0f0, nrow(df)) fill(1f0, nrow(df))]' )[:] for sbj in 1:20]...);

# f=Figure();
# ax=Axis3(f[1,1])
# scatter!(ax, df.nA, df.nB, outnn["fv-gB"][:,1])
# scatter!(ax, df.nA, df.nB, outnn["fv-gB"][:,5])
# display(f)
# fig, ax, hm = heatmap(cor(outnn["fv-gB"]))
# Colorbar(fig[:, end+1], hm)
# density( cor(outnn["fv-gB"]) |> c -> tril(c, -1) |> vec |> filter(x -> x!=0.)  )
# hist( cor(outnn["fv-gB"]) |> c -> tril(c, -1) |> vec |> filter(x -> x!=0.)  )

f=Figure();ax=GLMakie.Axis(f[1,1], limits =(nothing, (-1, 1)), xticks=(1:20));
scatter!(ax,
    mean.([cor(outnn["fv-gB"])[sbj, :] |> filter(x -> x!=1.) for sbj in 1:20])
)
display(f)


# After First Visit ------------------
f=Figure(size=(1_700, 1_000));
sbj = 0
for r in 1:4, c in 1:5
    ax=Axis3(f[r,c], xlabel="nA", ylabel="nB", zlabel="voi");
    sbj += 1
    inputs[sbj] |> DataFrame |> 
    df -> insertcols(df, :voi => vois[sbj][:, best_models[sbj]]) |> 
    df -> filter(r -> r.fv==0, df) |>
    df -> scatter!(ax, df.nA, df.nB, df.voi, color=df.gA)
end
display(f)

# Attended
df = vcat([inputs[sbj] |> DataFrame |> df -> filter(r -> r.fv==0 && r.gA==1, df) for sbj in 1:20]...) |> 
df -> select(df, [:nA, :nB]) |> 
df -> DataFrames.transform(df, :nA => (x -> Int.(round.(x))) => :nA, :nB => (x -> Int.(round.(x))) => :nB) |> 
unique;
outnn["afv-gA"] = hcat([nns[sbj].nn( [df.nA./100 df.nB./100 fill(1f0, nrow(df)) fill(0f0, nrow(df))]' )[:] for sbj in 1:20]...);

# f=Figure();
# ax=Axis3(f[1,1], xlabel="N. Attended", ylabel="N. Unattended", zlabel="VOI")
# scatter!(ax, df.nA, df.nB, outnn["afv-gA"][:,1])
# scatter!(ax, df.nA, df.nB, outnn["afv-gA"][:,12])
# display(f)
# f=Figure();
# ax=Axis3(f[1,1], xlabel="N. Attended", ylabel="N. Unattended", zlabel="VOI")
# [scatter!(ax, df.nA, df.nB, outnn["afv-gA"][:,i]) for i in 1:20]
# display(f)
# fig, ax, hm = heatmap(cor(outnn["afv-gA"]))
# Colorbar(fig[:, end+1], hm)
# density( cor(outnn["afv-gA"]) |> c -> tril(c, -1) |> vec |> filter(x -> x!=0.)  )

f=Figure();ax=GLMakie.Axis(f[1,1], limits =(nothing, (-1, 1)), xticks=(1:20));
scatter!(ax,
    mean.([cor(outnn["afv-gA"])[sbj, :] |> filter(x -> x!=1.) for sbj in 1:20])
)
display(f)

# Unattended
df = vcat([inputs[sbj] |> DataFrame |> df -> filter(r -> r.fv==0 && r.gA==0, df) for sbj in 1:20]...) |> 
df -> select(df, [:nA, :nB]) |> 
df -> DataFrames.transform(df, :nA => (x -> Int.(round.(x))) => :nA, :nB => (x -> Int.(round.(x))) => :nB) |> 
unique;
outnn["afv-gB"] = hcat([nns[sbj].nn( [df.nA./100 df.nB./100 fill(0f0, nrow(df)) fill(1f0, nrow(df))]' )[:] for sbj in 1:20]...);

# f=Figure();
# ax=Axis3(f[1,1])
# scatter!(ax, df.nA, df.nB, outnn["afv-gB"][:,1])
# scatter!(ax, df.nA, df.nB, outnn["afv-gB"][:,19])
# display(f)
# fig, ax, hm = heatmap(cor(outnn["afv-gB"]))
# Colorbar(fig[:, end+1], hm)
# density( cor(outnn["afv-gB"]) |> c -> tril(c, -1) |> vec |> filter(x -> x!=0.)  )
# hist( cor(outnn["afv-gB"]) |> c -> tril(c, -1) |> vec |> filter(x -> x!=0.)  )

f=Figure();ax=GLMakie.Axis(f[1,1], limits =(nothing, (-1, 1)), xticks=(1:20));
scatter!(ax,
    mean.([cor(outnn["afv-gB"])[sbj, :] |> filter(x -> x!=1.) for sbj in 1:20])
)
display(f)





# Average ---------------------------

# First Visit ------------------

# Attended
df1 = vcat([inputs[sbj] |> DataFrame |> df -> filter(r -> r.gA==1 && r.fv==1, df) for sbj in 1:20]...) |> 
df -> select(df, [:nA, :nB]) |> 
df -> DataFrames.transform(df, :nA => (x -> Int.(round.(x))) => :nA, :nB => (x -> Int.(round.(x))) => :nB) |> 
unique;
df1.voi = df1 |> df -> mean([nns[sbj].nn( [df.nA./100 df.nB./100 fill(1f0, nrow(df)) fill(1f0, nrow(df))]' )[:] for sbj in 1:20]);
df1.gA = fill(1, nrow(df1))

# Unattended
df0 = vcat([inputs[sbj] |> DataFrame |> df -> filter(r -> r.gA==0 && r.fv==1, df) for sbj in 1:20]...) |> 
df -> select(df, [:nA, :nB]) |> 
df -> DataFrames.transform(df, :nA => (x -> Int.(round.(x))) => :nA, :nB => (x -> Int.(round.(x))) => :nB) |> 
unique;
df0.voi = df0 |> df -> mean([nns[sbj].nn( [df.nA./100 df.nB./100 fill(0f0, nrow(df)) fill(1f0, nrow(df))]' )[:] for sbj in 1:20]);
df0.gA = fill(0, nrow(df0))


df = vcat(df1, df0);

f=Figure(size=(800, 600));
ax=Axis3(f[1,1], xlabel="nA", ylabel="nB", zlabel="voi");
scatter!(ax, df.nA, df.nB, df.voi, color=df.gA)
display(f)


# After First Visit ------------------

# Attended
df1 = vcat([inputs[sbj] |> DataFrame |> df -> filter(r -> r.gA==1 && r.fv==0, df) for sbj in 1:20]...) |> 
df -> select(df, [:nA, :nB]) |> 
df -> DataFrames.transform(df, :nA => (x -> Int.(round.(x))) => :nA, :nB => (x -> Int.(round.(x))) => :nB) |> 
unique;
df1.voi = df1 |> df -> mean([nns[sbj].nn( [df.nA./100 df.nB./100 fill(1f0, nrow(df)) fill(0f0, nrow(df))]' )[:] for sbj in 1:20]);
df1.gA = fill(1, nrow(df1))


# Unattended
df0 = vcat([inputs[sbj] |> DataFrame |> df -> filter(r -> r.gA==0 && r.fv==0, df) for sbj in 1:20]...) |> 
df -> select(df, [:nA, :nB]) |> 
df -> DataFrames.transform(df, :nA => (x -> Int.(round.(x))) => :nA, :nB => (x -> Int.(round.(x))) => :nB) |> 
unique;
df0.voi = df0 |> df -> mean([nns[sbj].nn( [df.nA./100 df.nB./100 fill(0f0, nrow(df)) fill(0f0, nrow(df))]' )[:] for sbj in 1:20]);
df0.gA = fill(0, nrow(df0))

df = vcat(df1, df0);

# First Visit ------------------
f=Figure(size=(800, 600));
ax=Axis3(f[1,1], xlabel="nA", ylabel="nB", zlabel="voi");
scatter!(ax, df.nA, df.nB, df.voi, color=df.gA)
display(f)




f=Figure(size=(800, 600));
ax=Axis3(f[1,1], xlabel="nA", ylabel="nB", zlabel="voi");
srXy["First-Visit"]["Attended"] |> df -> scatter!(ax, df.nA, df.nB, df.voi)
srXy["First-Visit"]["Unattended"] |> df -> scatter!(ax, df.nA, df.nB, df.voi)
display(f)

f=Figure(size=(800, 600));
ax=Axis3(f[1,1], xlabel="nA", ylabel="nB", zlabel="voi");
srXy["After-First-Visit"]["Attended"] |> df -> scatter!(ax, df.nA, df.nB, df.voi)
srXy["After-First-Visit"]["Unattended"] |> df -> scatter!(ax, df.nA, df.nB, df.voi)
display(f)

save_object( joinpath(path2root, "data", "data_sr.jld2"), srXy )





df4 = dictionary_dataframe["ct1"] |> 
filter(r -> r.subject .== "4" && r.session .== "1")
# Time
df4 |> df -> ((df.time[end] - df.time[1]) / 1000 / 60)
# N. Visits
df4 |> df -> mean(df.n_visits)
# N. Samples
df4 |> df -> mean(df.n_samples)
# N. Selections
df4 |> df -> sum(df.event .== "select")

df7 = dictionary_dataframe["ct1"] |> 
filter(r -> r.subject .== "7" && r.session .== "1")
# Time
df7 |> df -> ((df.time[end] - df.time[1]) / 1000 / 60)
# N. Visits
df7 |> df -> mean(df.n_visits)
# N. Samples
df7 |> df -> mean(df.n_samples)
# N. Selections
df7 |> df -> sum(df.event .== "select")

dfpl = [
    begin
        df = dictionary_dataframe["ct1"] |> 
        filter(r -> r.subject .== subject && r.session .== session)
        mean(df.n_samples)
        # sum(df.event .== "select")
    end for subject in string.(1:20), session in string.(1:4)
]


scatter(mean(dfpl; dims=2)[:], markersize=15)
scatter(1:20, n,  markersize=mean(dfpl; dims=2)[:])
scatter(n, mean(dfpl; dims=2)[:])


# Time
dfpl = [
    begin
        df = dictionary_dataframe["ct1"] |> 
        filter(r -> r.subject .== subject && r.session .== session)
        ((df.time[end] - df.time[1]) / 1000 / 60)
    end for subject in string.(1:20), session in string.(1:4)
]

f=Figure(fontsize=25)
ax=GLMakie.Axis(f[1,1])
[scatter!(ax, fill(i,4), dfpl[i,:], markersize=15) for i in 1:20]


