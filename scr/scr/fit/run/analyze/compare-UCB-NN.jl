using JSON, GLMakie, DataFrames;
path2root = dirname(Base.active_project());
path2output = joinpath(path2root, "scr", "Modelling", "analysis", "output");

# read the json file
function readinfo(subject::Int, modelfolder::String, var::String=nothing)
    path2subject = joinpath(path2output, string(subject));
    path2json = joinpath(path2subject, modelfolder, "info.json");
    if isnothing(var)
        JSON.parsefile(path2json);
    else
        JSON.parsefile(path2json)[var];
    end
end;

subject = 1;


df = DataFrame((; subject=1:25)) |> 
df -> transform(df, 
    :subject => ByRow(x -> readinfo(x, "cognitivemodel", "test_loss")) => :cognitivemodel,
    :subject => ByRow(x -> readinfo(x, "hybridmodel", "test_loss")) => :hybridmodel
) 


fig = Figure();
ax = Axis(fig[1,1], title="Test Loss", xlabel="Subject", ylabel="ΔLoss (cognitive model - hybrid model)", limits=((nothing, nothing), (-0.02, nothing)));
δl = df.cognitivemodel .- df.hybridmodel;
scatter!(ax, 1:25, δl);
[lines!(ax, [i, i], [0, δl[i]], color=Makie.wong_colors()[1]) for i in 1:25]
fig
save(joinpath(@__DIR__, "compare-UCB-NN.png"), fig, px_per_unit=2)