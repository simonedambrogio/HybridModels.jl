include("../fit.jl");

for subject in 1:25
    input2remove = "n-blocked__ρ-same__ρ-other";
    finalPath = joinpath(path2root, "scr/Modelling/analysis/output", string(subject));
    create_unique_path(finalPath);
    fit_model(subject, input2remove, finalPath);
end
