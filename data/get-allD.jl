using CSV, DataFrames;
path2root = dirname(Base.active_project());
include( joinpath(path2root, "scr", "Modelling", "utils.jl") );


ct1 = CSV.read( joinpath( path2data, "preprocessed", "binary", "ct1.csv"), DataFrame );
allD = vcat([
    begin    
        sbjdf = filter(r -> r.subject==subject, ct1);
        trials = vcat(
            [ # Trials
                sbjdf |> 
                filter(r -> r.trial==trial && r.event in ["switch", "stay", "select"] && r.visit>0) |> 
                df -> Trial(df) for trial in sbjdf.trial |> unique
            ]
        );
        vcat(DataFrame.(trials)...)
    end for subject in unique(ct1.subject)
]...);

# Save the data
CSV.write( joinpath(path2data, "preprocessed", "binary", "allD.csv"), allD );
