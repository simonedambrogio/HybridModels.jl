path2root = dirname(Base.active_project());
joinpath(path2root, "scr", "scr", "utils.jl") |> include;
joinpath(path2root, "scr", "scr", "model", "𝐷.jl") |> include;
joinpath(path2root, "scr", "scr", "model", "model.jl") |> include;
joinpath(path2root, "scr", "scr", "fit", "recover", "utils.jl") |> include;
using Functors, StatsFuns, JLD2, HybridModels, Optim, NNlib, MLDataUtils;
using CSV, DataFrames;
using Flux: logitcrossentropy, softmax;

println("\nLoading Data...")
path2data = joinpath(path2root, "data", "preprocessed", "binary");
ct1 = CSV.read( joinpath( path2data, "ct1.csv"), DataFrame );
d = vcat([
    begin
        sbjdf = filter(r -> r.subject==subject, ct1);
        vcat(
            [ # Trials
                sbjdf |> 
                filter(r -> r.trial==trial && r.event in ["switch", "stay", "select"] && r.visit>0) |> 
                df -> Trial(df) |> 
                create_dataframe
                for trial in sbjdf.trial |> unique
            ]...
        );
    end
    for subject in 1:15
]...);

println("Set Model and Parameters")
@hybridmodel function generative_model(X)
    # --- Transform and Extract Parameters --- #
    @kdc begin
        λ₀ = logit(0.99f0) => σ
        ω  = logit(0.60f0) => σ
        κ₁ = logit(0.25f0) => σ
        λ₂ = logit(0.01f0) => σ
        τ  = logit(0.08f0) => σ
    end
    @ddc β = 1.2f0

    # --- Compute α and β to establish the shape of the beta posterior distribution --- #
    nL = n0L(X) .+ n1_0L(X, ω)      .+ n1_1L(X) .+ n2_0L(X, λ₂) .+ n2_1L(X)
    rL = r0L(X) .+ r1_0L(X, λ₀, nL) .+ r1_1L(X) .+ r2_0L(X, λ₂) .+ r2_1L(X)

    nR = n0R(X) .+ n1_0R(X, ω)      .+ n1_1R(X) .+ n2_0R(X, λ₂) .+ n2_1R(X)
    rR = r0R(X) .+ r1_0R(X, λ₀, nR) .+ r1_1R(X) .+ r2_0R(X, λ₂) .+ r2_1R(X)

    # --- Value of Select --- #    
    ρL = rL ./ nL
    ρR = rR ./ nR

    # --- Value of Information --- #    
    voiL = ucb([nL nR]', β)'
    voiR = ucb([nR nL]', β)'

    # --- Cost of Information --- #    
    coiL = (1f0 .- X.gL) .* κ₁
    coiR = (1f0 .- X.gR) .* κ₁

    return [(voiL .- coiL) (voiR .- coiR) (ρL .- ρR) (ρR .- ρL)]' ./ τ
end;


println("Simulate data")
pact = generative_model(d) |> softmax;
d.act = [rand( Distributions.Categorical(p[:]) ) for p in eachcol(pact)];
data = splitobs(d, at=0.7) |> 
df -> (; 
    train = (; X=df[1], y=onehotbatch(df[1].act, 1:4)), 
    test  = (; X=df[2], y=onehotbatch(df[2].act, 1:4))
);


println("Randomly Initialize Parameters")
θinit = [
    randn(Float32) |> abs, # λ₀
    randn(Float32) |> abs, # ω
    -(rand(Float32) + 1),  # κ₁
    -(rand(Float32) + 4),  # λ₂
    -(rand(Float32) - 1),  # τ
    randn(Float32) |> abs, # β
];

println("Optimize Parameters")
# Define loss function to take θ and return a scalar
m = deepcopy(generative_model);
HybridModels.predict!(m, θinit, data.train.X)
function loss!(θ)
    predictions = HybridModels.predict!(m, θ, data.train.X)
    return logitcrossentropy(predictions, data.train.y)
end;
opt_prob = Optim.optimize(loss!, θinit, LBFGS()); # Create an optimization problem
θ_opt = Optim.minimizer(opt_prob); # Get the optimized parameters
# Print the results
println("Optimization results:")
println("Minimized loss: ", Optim.minimum(opt_prob))
println("Optimized parameters: "); display(m);

# Create Dict using vector of Symbols and vector of values

