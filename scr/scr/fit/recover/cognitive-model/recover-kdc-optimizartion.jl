path2root = dirname(Base.active_project());
joinpath(path2root, "scr", "scr", "utils.jl") |> include;
joinpath(path2root, "scr", "scr", "model", "𝐷.jl") |> include;
joinpath(path2root, "scr", "scr", "model", "model.jl") |> include;
joinpath(path2root, "scr", "scr", "fit", "recover", "utils.jl") |> include;
using Functors, StatsFuns, JLD2, HybridModels, Optim, NNlib;
using CSV, DataFrames;
using Flux: logitcrossentropy, softmax;


println("\nLoading Data...")
path2data = joinpath(path2root, "data", "preprocessed", "binary");
ct1 = CSV.read( joinpath( path2data, "ct1.csv"), DataFrame );
trials = vcat([
    begin
        sbjdf = filter(r -> r.subject==subject, ct1);
        trials = vcat(
            [ # Trials
                sbjdf |> 
                filter(r -> r.trial==trial && r.event in ["switch", "stay", "select"] && r.visit>0) |> 
                df -> Trial(df) for trial in sbjdf.trial |> unique
            ]
        );
        trials
    end
    for subject in 1:15
]...);
d = vcat([create_dataframe(trial) for trial in trials]...);


println("Set Model and Parameters")
@hybridmodel function generative_model(X)
    # --- Transform and Extract Parameters --- #
    @kdc begin
        λ₀ = logit(0.99f0) 
        ω  = logit(0.60f0) 
        κ₁ = logit(0.25f0) 
        λ₂ = logit(0.01f0) 
        τ  = logit(0.08f0) 
    end
    @ddc β = 1.2f0

    
    # --- Transform parameters --- #
    λ₀, ω, κ₁, λ₂, τ = σ(λ₀), σ(ω), σ(κ₁), σ(λ₂), σ(τ);

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
(train_data, test_data) = splitobs(d, at=0.7);
data = (;
    train = (; X=train_data, y=onehotbatch(train_data.act, 1:4)),
    test  = (; X=test_data, y=onehotbatch(test_data.act, 1:4))
);


println("Randomly Initialize Parameters")
θinit = [
    -(rand(Float32) + 1),  # κ₁
    randn(Float32) |> abs, # ω
    -(rand(Float32) - 1),  # τ
    -(rand(Float32) + 4),  # λ₂
    randn(Float32) |> abs, # λ₀
    # randn(Float32) |> abs, # β
];


println("Optimize Parameters")
# Define loss function to take θ and return a scalar
function loss(θ)
    predictions = (m)(θ, data.train.X)
    return logitcrossentropy(predictions, data.train.y)
end;
opt_prob = Optim.optimize(loss, θinit, LBFGS()); # Create an optimization problem
θ_opt = Optim.minimizer(opt_prob); # Get the optimized parameters
# Print the results
println("Optimization results:")
θ_opt_dict = Dict(m.kdc.names[i] => σ(θ_opt[i]) for i in 1:length(m.kdc.names));
println("Minimized loss: ", Optim.minimum(opt_prob))
println("Optimized parameters: "); display(θ_opt_dict);

# Create Dict using vector of Symbols and vector of values

