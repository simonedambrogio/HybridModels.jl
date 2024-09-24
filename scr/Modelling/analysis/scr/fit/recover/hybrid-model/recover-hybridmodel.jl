# using Pkg; Pkg.update(); Pkg.precompile()
println("Load Libraries and Model...")
path2root = dirname(Base.active_project());
include( joinpath(path2root, "scr", "Modelling", "utils.jl") );
using LinearAlgebra, OneHotArrays, JLD2, DataStructures, MLDataUtils, 
Optim, JSON, CSV, DataFrames, ProgressBars;
using GLMakie, Random; Random.seed!(123);
using StatsFuns, FluxOptTools, TensorBoardLogger, Logging
using NNlib: softmax, tanh
using HybridModels
path2modellingscr = joinpath(path2root, "scr", "Modelling", "analysis", "scr");
include( joinpath(path2modellingscr, "KDC.jl") );
include( joinpath(path2modellingscr, "VoiUCB.jl") );
include( joinpath(path2modellingscr, "VoiNN.jl") );
include( joinpath(path2modellingscr, "HybridModel.jl") );
include( joinpath(path2modellingscr, "fit", "recover", "utils.jl") );


subjects, input2remove = 1:10, "n-blocked__ρ-same__ρ-other";
finalPath = @__DIR__; path2ht=nothing; tune_hyperparameters=false;
var2remove = string.(split(input2remove, "__"));
idxinput = getidx(var2remove);

println("\nLoading Data...")
ct1 = CSV.read( joinpath( path2data, "preprocessed", "binary", "ct1.csv"), DataFrame );
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
    for subject in subjects
]...);
d = vcat([create_dataframe(trial) for trial in trials]...);


println("Set Parameters")
λ₀ = 0.99f0 |> logit;
ω  = 0.60f0 |> logit;
κ₁ = 0.25f0 |> logit;
λ₂ = 0.01f0 |> logit;
τ  = 0.08f0 |> logit;
β  = 1.20f0;
generative_model = Agent( 
    KDC(λ₀, ω, κ₁, λ₂, τ),
    VoiUCB(β)
);


println("Simulate data")
pact = generative_model(d) |> softmax;
d.act = [rand( Distributions.Categorical(p[:]) ) for p in eachcol(pact)];
data = getXy(d; batchdim=60);
shuffle!(data);
(train_data, test_data) = splitobs(data, at=0.7);
data = (; train_data, test_data);

println("Randomly Initialize Parameters")
rng = Xoshiro();
input_dim, ny, nh, γ = 2, 1, fill(32,4), 5;
θ = VoiNNParams(input_dim, nh, ny, γ; nl=Flux.tanh, learn_γ=true, rng);
hybrid_model = Agent(
    KDC(λ₀, ω, κ₁, λ₂, τ),
    VoiNN(θ)
);

println("Optimize Parameters")
loss(model, X, y) = Flux.logitcrossentropy(model(X), y);

maxiter, miniter = 120, 80;
hpvals = (; η_cm=0.001, η_nn=0.01);
pars = Flux.params(hybrid_model.ddc);
opt = Flux.Adam(Float32(hpvals[:η_nn]));
filename = joinpath(finalPath, "fit-hybridmodel.json");

trainAdam!(
    hybrid_model, 
    data, 
    loss, 
    [maxiter, miniter], 
    pars, opt, 
    filename
);



edf_hybrid = extract(hybrid_model, d) |> DataFrame
fig = Figure()
ax = Axis(fig[1, 1], xlabel="nA", ylabel="voi")
scatter!(ax, edf_hybrid.nA, edf_hybrid.voi)
display(fig)

# Training Function ----
function train!(model, test_data, train_data, loss, maxiter, miniter, parsnn, parscm, optnn, optcm, finalPath=nothing) 

    println("Start Training...")

    # Set up TensorBoard logger
    if !isnothing(finalPath)
        tb_logger = TBLogger(joinpath(finalPath, "tensorboard_logs"), tb_overwrite)
        println("run the following command to open the tensorboard:")
        println("\ntensorboard --logdir $(finalPath)\n")
    end

    loss_array = CircularBuffer{Float32}(5);
    isworse = 0f0;
    for epoch in ProgressBar(1:maxiter)
        
        validation_loss = mean(loss(model,x,y) for (x,y) in test_data);
        push!(loss_array, validation_loss)
        isworse = (isworse + 1) * Int(findmin(loss_array)[2]!=5)

        if epoch>miniter && isworse>=10
            println("Epochs: $(epoch)    Loss: $(validation_loss)")

            if !isnothing(finalPath)
                training_info = Dict(
                    "maxiter" => maxiter, "niter" => epoch, "test_loss" => validation_loss
                )
                filename = joinpath(finalPath, "training-info.json")
                open(filename, "w") do file
                    write(file, JSON.json(training_info))
                end
            end
            
            return nothing
        end
        
        for (input, label) in train_data
            # Calculate Gradient and Update Neural Network
            grads = gradient(() -> loss(model, input, label), parsnn);
            Flux.update!(optnn, parsnn, grads);
            # Calculate Gradient and Update Cognitive Model
            grads = gradient(() -> loss(model, input, label), parscm);
            Flux.update!(optcm, parscm, grads);
        end

        # Log to TensorBoard
        if !isnothing(finalPath)
            with_logger(tb_logger) do
                @info "training" validation_loss=validation_loss

                if epoch % 10 == 0
                    edf_hybrid = HybridModels.extract(model, d) |> DataFrame
                    fig = Figure()
                    ax = Axis(fig[1, 1], xlabel="nA", ylabel="voi", title="nA vs voi (Epoch $epoch)")
                    scatter!(ax, edf_hybrid.nA, edf_hybrid.voi)
                    
                    # Save the figure as PNG
                    png_path = joinpath(finalPath, "nA_vs_voi.png")
                    save(png_path, fig)
                end
            end
        end
        
    end 
    
    validation_loss = mean(loss(model,x,y) for (x,y) in test_data);

    if !isnothing(finalPath)
        training_info = Dict(
            "maxiter" => maxiter, "niter" => maxiter, "test_loss" => validation_loss
        )
        filename = joinpath(finalPath, "training-info.json")
        open(filename, "w") do file
            write(file, JSON.json(training_info))
        end
    end
    
    println("Epochs: $(maxiter)    Loss: $(validation_loss)")
end;


