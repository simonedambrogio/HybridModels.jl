path2root = dirname(Base.active_project());
joinpath(path2root, "scr", "scr", "utils.jl") |> include;
joinpath(path2root, "scr", "scr", "model", "𝐷.jl") |> include;
joinpath(path2root, "scr", "scr", "model", "model.jl") |> include;
joinpath(path2root, "scr", "scr", "fit", "recover", "utils.jl") |> include;
using Functors, StatsFuns, JLD2, HybridModels, Optim, NNlib, MLDataUtils;
using CSV, DataFrames;
using Flux: logitcrossentropy, softmax;
using TensorBoardLogger, Logging, RobustNeuralNetworks;

println("\nLoading Data...");
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
rng = Xoshiro();
input_dim, ny, nh, γ = 2, 1, fill(32,4), 5;
nnpars = DenseLBDNParams{Float32}(input_dim, nh, ny, γ; nl=Flux.tanh, learn_γ=true, rng);

@hybridmodel function hybrid_model(X)

    # --- Transform and Extract Parameters --- #
    @kdc begin
        λ₀ = logit(0.99f0) => σ
        ω  = logit(0.60f0) => σ
        κ₁ = logit(0.25f0) => σ
        λ₂ = logit(0.01f0) => σ
        τ  = logit(0.08f0) => σ
    end
    @ddc θ = DenseLBDNParams{Float32}(input_dim, nh, ny, γ; nl=Flux.tanh, learn_γ=true, rng);

    # --- Compute α and β to establish the shape of the beta posterior distribution --- #
    nL = n0L(X) .+ n1_0L(X, ω)      .+ n1_1L(X) .+ n2_0L(X, λ₂) .+ n2_1L(X)
    rL = r0L(X) .+ r1_0L(X, λ₀, nL) .+ r1_1L(X) .+ r2_0L(X, λ₂) .+ r2_1L(X)
    nR = n0R(X) .+ n1_0R(X, ω)      .+ n1_1R(X) .+ n2_0R(X, λ₂) .+ n2_1R(X)
    rR = r0R(X) .+ r1_0R(X, λ₀, nR) .+ r1_1R(X) .+ r2_0R(X, λ₂) .+ r2_1R(X)

    # --- Value of Select --- #    
    ρL = rL ./ nL
    ρR = rR ./ nR

    # --- Value of Information --- #    
    nn = LBDN(θ);
    voiL = nn([nL nR]')'
    voiR = nn([nR nL]')'

    # --- Cost of Information --- #    
    coiL = (1f0 .- X.gL) .* κ₁
    coiR = (1f0 .- X.gR) .* κ₁

    return [(voiL .- coiL) (voiR .- coiR) (ρL .- ρR) (ρR .- ρL)]' ./ τ
end;



println("Optimize Parameters")
using DataStructures, ProgressBars;
using Flux: DataLoader

dataminibatch = (;
    train = DataLoader(data.train, batchsize=64, shuffle=true),
    test  = DataLoader(data.test, batchsize=64, shuffle=true)
);


# X, y = dataminibatch.train |> first;
loss(model, X, y) = Flux.logitcrossentropy(model(X), y);

iter = 120;
η_cm=0.001;
par = Flux.params(hybrid_model.ddc);
opt = Flux.Adam(Float32(η_cm));
filename = joinpath(@__DIR__, "fit-hybridmodel.json");


for epoch in 1:100
  Flux.train!(hybrid_model, data, opt_state) do m, x, y
    Flux.logitcrossentropy(m(x), y)
  end
end

function train!(model::HybridModel, data, pars, opt) 

    for (input, label) in data
        # Calculate Gradient and Update Neural Network
        grads = gradient(() -> loss(model, input, label), pars);
        Flux.update!(opt, pars, grads);
    end

end

train!(hybrid_model, dataminibatch.train, par, opt)

println("Start Training...")
train_data, test_data = dataminibatch;
folder = isnothing(filename) ? nothing : dirname(filename);

# Set up TensorBoard logger
if !isnothing(folder)
    tb_logger = TBLogger(joinpath(folder, "tensorboard_logs"))
    println("run the following command to open the tensorboard:")
    println("\ntensorboard --logdir $folder\n")
end

iter = ProgressBar(1:maxiter);
for epoch in iter
    
    validation_loss = mean(loss(model,x,y) for (x,y) in test_data);
    training_loss = mean(loss(model,x,y) for (x,y) in train_data);
    
    set_postfix(iter, 
        TrainLoss = round(training_loss; digits=4), 
        ValLoss   = round(validation_loss; digits=4)
    )
    
    for (input, label) in train_data
        # Calculate Gradient and Update Neural Network
        grads = gradient(() -> loss(model, input, label), par);
        Flux.update!(opt, par, grads);
    end

    # Log to TensorBoard
    if !isnothing(folder)
        with_logger(tb_logger) do
            @info "training"   training_loss=training_loss
            @info "validation" validation_loss=validation_loss
        end
    end
    
end 

validation_loss = mean(loss(model,x,y) for (x,y) in test_data);
!isnothing(folder) && save_training_info(filename, maxiter, maxiter, validation_loss)
println("Epochs: $(maxiter)    Loss: $(validation_loss)")






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


