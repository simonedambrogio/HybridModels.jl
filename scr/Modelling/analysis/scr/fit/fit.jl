
# using Pkg; Pkg.update(); Pkg.precompile()
println("Load Libraries and Model...")
path2root = dirname(Base.active_project());
path2output = joinpath(path2root, "scr", "Modelling", "analysis", "output");
path2modellingscr = joinpath(path2root, "scr", "Modelling", "analysis", "scr");
include( joinpath(path2root, "scr", "Modelling", "utils.jl") );
include( joinpath(path2modellingscr, "fit", "training.jl") );
using GLMakie, CSV, Random, MLDataUtils, JSON, JLD2; Random.seed!(123);
using HybridModels
# list all folders in the model folder
[
    include( joinpath(path2modellingscr, "model", file) ) 
    for file in readdir( joinpath(path2modellingscr, "model") ) |> filter(x -> !occursin("._", x))
];

# subject, input2remove = 1, "n-blocked__ρ-same__ρ-other";
# finalPath = joinpath(path2output, string(subject));
function fit_model(subject::Int, input2remove::String, finalPath::String=nothing)
    # Inputs to NN
    begin
        var2remove = string.(split(input2remove, "__"));
        idxinput   = getidx(var2remove);
        inputs = (; 
            allinputs = ["n-same", "n-other", "n-blocked", "ρ-same", "ρ-other", "gaze", "visit"],
            idxinput  = idxinput
        );
        open(joinpath(finalPath, "inputs.json"), "w") do file
            write(file, JSON.json(inputs, 4))
        end;
        hybrid_folder = joinpath(finalPath, "hybridmodel");
        # if hybrid_folder exists, remove all folders that starts with tensor_board in the hybrid_folder
        if isdir(hybrid_folder)
            [
                rm(joinpath(hybrid_folder, folder), recursive=true) 
                for folder in readdir(hybrid_folder) |> filter(x -> startswith(x, "tensorboard_"))
            ];
        end
    end

    println("--- Loading Data ---")
    ct1 = CSV.read( joinpath( path2data, "preprocessed", "binary", "ct1.csv"), DataFrame );
    sbjdf = filter(r -> r.subject==subject, ct1);
    trials = vcat(
        [ # Trials
            sbjdf |> 
            filter(r -> r.trial==trial && r.event in ["switch", "stay", "select"] && r.visit>0) |> 
            df -> Trial(df) for trial in sbjdf.trial |> unique
        ]
    );
    alldata = trial2𝐷y(trials, batchdim=60);
    shuffle!(alldata);
    (train_data, test_data) = splitobs(alldata, at=0.7);
    data = (; train_data, test_data);
    save_object( joinpath(finalPath, "train_data.jld2"), train_data);
    save_object( joinpath(finalPath, "test_data.jld2"),  test_data);


    println("--- Fit Standard Cognitive Model ---")
    begin
        println("Randomly Initialize Parameters...")
        cognitive_model = Agent(KDC(), VoiUCB());
    
        println("Optimize Parameters...")
        # Training model
        loss(model::Agent{KDC{Float32}, VoiUCB{Float32}}, X, y) = Flux.logitcrossentropy(model(X), y);
        ls() = mean(loss(cognitive_model, input, label) for (input, label) in train_data);
        pars = cognitive_model |> m -> Flux.params( m.kdc, m.ddc );
        iter = 500;
        trainBFGS!(ls, pars, iter);
        # Save training output
        filename = joinpath(finalPath, "cognitivemodel", "info.json");
        validation_loss = mean(loss(cognitive_model, input, label) for (input, label) in test_data)
        save_training_info(filename, iter, iter, validation_loss);
        save_object(joinpath(dirname(filename), "model.jld2"), cognitive_model);
    end

        
    println("--- Fit Hybrid Model ---")
    begin
        println("Optimize Both Components...")
        begin
            println("\tRandomly Initialize Parameters...")
            rng = Xoshiro();
            input_dim, ny, nh, γ = 4, 1, fill(32,4), 5;
            θ = VoiNNParams(input_dim, nh, ny, γ; nl=Flux.tanh, learn_γ=true, rng);
            hybrid_model = Agent(
                KDC(),
                VoiNN(θ)
            );
            println("\tStart Optimization...")
            hybrid_loss(model::Agent{KDC{Float32}, VoiNN{Float32}}, X, idxinput, y) = Flux.logitcrossentropy(model(X, idxinput), y);
            maxiter, miniter = 300, 300;
            hpvals = (; η_cm=0.02, η_nn=0.001);
            pars = [Flux.params(hybrid_model.kdc), Flux.params(hybrid_model.ddc)];
            opt  = [Flux.Adam(Float32(hpvals[:η_cm])), Flux.Adam(Float32(hpvals[:η_nn]))];
            filename = joinpath(finalPath, "hybridmodel", "info.json");
            trainAdam!(
                hybrid_model, 
                data, idxinput,
                hybrid_loss, 
                [maxiter, miniter], 
                pars, opt, 
                filename
            );
        end
    
        println("Optimize Knowledge-Driven Component...")
        begin    
            voi  = [ 
                begin
                    voi = extract(hybrid_model, x, idxinput).voi 
                    [voi[1:60]'; voi[61:end]']
                end for (x,_) in train_data 
            ];
    
            # Training model
            loss(model::Agent{KDC{Float32}, VoiNN{Float32}}, X, voi, y) = Flux.logitcrossentropy(model(X, voi), y);
            hybrid_ls() = mean(loss(hybrid_model, input, voi[i], label) for (i, (input, label)) in enumerate(train_data));
            pars = hybrid_model |> m -> Flux.params( m.kdc );
            iter = 500;
            trainBFGS!(hybrid_ls, pars, iter);
        end;
    
        println("Optimization Hybrid Model Given Good KDC Parameters...")
        begin
            println("\tStart Optimization...")
            maxiter, miniter = 3_000, 200;
            θ = VoiNNParams(input_dim, nh, ny, γ; nl=Flux.tanh, learn_γ=true, rng);
            hybrid_model = Agent(
                hybrid_model.kdc,
                VoiNN(θ)
            );
    
            # Training model
            hpvals = (; η_cm=0.0001, η_nn=0.001);
            pars = [Flux.params(hybrid_model.kdc), Flux.params(hybrid_model.ddc)];
            opt  = [Flux.Adam(Float32(hpvals[:η_cm])), Flux.Adam(Float32(hpvals[:η_nn]))];
            filename = joinpath(finalPath, "hybridmodel", "info.json");
            trainAdam!(
                hybrid_model, 
                data, idxinput,
                loss, 
                [maxiter, miniter], 
                pars, opt, 
                filename
            );
        end
    
        println("Final Knowledge-Driven Component Optimization...")
        begin    
            voi  = [ 
                begin
                    voi = extract(hybrid_model, x, idxinput).voi 
                    [voi[1:60]'; voi[61:end]']
                end for (x,_) in train_data 
            ];
    
            # Training model
            pars = hybrid_model |> m -> Flux.params( m.kdc );
            iter = 500;
            trainBFGS!(hybrid_ls, pars, iter);
        end
    end
    
    println("--- Saving model ---")
    hybrid_folder = dirname(filename);
    save_object(joinpath(hybrid_folder, "model.jld2"), hybrid_model)

    println("--- Plotting ---")
    begin
        # Extract variables
        train_data_df = vcat([ DataFrame( first(d) ) for d in train_data]...);
        ex = extract(hybrid_model, train_data_df, idxinput);
        # First-Visit Figure 
        f=Figure(size=(900, 550), fontsize=20, figure_padding=40);
        ax = GLMakie.Axis3(f[1, 1], xlabel="N. Current", ylabel="N. Other", zlabel="VOI"); 
        ( ex.fv .&&   ex.gA ) |> idx -> scatter!(ax, ex.nA[idx], ex.nB[idx], ex.voi[idx], label="Attended")
        ( ex.fv .&& .!ex.gA ) |> idx -> scatter!(ax, ex.nA[idx], ex.nB[idx], ex.voi[idx], label="Unattended")
        f[1, 2] = Legend(f, ax, "", framevisible = false)
        # display(f)
        # Save Video
        framerate = 8
        record(f, joinpath(hybrid_folder, "first-visit.mp4"), collect(4:0.2:16); framerate = framerate) do rotation
            ax.azimuth[] = rotation
        end
        # After First-Visit Figure 
        f=Figure(size=(900, 550), fontsize=20, figure_padding=40);
        ax = GLMakie.Axis3(f[1, 1], xlabel="N. Current", ylabel="N. Other", zlabel="VOI"); 
        ( .!ex.fv .&&   ex.gA ) |> idx -> scatter!(ax, ex.nA[idx], ex.nB[idx], ex.voi[idx], label="Attended")
        ( .!ex.fv .&& .!ex.gA ) |> idx -> scatter!(ax, ex.nA[idx], ex.nB[idx], ex.voi[idx], label="Unattended")
        f[1, 2] = Legend(f, ax, "", framevisible = false)
        # display(f)
        # Save Video
        framerate = 8
        record(f, joinpath(hybrid_folder, "after-first-visit.mp4"), collect(4:0.2:16); framerate = framerate) do rotation
            ax.azimuth[] = rotation
        end;
    end

end;

# subject, input2remove = 1, "n-blocked__ρ-same__ρ-other";
# finalPath = joinpath(path2output, string(subject));
# fit_model(subject, input2remove, finalPath)

test=false
if test

        # using Pkg; Pkg.update(); Pkg.precompile()
    println("Load Libraries and Model...")
    path2root = dirname(Base.active_project());
    path2modellingscr = joinpath(path2root, "scr", "Modelling", "analysis", "scr");
    include( joinpath(path2root, "scr", "Modelling", "utils.jl") );
    include( joinpath(path2modellingscr, "fit", "training.jl") );
    using GLMakie, CSV, Random, MLDataUtils, JSON, JLD2; Random.seed!(123);
    using NNlib: tanh
    using Flux: tanh
    using HybridModels
    # list all folders in the model folder
    [
        include( joinpath(path2modellingscr, "model", file) ) 
        for file in readdir( joinpath(path2modellingscr, "model") ) |> filter(x -> !occursin("._", x))
    ];

    path2output = joinpath(path2root, "scr", "Modelling", "analysis", "output");
    subject, input2remove = 1, "n-blocked__ρ-same__ρ-other";
    finalPath, path2ht, tune_hyperparameters = joinpath(path2output, string(subject)), nothing, false;


    # Inputs to NN
    var2remove = string.(split(input2remove, "__"));
    idxinput   = getidx(var2remove);
    inputs = (; 
        allinputs = ["n-same", "n-other", "n-blocked", "ρ-same", "ρ-other", "gaze", "visit"],
        idxinput  = idxinput
    );
    open(joinpath(finalPath, "inputs.json"), "w") do file
        write(file, JSON.json(inputs, 4))
    end;


    println("\nLoading Data...")
    ct1 = CSV.read( joinpath( path2data, "preprocessed", "binary", "ct1.csv"), DataFrame );
    sbjdf = filter(r -> r.subject==subject, ct1);
    trials = vcat(
        [ # Trials
            sbjdf |> 
            filter(r -> r.trial==trial && r.event in ["switch", "stay", "select"] && r.visit>0) |> 
            df -> Trial(df) for trial in sbjdf.trial |> unique
        ]
    );
    alldata = trial2𝐷y(trials, batchdim=60);
    shuffle!(alldata);
    (train_data, test_data) = splitobs(alldata, at=0.7);
    data = (; train_data, test_data);
    save_object( joinpath(finalPath, "train_data.jld2"), train_data);
    save_object( joinpath(finalPath, "test_data.jld2"),  test_data);


    println("--- Fit Standard Cognitive Model ---")
    println("Randomly Initialize Parameters...")
    cognitive_model = Agent(KDC(), VoiUCB());

    println("Optimize Parameters...")
    # Training model
    loss(model, X, y) = Flux.logitcrossentropy(model(X), y);
    ls() = mean(loss(cognitive_model, input, label) for (input, label) in train_data);
    pars = cognitive_model |> m -> Flux.params( m.kdc, m.ddc );
    iter = 500;
    trainBFGS!(ls, pars, iter);
    # Save training output
    filename = joinpath(finalPath, "cognitivemodel", "info.json");
    validation_loss = mean(loss(cognitive_model, input, label) for (input, label) in test_data)
    save_training_info(filename, iter, iter, validation_loss);
    save_object(joinpath(dirname(filename), "model.jld2"), cognitive_model);



    println("--- Fit Hybrid Model ---")
    println("Optimize Both Components...")
    begin
        println("\tRandomly Initialize Parameters...")
        rng = Xoshiro();
        input_dim, ny, nh, γ = 4, 1, fill(32,4), 5;
        θ = VoiNNParams(input_dim, nh, ny, γ; nl=Flux.tanh, learn_γ=true, rng);
        hybrid_model = Agent(
            KDC(),
            VoiNN(θ)
        );
        println("\tStart Optimization...")
        hybrid_loss(model, X, idxinput, y) = Flux.logitcrossentropy(model(X, idxinput), y);
        maxiter, miniter = 300, 300;
        hpvals = (; η_cm=0.02, η_nn=0.001);
        pars = [Flux.params(hybrid_model.kdc), Flux.params(hybrid_model.ddc)];
        opt  = [Flux.Adam(Float32(hpvals[:η_cm])), Flux.Adam(Float32(hpvals[:η_nn]))];
        filename = joinpath(finalPath, "hybridmodel", "info.json");
        trainAdam!(
            hybrid_model, 
            data, idxinput,
            hybrid_loss, 
            [maxiter, miniter], 
            pars, opt, 
            filename
        );
    end

    println("Optimize Knowledge-Driven Component...")
    begin    
        voi  = [ 
            begin
                voi = extract(hybrid_model, x, idxinput).voi 
                [voi[1:60]'; voi[61:end]']
            end for (x,_) in train_data 
        ];

        # Training model
        loss(model, X, voi, y) = Flux.logitcrossentropy(model(X, voi), y);
        ls() = mean(loss(hybrid_model, input, voi[i], label) for (i, (input, label)) in enumerate(train_data));
        pars = hybrid_model |> m -> Flux.params( m.kdc );
        iter = 500;
        trainBFGS!(ls, pars, iter);
    end;

    println("Optimization Hybrid Model Given Good KDC Parameters...")
    begin
        println("\tStart Optimization...")
        maxiter, miniter = 3_000, 200;
        θ = VoiNNParams(input_dim, nh, ny, γ; nl=Flux.tanh, learn_γ=true, rng);
        hybrid_model = Agent(
            hybrid_model.kdc,
            VoiNN(θ)
        );

        # Training model
        hpvals = (; η_cm=0.0001, η_nn=0.001);
        pars = [Flux.params(hybrid_model.kdc), Flux.params(hybrid_model.ddc)];
        opt  = [Flux.Adam(Float32(hpvals[:η_cm])), Flux.Adam(Float32(hpvals[:η_nn]))];
        filename = joinpath(finalPath, "hybridmodel", "info.json");
        trainAdam!(
            hybrid_model, 
            data, idxinput,
            loss, 
            [maxiter, miniter], 
            pars, opt, 
            filename
        );
    end

    println("Final Knowledge-Driven Component Optimization...")
    begin    
        voi  = [ 
            begin
                voi = extract(hybrid_model, x, idxinput).voi 
                [voi[1:60]'; voi[61:end]']
            end for (x,_) in train_data 
        ];

        # Training model
        loss(model, X, voi, y) = Flux.logitcrossentropy(model(X, voi), y);
        ls() = mean(loss(hybrid_model, input, voi[i], label) for (i, (input, label)) in enumerate(train_data));
        pars = hybrid_model |> m -> Flux.params( m.kdc );
        iter = 500;
        trainBFGS!(ls, pars, iter);
    end


    println("Saving model...")
    hybrid_folder = dirname(filename);


    # Save parameters instead of the whole model
    println("Saving model parameters...")
    hybrid_folder = dirname(filename);
    save_object(joinpath(hybrid_folder, "model.jld2"), hybrid_model)


    println("Plotting...")
    # Extract variables
    train_data_df = vcat([ DataFrame( first(d) ) for d in train_data]...);
    ex = extract(hybrid_model, train_data_df, idxinput);
    # First-Visit Figure 
    f=Figure(size=(900, 550), fontsize=20, figure_padding=40);
    ax = GLMakie.Axis3(f[1, 1], xlabel="N. Current", ylabel="N. Other", zlabel="VOI"); 
    ( ex.fv .&&   ex.gA ) |> idx -> scatter!(ax, ex.nA[idx], ex.nB[idx], ex.voi[idx], label="Attended")
    ( ex.fv .&& .!ex.gA ) |> idx -> scatter!(ax, ex.nA[idx], ex.nB[idx], ex.voi[idx], label="Unattended")
    f[1, 2] = Legend(f, ax, "", framevisible = false)
    # display(f)
    # Save Video
    framerate = 8
    record(f, joinpath(hybrid_folder, "first-visit.mp4"), collect(4:0.2:16); framerate = framerate) do rotation
        ax.azimuth[] = rotation
    end
    # After First-Visit Figure 
    f=Figure(size=(900, 550), fontsize=20, figure_padding=40);
    ax = GLMakie.Axis3(f[1, 1], xlabel="N. Current", ylabel="N. Other", zlabel="VOI"); 
    ( .!ex.fv .&&   ex.gA ) |> idx -> scatter!(ax, ex.nA[idx], ex.nB[idx], ex.voi[idx], label="Attended")
    ( .!ex.fv .&& .!ex.gA ) |> idx -> scatter!(ax, ex.nA[idx], ex.nB[idx], ex.voi[idx], label="Unattended")
    f[1, 2] = Legend(f, ax, "", framevisible = false)
    # display(f)
    # Save Video
    framerate = 8
    record(f, joinpath(hybrid_folder, "after-first-visit.mp4"), collect(4:0.2:16); framerate = framerate) do rotation
        ax.azimuth[] = rotation
    end;


end
