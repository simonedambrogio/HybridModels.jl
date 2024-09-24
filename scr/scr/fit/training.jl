
using Zygote, Optim, JSON, ProgressBars, JLD2, Zygote, 
JLD2, DataStructures, TensorBoardLogger, FluxOptTools, Logging;
using HybridModels: AbstractAgent;

function trainBFGS!(
        loss_function::Function, 
        pars::Zygote.Params, 
        iter::Int
    )

    println("Computing gradient...")
    _, _, fg!, p0 = optfuns(loss_function, pars);
    println("Optimizing...")
    Optim.optimize(Optim.only_fg!(fg!), p0, BFGS(), Optim.Options(iterations=iter, store_trace=true));
    
    # validation_loss = mean(loss_function(model, input, label) for (input, label) in data[2]);

    # Save training info
    # !isnothing(filename) && save_training_info(filename, iter, iter, validation_loss);
    
    # return validation_loss
end




function train!(
        model, 
        data, 
        lossfun, 
        iter,
        parsnn, 
        parscm, 
        optnn, 
        optcm, 
        finalPath=nothing
    )

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

function trainAdam!(
        model::AbstractAgent, 
        data::NamedTuple, 
        loss::Function, 
        iter::Vector{Int},
        pars::Zygote.Params, 
        opt::Flux.Optimise.AbstractOptimiser,
        filename=nothing
    ) 

    println("Start Training...")
    train_data, test_data = data;
    maxiter, miniter = iter;
    folder = isnothing(filename) ? nothing : dirname(filename);
    
    # Set up TensorBoard logger
    if !isnothing(folder)
        tb_logger = TBLogger(joinpath(folder, "tensorboard_logs"))
        println("run the following command to open the tensorboard:")
        println("\ntensorboard --logdir $folder\n")
    end

    loss_array = CircularBuffer{Float32}(5);
    isworse = 0f0;
    for epoch in ProgressBar(1:maxiter)
        
        validation_loss = mean(loss(model,x,y) for (x,y) in test_data);
        push!(loss_array, validation_loss)
        isworse = (isworse + 1) * Int(findmin(loss_array)[2]!=5)

        if epoch>miniter && isworse>=10
            println("Epochs: $(epoch)    Loss: $(validation_loss)")
            !isnothing(folder) && save_training_info(filename, maxiter, epoch, validation_loss)
            return nothing
        end
        
        for (input, label) in train_data
            # Calculate Gradient and Update Neural Network
            grads = gradient(() -> loss(model, input, label), pars);
            Flux.update!(opt, pars, grads);
        end

        # Log to TensorBoard
        if !isnothing(folder)
            with_logger(tb_logger) do
                @info "training" validation_loss=validation_loss
            end
        end
        
    end 
    
    validation_loss = mean(loss(model,x,y) for (x,y) in test_data);
    !isnothing(folder) && save_training_info(filename, maxiter, maxiter, validation_loss)
    println("Epochs: $(maxiter)    Loss: $(validation_loss)")

end;

function trainAdam!(
        model::AbstractAgent, 
        data::NamedTuple, 
        loss::Function, 
        iter::Vector{Int},
        pars::Vector{<:Zygote.Params}, 
        opt::Vector{<:Flux.Optimise.AbstractOptimiser},
        filename=nothing
    )

    println("Start Training...")
    train_data, test_data = data;
    maxiter, miniter = iter;
    folder = isnothing(filename) ? nothing : dirname(filename);

    # Set up TensorBoard logger
    if !isnothing(folder)
        tb_logger = TBLogger(joinpath(folder, "tensorboard_logs"))
        println("run the following command to open the tensorboard:")
        println("\ntensorboard --logdir $folder\n")
    end

    loss_array = CircularBuffer{Float32}(5);
    isworse = 0f0;

    iter = ProgressBar(1:maxiter);
    for epoch in iter
        
        validation_loss = mean(loss(model,x,y) for (x,y) in test_data);
        training_loss = mean(loss(model,x,y) for (x,y) in train_data);
        
        set_postfix(iter, 
            TrainLoss = round(training_loss; digits=4), 
            ValLoss   = round(validation_loss; digits=4)
        )

        push!(loss_array, validation_loss)
        isworse = (isworse + 1) * Int(findmin(loss_array)[2]!=5)

        if epoch>miniter && isworse>=10
            println("Epochs: $(epoch)    Loss: $(validation_loss)")
            !isnothing(folder) && save_training_info(filename, maxiter, epoch, validation_loss)
            return nothing
        end
        
        for (input, label) in train_data
            for (i, par) in enumerate(pars)
                # Calculate Gradient and Update Neural Network
                grads = gradient(() -> loss(model, input, label), par);
                Flux.update!(opt[i], par, grads);
            end
        end


        # Log to TensorBoard
        if !isnothing(folder)
            with_logger(tb_logger) do
                @info "training" validation_loss=validation_loss
            end
        end
        
    end 

    validation_loss = mean(loss(model,x,y) for (x,y) in test_data);
    !isnothing(folder) && save_training_info(filename, maxiter, maxiter, validation_loss)
    println("Epochs: $(maxiter)    Loss: $(validation_loss)")

end;

function trainAdam!(
        model::AbstractAgent, 
        data::NamedTuple, 
        idxinput::BitVector,
        loss::Function, 
        iter::Vector{Int},
        pars::Vector{<:Zygote.Params}, 
        opt::Vector{<:Flux.Optimise.AbstractOptimiser},
        filename=nothing
    )

    println("Start Training...")
    train_data, test_data = data;
    maxiter, miniter = iter;
    folder = isnothing(filename) ? nothing : dirname(filename);

    # Set up TensorBoard logger
    if !isnothing(folder)
        tb_logger = TBLogger(joinpath(folder, "tensorboard_logs"))
        println("run the following command to open the tensorboard:")
        println("\ntensorboard --logdir $folder\n")
    end

    loss_array = CircularBuffer{Float32}(5);
    isworse = 0f0;

    iter = ProgressBar(1:maxiter);
    for epoch in iter
        
        validation_loss = mean(loss(model,x, idxinput, y) for (x,y) in test_data);
        training_loss = mean(loss(model,x, idxinput, y) for (x,y) in train_data);
        
        set_postfix(iter, 
            TrainLoss = round(training_loss; digits=4), 
            ValLoss   = round(validation_loss; digits=4)
        )

        push!(loss_array, validation_loss)
        isworse = (isworse + 1) * Int(findmin(loss_array)[2]!=5)

        if epoch>miniter && isworse>=10
            println("Epochs: $(epoch)    Loss: $(validation_loss)")
            !isnothing(folder) && save_training_info(filename, maxiter, epoch, validation_loss)
            return nothing
        end
        
        for (input, label) in train_data
            for (i, par) in enumerate(pars)
                # Calculate Gradient and Update Neural Network
                grads = gradient(() -> loss(model, input, idxinput, label), par);
                Flux.update!(opt[i], par, grads);
            end
        end


        # Log to TensorBoard
        if !isnothing(folder)
            with_logger(tb_logger) do
                @info "training" validation_loss=validation_loss
            end
        end
        
    end 

    validation_loss = mean(loss(model,x,idxinput,y) for (x,y) in test_data);
    !isnothing(folder) && save_training_info(filename, maxiter, maxiter, validation_loss)
    println("Epochs: $(maxiter)    Loss: $(validation_loss)")

end;

function save_training_info(filename, maxiter, niter, test_loss)

    mkpath(dirname(filename))
    training_info = Dict(
        "maxiter" => maxiter, 
        "niter" => niter, 
        "test_loss" => test_loss
    )
    
    open(filename, "w") do file
        write(file, JSON.json(training_info, 4))
    end

end;