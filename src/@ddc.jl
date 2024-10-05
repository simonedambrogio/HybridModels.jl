using MacroTools;
# include("DDC.jl");

macro ddc(args...)
    param_values = Expr(:vect)
    param_names = Expr(:vect)
    link_functions = Expr(:vect)

    function process_arg(arg)
        if isa(arg, Expr) && arg.head == :(=)
            key = arg.args[1]
            value = arg.args[2]
            push!(param_names.args, QuoteNode(key))
            
            if isa(value, Expr) && value.head == :call && value.args[1] == :(|>)
                push!(param_values.args, value.args[2])
                push!(link_functions.args, value.args[3])
            else
                push!(param_values.args, value)
                push!(link_functions.args, :identity)
            end
        else
            error("Invalid syntax in @kdc macro. Use 'parameter = value' or 'parameter = value |> function' format.")
        end
    end

    if length(args) == 1 && isa(args[1], Expr) && args[1].head == :block
        for arg in args[1].args
            if !(arg isa LineNumberNode)
                process_arg(arg)
            end
        end
    else
        foreach(process_arg, args)
    end

    return quote
        DDC(
            ComponentParams(
                $(esc(param_values)), 
                $(esc(param_names)), 
                $(esc(link_functions))
            )
        )
    end
end


# Example usage:
# using StatsFuns, Flux;
# d = @ddc begin
#     α = logit(0.5f0) |> σ
#     ω = 1.2f0
#     β = exp(0.3f0)
# end;

# d.params.params
# d.params.names
# d.params.link

# d2 = @kdc α = logit(0.5f0) |> σ ω = 1.2f0 β = exp(0.3f0)

# d2.params.params
# d2.params.names
# d2.params.link

# d3 = @kdc α = logit(0.5f0) |> σ ω = 1.2f0 β = exp(0.3f0)
# d3.params.params
# d3.params.names
# d3.params.link


# using Random, RobustNeuralNetworks, ProgressBars, JSON;
# input_dim = 2;
# rng = Xoshiro();
# ny = 1
# nh = fill(32,4)   # 4 hidden layers, each with 32 neurons
# γ = 5             # Start with a Lipschitz bound of 5
# nnpars = DenseLBDNParams{Float32}(input_dim, nh, ny, γ; nl=Flux.tanh, learn_γ=true, rng);
# d = @ddc begin
#     θ = nnpars
# end;

# d.params.params
# d.params.names
# d.params.link

# d = @ddc θ = nnpars;

# d.params.params
# d.params.names
# d.params.link

