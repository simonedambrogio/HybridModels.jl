using MacroTools;
# include("KDC.jl");

macro kdc(args...)
    param_values = Expr(:vect)
    param_names = Expr(:vect)
    link_functions = Expr(:vect)

    function process_arg(arg)
        if isa(arg, Expr) && arg.head == :(=)
            key = arg.args[1]
            value = arg.args[2]
            push!(param_names.args, QuoteNode(key))
            
            if isa(value, Expr) && value.head == :call && value.args[1] == :(=>)
                push!(param_values.args, value.args[2])
                push!(link_functions.args, value.args[3])
            else
                push!(param_values.args, value)
                push!(link_functions.args, :identity)
            end
        else
            error("Invalid syntax in @kdc macro. Use 'parameter = value' or 'parameter = value => function' format.")
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
        KDC(ComponentParams($(esc(param_values)), $(esc(param_names)), $(esc(link_functions))))
    end
end

# Example usage:
# using StatsFuns, Flux;
k = @kdc begin
    α = 0.5f0 => sum
    ω = 1.2f0
    β = exp(0.3f0)
end;

# k.params.params
# k.params.names
# k.params.link

# k2 = @kdc α = logit(0.5f0) => σ ω = 1.2f0 β = exp(0.3f0)

# k2.params.params
# k2.params.names
# k2.params.link

# k3 = @kdc α = logit(0.5f0) => σ ω = 1.2f0 β = exp(0.3f0)
# k3.params.params
# k3.params.names
# k3.params.link

# @kdc begin 
#     A = log(1) => exp # starting point of each accumulator is sampled uniformly between k=[0, A]
#     b = log(5) => exp # boundary is sampled uniformly between [k, k+b]
#     τ = log(0.3) => exp # Non-decision time is an additive constant representing encoding and motor response time.
# end 

# @kdc begin
#     α = 0.5f0 => sum
#     ω = 1.2f0
#     β = exp(0.3f0)
# end
