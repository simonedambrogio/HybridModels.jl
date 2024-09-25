using MacroTools;

macro kdc(expr)
    param_values = Expr(:vect)
    param_names = Expr(:vect)
    link_functions = Expr(:vect)

    function process_arg(arg)
        if isa(arg, Expr) && arg.head == :(=)
            key = arg.args[1]
            value = arg.args[2]
            push!(param_names.args, QuoteNode(key))
            
            if isa(value, Expr) && value.head == :tuple && length(value.args) == 2 &&
               isa(value.args[2], Expr) && value.args[2].head == :macrocall &&
               value.args[2].args[1] == Symbol("@link")
                push!(param_values.args, value.args[1])
                push!(link_functions.args, value.args[2].args[3])
            else
                push!(param_values.args, value)
                push!(link_functions.args, :identity)
            end
        else
            error("Invalid syntax in @kdc macro. Use 'parameter = value' or 'parameter = value, @link(function)' format.")
        end
    end

    if expr.head == :block
        for arg in expr.args
            if !(arg isa LineNumberNode)
                process_arg(arg)
            end
        end
    else
        process_arg(expr)
    end

    return quote
        KDCParams{Float32}($(param_values), $(param_names), $(link_functions))
    end
end

# Example usage:
# k = @kdc begin
#     α = logit(0.5f0), @link(σ)
#     ω = 1.2 
#     β = exp(0.3f0)
# end 

