using MacroTools;

macro kdc(expr...)
    param_values = Expr(:vect)
    param_names = Expr(:vect)

    function process_arg(arg)
        if @capture(arg, key_ = value_)
            push!(param_values.args, value)
            push!(param_names.args, QuoteNode(key))
        else
            error("Invalid syntax in @kdc macro. Use 'parameter = value' format.")
        end
    end

    if length(expr) == 1 && expr[1] isa Expr && expr[1].head == :block
        for arg in expr[1].args
            if !(arg isa LineNumberNode)
                process_arg(arg)
            end
        end
    else
        for arg in expr
            process_arg(arg)
        end
    end

    return quote
        KDCParams{Float32}($(param_values), $(param_names))
    end
end
