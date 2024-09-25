using MacroTools;

macro ddc(expr)
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

    if expr isa Expr
        if expr.head == :block
            for arg in expr.args
                arg isa LineNumberNode || process_arg(arg)
            end
        else
            process_arg(expr)
        end
    else
        for arg in expr
            process_arg(arg)
        end
    end

    return quote
        DDCParams($(param_values), $(param_names))
    end
end
