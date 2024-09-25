using MacroTools;
include("@kdc.jl"); 
include("@ddc.jl");

macro hybridmodel(expr)
    @capture(expr, function func_name_(X_)
        kdc_block_
        ddc_block_
        body__
    end) || error("Invalid @hybridmodel syntax")

    @capture(kdc_block, @kdc(kdc_params__)) || error("Invalid @kdc syntax")
    @capture(ddc_block, @ddc(ddc_params__)) || error("Invalid @ddc syntax")

    # Handle both block and non-block syntax for @kdc
    kdc_params_expr = if length(kdc_params) == 1 && kdc_params[1].head == :block
        Expr(:macrocall, Symbol("@kdc"), LineNumberNode(0), kdc_params[1])
    else
        Expr(:macrocall, Symbol("@kdc"), LineNumberNode(0), kdc_params...)
    end

    # Handle both block and non-block syntax for @ddc
    ddc_params_expr = if length(ddc_params) == 1 && ddc_params[1].head == :block
        Expr(:macrocall, Symbol("@ddc"), LineNumberNode(0), ddc_params[1])
    else
        Expr(:macrocall, Symbol("@ddc"), LineNumberNode(0), ddc_params...)
    end

    # Extract parameter names and values from kdc_params and ddc_params
    kdc_param_names = []
    kdc_param_values = []
    ddc_param_names = []
    ddc_param_values = []
    
    function extract_params(params, names, values)
        for param in params
            if @capture(param, name_ = value_)
                push!(names, name)
                push!(values, value)
            elseif param isa Expr && param.head == :block
                extract_params(param.args, names, values)
            end
        end
    end

    extract_params(kdc_params, kdc_param_names, kdc_param_values)
    extract_params(ddc_params, ddc_param_names, ddc_param_values)

    # Infer link functions for KDC parameters
    kdc_link_functions = Dict{Symbol, Symbol}()
    for body_expr in body
        if @capture(body_expr, (vars__,) = (funcs__,))
            for (var, func) in zip(vars, funcs)
                if var in kdc_param_names && func isa Expr && func.args[1] isa Symbol
                    kdc_link_functions[var] = func.args[1]
                end
            end
        end
    end
    println(kdc_link_functions)
    
    # Create expressions to access parameters from KDC and DDC instances
    kdc_param_access = [:($(name) = m.kdc.params[m.kdc.names .== $(QuoteNode(name))][1]) for name in kdc_param_names]
    ddc_param_access = [:($(name) = m.ddc.params[m.ddc.names .== $(QuoteNode(name))][1]) for name in ddc_param_names]

    hybrid_model = quote
        function (m::HybridModel)(X)
            $(kdc_param_access...)
            $(ddc_param_access...)
            $(body...)
        end

        function (m::HybridModel)(params::Vector, X)
            # Assuming params is a flat vector of all parameters
            kdc_params = @view params[1:m.kdc.n_params]
            m.kdc.params .= kdc_params
        
            if length(params) > m.kdc.n_params
                ddc_params = @view params[m.kdc.n_params+1:end]
                m.ddc.params .= ddc_params
            end
            
            # Call the original method with updated parameters
            m(X)
        end
    end

    result = quote
        $hybrid_model

        kdc_params = KDCParams{Float32}($(Expr(:vect, kdc_param_values...)), $(Expr(:vect, QuoteNode.(kdc_param_names)...)))
        ddc_params = DDCParams($(Expr(:vect, ddc_param_values...)), $(Expr(:vect, QuoteNode.(ddc_param_names)...)))
        $(func_name) = HybridModel(kdc_params, ddc_params)

        # Add inferred link functions to the result
        kdc_link_functions = $(QuoteNode(kdc_link_functions))

        $(func_name)
    end

    return esc(result)
end
