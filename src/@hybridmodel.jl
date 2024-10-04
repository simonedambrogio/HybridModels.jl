using MacroTools;
# include("HybridModel.jl");
# include("@kdc.jl"); 
# include("@ddc.jl");

macro hybridmodel(expr)
    @capture(expr, function func_name_(X_)
        kdc_block_
        ddc_block_
        body__
    end) || error("Invalid @hybridmodel syntax")

    @capture(kdc_block, @kdc(kdc_params__)) || error("Invalid @kdc syntax")
    @capture(ddc_block, @ddc(ddc_params__)) || error("Invalid @ddc syntax")

    # Handle both block and non-block syntax for @kdc and @ddc
    kdc_params_expr = length(kdc_params) == 1 && kdc_params[1].head == :block ?
        Expr(:macrocall, Symbol("@kdc"), LineNumberNode(0), kdc_params[1]) :
        Expr(:macrocall, Symbol("@kdc"), LineNumberNode(0), kdc_params...)

    ddc_params_expr = length(ddc_params) == 1 && ddc_params[1].head == :block ?
        Expr(:macrocall, Symbol("@ddc"), LineNumberNode(0), ddc_params[1]) :
        Expr(:macrocall, Symbol("@ddc"), LineNumberNode(0), ddc_params...)

    # Extract parameter names from kdc_params and ddc_params
    kdc_param_names = []
    ddc_param_names = []
    
    function extract_params(params, names)
        for param in params
            if @capture(param, name_ = value_)
                push!(names, name)
            elseif param isa Expr && param.head == :block
                extract_params(param.args, names)
            end
        end
    end

    extract_params(kdc_params, kdc_param_names)
    extract_params(ddc_params, ddc_param_names)
    
    # Create expressions to access and transform parameters from KDC and DDC instances
    kdc_param_access = [:($(name) = m.kdc.params.link[m.kdc.params.names .== $(QuoteNode(name))][1](m.kdc.params.params[m.kdc.params.names .== $(QuoteNode(name))][1])) for name in kdc_param_names]
    ddc_param_access = [:($(name) = m.ddc.params.link[m.ddc.params.names .== $(QuoteNode(name))][1](m.ddc.params.params[m.ddc.params.names .== $(QuoteNode(name))][1])) for name in ddc_param_names]

    hybrid_model = quote
        function (m::HybridModel)(X)
            $(kdc_param_access...)
            $(ddc_param_access...)
            $(body...)
        end
    end

    result = quote
        $hybrid_model

        kdc = $kdc_params_expr
        ddc = $ddc_params_expr
        $(func_name) = HybridModel(kdc, ddc)

        $(func_name)
    end

    return esc(result)
end


# @hybridmodel function m(X)
#     # --- Transform and Extract Parameters --- #
#     @kdc begin
#         λ₀ = logit(0.99f0) 
#         ω  = logit(0.60f0) 
#         κ₁ = logit(0.25f0) 
#         λ₂ = logit(0.01f0) 
#         τ  = logit(0.08f0) 
#     end
#     @ddc β = 1.2f0
    
# end;
