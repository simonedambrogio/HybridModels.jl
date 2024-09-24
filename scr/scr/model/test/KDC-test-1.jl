path2root = dirname(Base.active_project());
include( joinpath(path2root, "scr", "Modelling", "utils.jl") );
using Functors, StatsFuns, MacroTools, JLD2, HybridModels;
include("𝐷.jl");
X, y = load_object("scr/Modelling/analysis/output/1/test_data.jld2")[1];

struct KDCParams{T} <: HybridModels.AbstractKnowledgeDrivenComponent
    params::Vector{T}
    names::Vector{Symbol}
end

function Base.show(io::IO, p::KDCParams)
    print(io, "KDCParams{$(eltype(p.params))}(\n")
    for (name, value) in zip(p.names, p.params)
        println(io, "  ", name, " = ", value)
    end
    print(io, ")")
end

function KDCParams{T}(; kwargs...) where T
    params = T[]
    names = Symbol[]
    for (key, value) in kwargs
        push!(params, value)
        push!(names, key)
    end
    KDCParams{T}(params, names)
end

struct Agent{T} <: HybridModels.AbstractAgent
    ddc::KDCParams{T}
end;


macro hybridmodel(expr)
    @capture(expr, function func_name_(X_)
        kdc_block_
        body__
    end) || error("Invalid @hybridmodel syntax")

    @capture(kdc_block, @kdc(kdc_params__)) || error("Invalid @kdc syntax")

    kdc_params_dict = Dict{Symbol, Any}()
    for param in kdc_params
        if @capture(param, key_ = value_)
            kdc_params_dict[key] = value
        else
            error("Invalid syntax in @kdc block. Use 'parameter = value' format.")
        end
    end

    param_names = collect(keys(kdc_params_dict))
    param_values = [kdc_params_dict[name] for name in param_names]

    # Create expressions to access parameters from KDC instance
    param_access = [:($(name) = m.ddc.params[m.ddc.names .== $(QuoteNode(name))][1]) for name in param_names]

    kdc_struct = quote
        function (m::Agent)(X)
            $(param_access...)
            $(body...)
        end
    end

    result = quote
        $kdc_struct

        kdc_params = KDCParams{Float32}(
            Float32[$(param_values...)],
            $param_names
        )
        $(func_name) = Agent{Float32}(kdc_params)

        $(func_name)
    end

    return esc(result)
end


@hybridmodel function mymodel(X)
    # --- Transform and Extract Parameters --- #
    @kdc λ₀ = 0.99f0 ω = 0.60f0 κ₁ = 0.25f0 λ₂ = 0.01f0 τ = 0.08f0

    # --- Compute α and β to establish the shape of the beta posterior distribution --- #
    nL = n0L(X) .+ n1_0L(X, ω)      .+ n1_1L(X) .+ n2_0L(X, λ₂) .+ n2_1L(X)
    rL = r0L(X) .+ r1_0L(X, λ₀, nL) .+ r1_1L(X) .+ r2_0L(X, λ₂) .+ r2_1L(X)

    nR = n0R(X) .+ n1_0R(X, ω)      .+ n1_1R(X) .+ n2_0R(X, λ₂) .+ n2_1R(X)
    rR = r0R(X) .+ r1_0R(X, λ₀, nR) .+ r1_1R(X) .+ r2_0R(X, λ₂) .+ r2_1R(X)

    # --- Value of Select --- #    
    ρL = rL ./ nL
    ρR = rR ./ nR

    
    # --- Cost of Information --- #    
    coiL = (1f0 .- X.gL) .* κ₁
    coiR = (1f0 .- X.gR) .* κ₁

    return [(coiL) (coiR) (ρL .- ρR) (ρR .- ρL)]' ./ τ
end

mymodel(X)


