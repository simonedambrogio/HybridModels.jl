using Functors, HybridModels, StatsFuns, MacroTools

struct KDCParams{T} <: HybridModels.AbstractParams
    params::Vector{T}
    names::Vector{Symbol}
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

macro kdc(args...)
    param_values = Expr(:vect)
    param_names = Expr(:vect)
    for arg in args
        if @capture(arg, key_ = value_)
            push!(param_values.args, value)
            push!(param_names.args, QuoteNode(key))
        else
            error("Invalid syntax in @kdc macro. Use 'parameter = value' format.")
        end
    end

    return quote
        KDCParams{Float32}($(param_values), $(param_names))
    end
end

# Test the macro
kdc_params = @kdc begin
    λ₀ = 0.99f0 
    ω = 0.60f0 
    κ₁ = 0.25f0
end;

println(kdc_params)

function Base.show(io::IO, p::KDCParams)
    print(io, "KDCParams{$(eltype(p.params))}(")
    for (name, value) in zip(p.names, p.params)
        print(io, name, " = ", value, ", ")
    end
    print(io, ")")
end

