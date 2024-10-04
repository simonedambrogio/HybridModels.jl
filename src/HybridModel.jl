using Functors

struct HybridModel <: AbstractHybridModel
    kdc::AbstractKnowledgeDrivenComponent
    ddc::AbstractDataDrivenComponent
end

(hm::HybridModel)(ddc::AbstractDataDrivenComponent) = HybridModel(hm.kdc, ddc)
(hm::HybridModel)(kdc::AbstractKnowledgeDrivenComponent) = HybridModel(kdc, hm.ddc)

function Base.show(io::IO, ac::HybridModel)
    println(io, "\n --------- Hybrid Model --------- ")
    show(io, ac.kdc)
    show(io, ac.ddc)
    print(io,   "\n -------------------------------- ")
end

function Base.show(m::HybridModel; kdclink::Function=identity, ddclink::Function=identity)
    kdclink == identity ?  show(m.kdc) : show(m.kdc, kdclink)
    ddclink == identity ?  show(m.ddc) : show(m.ddc, ddclink)
end

function predict!(m::HybridModel, params::Vector, X)
    # Assuming params is a flat vector of all parameters
    kdc_params = @view params[1:length(m.kdc.params.params)]
    m.kdc.params.params .= kdc_params

    if length(params) > length(m.kdc.params.params)
        ddc_params = @view params[length(m.kdc.params.params)+1:end]
        m.ddc.params.params .= ddc_params
    end
    
    # Call the original method with updated parameters
    m(X)
end

@functor HybridModel