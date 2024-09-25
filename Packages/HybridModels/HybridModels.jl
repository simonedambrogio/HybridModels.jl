__precompile__(false)

module HybridModels
    
    include("HybridModel.jl")
    include("@hybridmodel.jl")
    
    export HybridModel, KDCParams, DDCParams, @hybridmodel, @kdc, @ddc
end

