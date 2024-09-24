path2root = dirname(Base.active_project());
include( joinpath(path2root, "scr", "scr", "utils.jl") );
using Functors, StatsFuns, MacroTools, JLD2, HybridModels;
include("../𝐷.jl"); include("../model.jl"); 
X, y = d = joinpath(path2root, "scr", "outcome", "test_data.jld2") |> load_object |> first;


macro hybridmodel(expr)
    @capture(expr, function func_name_(X_)
        kdc_block_
        ddc_block_
        body__
    end) || error("Invalid @hybridmodel syntax")

    @capture(kdc_block, @kdc(kdc_params__)) || error("Invalid @kdc syntax")
    @capture(ddc_block, @ddc(ddc_params__)) || error("Invalid @ddc syntax")

    # Process KDC parameters
    kdc_params_dict = Dict{Symbol, Any}()
    for param in kdc_params
        if @capture(param, key_ = value_)
            kdc_params_dict[key] = value
        else
            error("Invalid syntax in @kdc block. Use 'parameter = value' format.")
        end
    end

    kdc_param_names = collect(keys(kdc_params_dict))
    kdc_param_values = [kdc_params_dict[name] for name in kdc_param_names]

    # Process DDC parameters
    ddc_params_dict = Dict{Symbol, Any}()
    for param in ddc_params
        if @capture(param, key_ = value_)
            ddc_params_dict[key] = value
        else
            error("Invalid syntax in @ddc block. Use 'parameter = value' format.")
        end
    end

    ddc_param_names = collect(keys(ddc_params_dict))
    ddc_param_values = [ddc_params_dict[name] for name in ddc_param_names]

    # Create expressions to access parameters from KDC and DDC instances
    kdc_param_access = [:($(name) = m.kdc.params[m.kdc.names .== $(QuoteNode(name))][1]) for name in kdc_param_names]
    ddc_param_access = [:($(name) = m.ddc.params[m.ddc.names .== $(QuoteNode(name))][1]) for name in ddc_param_names]
    
    num_kdc_params = length(kdc_param_names)

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

        kdc_params = KDCParams{Float32}(
            Float32[$(kdc_param_values...)],
            $kdc_param_names  # Changed this line
        )
        ddc_params = DDCParams(
            [$(ddc_param_values...)],
            $ddc_param_names  # Changed this line
        )
        $(func_name) = HybridModel(kdc_params, ddc_params)

        $(func_name)
    end

    return esc(result)
end


@hybridmodel function m(X)
    # --- Transform and Extract Parameters --- #
    @kdc λ₀ = logit(0.99f0) ω = logit(0.60f0) κ₁ = logit(0.25f0) λ₂ = logit(0.01f0) τ = logit(0.08f0); 
    @ddc β = 1.2f0

    
    # --- Transform parameters --- #
    λ₀, ω, κ₁, λ₂, τ = σ(λ₀), σ(ω), σ(κ₁), σ(λ₂), σ(τ);

    # --- Compute α and β to establish the shape of the beta posterior distribution --- #
    nL = n0L(X) .+ n1_0L(X, ω)      .+ n1_1L(X) .+ n2_0L(X, λ₂) .+ n2_1L(X)
    rL = r0L(X) .+ r1_0L(X, λ₀, nL) .+ r1_1L(X) .+ r2_0L(X, λ₂) .+ r2_1L(X)

    nR = n0R(X) .+ n1_0R(X, ω)      .+ n1_1R(X) .+ n2_0R(X, λ₂) .+ n2_1R(X)
    rR = r0R(X) .+ r1_0R(X, λ₀, nR) .+ r1_1R(X) .+ r2_0R(X, λ₂) .+ r2_1R(X)

    # --- Value of Select --- #    
    ρL = rL ./ nL
    ρR = rR ./ nR

    # --- Value of Information --- #    
    voiL = ucb([nL nR]', β)'
    voiR = ucb([nR nL]', β)'

    # --- Cost of Information --- #    
    coiL = (1f0 .- X.gL) .* κ₁
    coiR = (1f0 .- X.gR) .* κ₁

    return [(voiL .- coiL) (voiR .- coiR) (ρL .- ρR) (ρR .- ρL)]' ./ τ
end;


m(X)

θ = [
    randn(Float32) |> abs, # λ₀
    randn(Float32) |> abs, # ω
    -(rand(Float32) + 1), # κ₁
    -(rand(Float32) + 4), # λ₂
    -(rand(Float32) - 1),  # τ
    randn(Float32) |> abs, # β
];


(m)(X, θ)
(m)(X)





using Random, RobustNeuralNetworks
input_dim = 2;
rng = Xoshiro();
ny = 1
nh = fill(32,4)   # 4 hidden layers, each with 32 neurons
γ = 5  



@hybridmodel function mymodel(X)
    # --- Transform and Extract Parameters --- #
    @kdc λ₀ = 0.99f0 ω = 0.60f0 κ₁ = 0.25f0 λ₂ = 0.01f0 τ = 0.08f0 
    @ddc θ = DenseLBDNParams{Float32}(input_dim, nh, ny, γ; nl=Flux.tanh, learn_γ=true, rng)

    # --- Compute α and β to establish the shape of the beta posterior distribution --- #
    nL = n0L(X) .+ n1_0L(X, ω)      .+ n1_1L(X) .+ n2_0L(X, λ₂) .+ n2_1L(X)
    rL = r0L(X) .+ r1_0L(X, λ₀, nL) .+ r1_1L(X) .+ r2_0L(X, λ₂) .+ r2_1L(X)

    nR = n0R(X) .+ n1_0R(X, ω)      .+ n1_1R(X) .+ n2_0R(X, λ₂) .+ n2_1R(X)
    rR = r0R(X) .+ r1_0R(X, λ₀, nR) .+ r1_1R(X) .+ r2_0R(X, λ₂) .+ r2_1R(X)

    # --- Value of Select --- #    
    ρL = rL ./ nL
    ρR = rR ./ nR

    # --- Value of Information --- #    
    nn = LBDN(θ)
    voiL = nn([nL nR]')
    voiR = nn([nR nL]')

    # --- Cost of Information --- #    
    coiL = (1f0 .- X.gL) .* κ₁
    coiR = (1f0 .- X.gR) .* κ₁

    return [(voiL' .- coiL) (voiR' .- coiR) (ρL .- ρR) (ρR .- ρL)]' ./ τ
end;

mymodel(X)

mymodel.kdc

