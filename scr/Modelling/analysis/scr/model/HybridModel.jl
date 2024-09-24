using Functors, HybridModels
using HybridModels: AbstractAgent
include("KDC.jl")
include("VoiUCB.jl")
include("𝐷.jl")

function (m::AbstractAgent)(X)

    # --- Transform and Extract Parameters --- #
    c = transformpars(m.kdc)

    # --- Compute α and β to establish the shape of the beta posterior distribution --- #
    nL = n0L(m, X, c) .+ n1_0L(m, X, c)     .+ n1_1L(m, X, c) .+ n2_0L(m, X, c) .+ n2_1L(m, X, c)
    rL = r0L(m, X, c) .+ r1_0L(m, X, c, nL) .+ r1_1L(m, X, c) .+ r2_0L(m, X, c) .+ r2_1L(m, X, c)

    nR = n0R(m, X, c) .+ n1_0R(m, X, c)     .+ n1_1R(m, X, c) .+ n2_0R(m, X, c) .+ n2_1R(m, X, c)
    rR = r0R(m, X, c) .+ r1_0R(m, X, c, nR) .+ r1_1R(m, X, c) .+ r2_0R(m, X, c) .+ r2_1R(m, X, c)

    # --- Value of Select --- #    
    ρL = rL ./ nL
    ρR = rR ./ nR

    # --- Value of Information --- #    
    voiL = m.ddc([nL nR]')[:]
    voiR = m.ddc([nR nL]')[:]

    # --- Cost of Information --- #    
    coiL = (1f0 .- X.gL) .* c.κ₁
    coiR = (1f0 .- X.gR) .* c.κ₁

    return [(voiL .- coiL) (voiR .- coiR) (ρL .- ρR) (ρR .- ρL)]' ./ c.τ
end;

function (m::AbstractAgent)(X, idxinput::BitVector)

    # --- Transform and Extract Parameters --- #
    c = transformpars(m.kdc)

    # --- Compute α and β to establish the shape of the beta posterior distribution --- #
    nL = n0L(m, X, c) .+ n1_0L(m, X, c)     .+ n1_1L(m, X, c) .+ n2_0L(m, X, c) .+ n2_1L(m, X, c)
    rL = r0L(m, X, c) .+ r1_0L(m, X, c, nL) .+ r1_1L(m, X, c) .+ r2_0L(m, X, c) .+ r2_1L(m, X, c)

    nR = n0R(m, X, c) .+ n1_0R(m, X, c)     .+ n1_1R(m, X, c) .+ n2_0R(m, X, c) .+ n2_1R(m, X, c)
    rR = r0R(m, X, c) .+ r1_0R(m, X, c, nR) .+ r1_1R(m, X, c) .+ r2_0R(m, X, c) .+ r2_1R(m, X, c)

    # --- Value of Select --- #    
    ρL = rL ./ nL
    ρR = rR ./ nR

    # --- Value of Information --- #    
    input = Matrix([nL./100 nR./100 X.nB./100 ρL ρR X.gL X.first_visit]')[idxinput, :]
    voiL = m.ddc(input)[:]
    input = Matrix([nR./100 nL./100 X.nB./100 ρR ρL X.gR X.first_visit]')[idxinput, :]
    voiR = m.ddc(input)[:]

    # --- Cost of Information --- #    
    coiL = (1f0 .- X.gL) .* c.κ₁
    coiR = (1f0 .- X.gR) .* c.κ₁

    return [(voiL .- coiL) (voiR .- coiR) (ρL .- ρR) (ρR .- ρL)]' ./ c.τ
end;

function (m::AbstractAgent)(X, voi::Matrix)

    # --- Transform and Extract Parameters --- #
    c = transformpars(m.kdc)

    # --- Compute α and β to establish the shape of the beta posterior distribution --- #
    nL = n0L(m, X, c) .+ n1_0L(m, X, c)     .+ n1_1L(m, X, c) .+ n2_0L(m, X, c) .+ n2_1L(m, X, c)
    rL = r0L(m, X, c) .+ r1_0L(m, X, c, nL) .+ r1_1L(m, X, c) .+ r2_0L(m, X, c) .+ r2_1L(m, X, c)

    nR = n0R(m, X, c) .+ n1_0R(m, X, c)     .+ n1_1R(m, X, c) .+ n2_0R(m, X, c) .+ n2_1R(m, X, c)
    rR = r0R(m, X, c) .+ r1_0R(m, X, c, nR) .+ r1_1R(m, X, c) .+ r2_0R(m, X, c) .+ r2_1R(m, X, c)

    # --- Value of Select --- #    
    ρL = rL ./ nL
    ρR = rR ./ nR

    # --- Cost of Sampling --- #    
    costL = (1f0 .- X.gL) .* c.κ₁
    costR = (1f0 .- X.gR) .* c.κ₁

    [(voi[1,:] .- costL) (voi[2,:] .- costR) (ρL .- ρR) (ρR .- ρL)]' ./ c.τ

end;

function extract(m::AbstractAgent, X; type=:long)

    # --- Tranform and Extract Parameters --- #
    c = transformpars(m.kdc)

    # --- Compute α and β to establish the shape of the beta posterior distribution --- #
    nL = @inline n0L(m, X, c) .+ n1_0L(m, X, c)     .+ n1_1L(m, X, c) .+ n2_0L(m, X, c) .+ n2_1L(m, X, c)
    rL = @inline r0L(m, X, c) .+ r1_0L(m, X, c, nL) .+ r1_1L(m, X, c) .+ r2_0L(m, X, c) .+ r2_1L(m, X, c)

    nR = @inline n0R(m, X, c) .+ n1_0R(m, X, c)     .+ n1_1R(m, X, c) .+ n2_0R(m, X, c) .+ n2_1R(m, X, c)
    rR = @inline r0R(m, X, c) .+ r1_0R(m, X, c, nR) .+ r1_1R(m, X, c) .+ r2_0R(m, X, c) .+ r2_1R(m, X, c)

    # --- Value of Select --- #    
    ρL = rL ./ nL
    ρR = rR ./ nR

    # --- Value of Information --- #    
    voiL = m.ddc([nL nR]')[:]
    voiR = m.ddc([nR nL]')[:]

    # --- Cost of Sampling --- #    
    costL = (1 .- X.gL) .* c.κ₁
    costR = (1 .- X.gR) .* c.κ₁

    QsamplingL = (voiL .- costL)
    QsamplingR = (voiR .- costR)

    if type==:long
        (;
            voi = vcat( voiL[:], voiR[:] ),
            sampling = vcat( QsamplingL[:], QsamplingR[:] ), #vcat( Vector{Float32}(QsamplingL), Vector{Float32}(QsamplingR)),
            nA = vcat( Vector{Float32}(nL), Vector{Float32}(nR)),
            nB = vcat( Vector{Float32}(nR), Vector{Float32}(nL)),
            nC = vcat( Vector{Float32}(X.nB), Vector{Float32}(X.nB)),
            ρA = vcat( Vector{Float32}(ρL), Vector{Float32}(ρR)),
            ρB = vcat( Vector{Float32}(ρR), Vector{Float32}(ρL)), 
            gA = vcat( Vector{Float32}(X.gL), Vector{Float32}(X.gR)) .|> Bool, 
            fv = vcat( Vector{Float32}(X.first_visit), Vector{Float32}(X.first_visit) ) .|> Bool
        )
    elseif type==:wide

        (;
            voiL       = bonusL[:], 
            voiR       = bonusR[:],
            coiL       = costL[:], 
            coiR       = costR[:], 
            QsamplingL = QsamplingL[:], 
            QsamplingR = QsamplingR[:],
            nL         = nL,
            nR         = nR,
            ρL         = ρL,
            ρR         = ρR,
            gL         = X.gL .|> Bool, 
            gR         = X.gR .|> Bool, 
            fv         = X.first_visit .|> Bool
        )

    end

end;

function extract(m::AbstractAgent, X, idxinput::BitVector; type=:long)

    # --- Tranform and Extract Parameters --- #
    c = transformpars(m.kdc)

    # --- Compute α and β to establish the shape of the beta posterior distribution --- #
    nL = @inline n0L(m, X, c) .+ n1_0L(m, X, c)     .+ n1_1L(m, X, c) .+ n2_0L(m, X, c) .+ n2_1L(m, X, c)
    rL = @inline r0L(m, X, c) .+ r1_0L(m, X, c, nL) .+ r1_1L(m, X, c) .+ r2_0L(m, X, c) .+ r2_1L(m, X, c)

    nR = @inline n0R(m, X, c) .+ n1_0R(m, X, c)     .+ n1_1R(m, X, c) .+ n2_0R(m, X, c) .+ n2_1R(m, X, c)
    rR = @inline r0R(m, X, c) .+ r1_0R(m, X, c, nR) .+ r1_1R(m, X, c) .+ r2_0R(m, X, c) .+ r2_1R(m, X, c)

    # --- Value of Select --- #    
    ρL = rL ./ nL
    ρR = rR ./ nR

    # --- Value of Information --- #    
    input = Matrix([nL./100 nR./100 X.nB./100 ρL ρR X.gL X.first_visit]')[idxinput, :]
    voiL = m.ddc(input)[:]
    input = Matrix([nR./100 nL./100 X.nB./100 ρR ρL X.gR X.first_visit]')[idxinput, :]
    voiR = m.ddc(input)[:]

    # --- Cost of Sampling --- #    
    costL = (1 .- X.gL) .* c.κ₁
    costR = (1 .- X.gR) .* c.κ₁

    QsamplingL = (voiL .- costL)
    QsamplingR = (voiR .- costR)

    if type==:long
        (;
            voi = vcat( voiL[:], voiR[:] ),
            sampling = vcat( QsamplingL[:], QsamplingR[:] ), #vcat( Vector{Float32}(QsamplingL), Vector{Float32}(QsamplingR)),
            nA = vcat( Vector{Float32}(nL), Vector{Float32}(nR)),
            nB = vcat( Vector{Float32}(nR), Vector{Float32}(nL)),
            nC = vcat( Vector{Float32}(X.nB), Vector{Float32}(X.nB)),
            ρA = vcat( Vector{Float32}(ρL), Vector{Float32}(ρR)),
            ρB = vcat( Vector{Float32}(ρR), Vector{Float32}(ρL)), 
            gA = vcat( Vector{Float32}(X.gL), Vector{Float32}(X.gR)) .|> Bool, 
            fv = vcat( Vector{Float32}(X.first_visit), Vector{Float32}(X.first_visit) ) .|> Bool
        )
    elseif type==:wide

        (;
            voiL       = bonusL[:], 
            voiR       = bonusR[:],
            coiL       = costL[:], 
            coiR       = costR[:], 
            QsamplingL = QsamplingL[:], 
            QsamplingR = QsamplingR[:],
            nL         = nL,
            nR         = nR,
            ρL         = ρL,
            ρR         = ρR,
            gL         = X.gL .|> Bool, 
            gR         = X.gR .|> Bool, 
            fv         = X.first_visit .|> Bool
        )

    end

end;
