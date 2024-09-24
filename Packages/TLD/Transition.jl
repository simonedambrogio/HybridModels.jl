# ----------- Define Transition ----------- #
struct Transition{State, Action}
    s::State
    a::Action
    sʼ::State
end

function Base.show(io::IO, t::Transition)

    Option = t.s.type==1 ? [" Left", "Right"] : ["Attended", "Unattended"]

    for idx in 1:2
        r, rʼ = t.s.red[idx], t.sʼ.red[idx]
        b, bʼ = (t.s.colored[idx]-r)t.s.visited[idx], (t.sʼ.colored[idx]-rʼ)t.sʼ.visited[idx]
        if (t.s.colored[idx]!=t.sʼ.colored[idx]) || (t.s.visited[idx]==0 && t.sʼ.visited[idx]==1)
            str = string("\n", Option[idx], ": (", r)
            str = string(str, rʼ-r>0 ? string("+", rʼ-r, ", ", b) : string(", ", b))
            str = string(str, bʼ-b>0 ? string("+", bʼ-b, ")") : string(")"));
        else
            str = string("\n", Option[idx], ": (", r, ", ", b, ")");
        end
        print(io, string(str, t.s.gaze==idx ? " * " : "   ") )
        print(io, " " ^ (23-length(str)), " N = ", t.s.colored[idx])
    end

end
