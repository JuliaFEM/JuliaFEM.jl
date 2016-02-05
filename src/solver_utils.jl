# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

function subscript(i)
    map(repr(i)) do c
        c   ==  '1' ? '\u2081' :
        c   ==  '2' ? '\u2082' :
        c   ==  '3' ? '\u2083' :
        c   ==  '4' ? '\u2084' :
        c   ==  '5' ? '\u2085' :
        c   ==  '6' ? '\u2086' :
        c   ==  '7' ? '\u2087' :
        c   ==  '8' ? '\u2088' :
        c   ==  '9' ? '\u2089' :
        c   ==  '0' ? '\u2080' :
        error("Unexpected character")
    end
end

function pretty_print_constraint_equation(a, b, g; char1="u", char2="λ")
    s = ""
    for (i, c) in enumerate(find(a))
        if (i != 1)
            s = s * ((a[c] < 0) ? " - " : " + ")
        end
        s = s*"$(round(abs(a[c]), 3))*$char1$(subscript(c))"
    end
    for (i, c) in enumerate(find(b))
        if (length(find(a)) != 0) || (i != 1)
            s = s * ((b[c] < 0) ? " - " : " + ")
        end
        s = s*"$(round(abs(b[c]), 3))*$char2$(subscript(c))"
    end
    s = s * " = $(round(g, 3))"
    s
end

function pretty_print_C1_row(r)
    s = "⋯ "
    for a in find(r)
        c = r[a] < 0 ? "-" : "+"
        s = s * "$c $(round(abs(r[a]), 3))*λ$(subscript(a)) "
    end
    s = s * "⋯"
    return s
end

function handle_overconstraint_error!(problem, nodes, all_dofs, C1_, C1, C2_, C2, D_, D, g_, g)
    # old, new, old, new...

    """ Return all other dofs which connects to overconstrained dofs. """
    function get_related_dofs(dofs)
        dofs_ = Set(dofs)
        for dof in copy(dofs_)
            c = find(C2[dof,:])
            length(c) != 0 && push!(dofs_, c...)
            c = find(C2_[dof,:])
            length(c) != 0 && push!(dofs_, c...)
        end
        for dof in copy(dofs_)
            c = find(C2[dof,:])
            length(c) != 0 && push!(dofs_, c...)
            c = find(C2_[dof,:])
            length(c) != 0 && push!(dofs_, c...)
        end
        for dof in copy(dofs_)
            c = find(C2[dof,:])
            length(c) != 0 && push!(dofs_, c...)
            c = find(C2_[dof,:])
            length(c) != 0 && push!(dofs_, c...)
        end
        dofs_ = sort(collect(dofs_))
        return dofs_
    end

    """ Return true if dofs has lagrange coefficients, i.e. D is nonzero. """
    function has_lagrange_coefficients(dof::Int)
        return (countnz(D[dof, :]) != 0) || (countnz(D_[dof, :]) != 0)
    end
    function has_lagrange_coefficients(dofs::Vector{Int})
        return map(has_lagrange_coefficients, dofs)
    end

    """ Test is dof single point constraint. """
    function is_spc(C, dof)
        return countnz(C[dof, :]) == 1
    end
    function is_spc(dof::Int)
        return is_spc(C2, dof) && is_spc(C2_, dof)
    end
    function is_spc(dofs::Vector{Int})
        return map(is_spc, dofs)
    end
    
    """ Algorithm 1. Calculate rank of overdetermined system and do LSQ if
        rank(C) equals to number of unique dofs.
    """
    function action1(node_id, dofs)
        dofs_ = get_related_dofs(intersect(dofs, all_dofs))
        # this will fail with dofs > 2 for some yet unknown reason
        length(dofs_) > 2 && return dofs_, false

        any(has_lagrange_coefficients(dofs_)) && return dofs_, false
        C = full([C2[dofs_, :]; C2_[dofs_, :]])
        d = full([g[dofs_]; g_[dofs_]])
        info("rank = $(rank(C)), dofs = $(length(dofs_))")
        rank(C) != length(dofs_) && return dofs_, false
        C2_[dofs_,:] = C2[dofs_,:] = 0
        g_[dofs_,:] = g[dofs_,:] = 0
        x = C \ d
        C2[dofs_, dofs_] = eye(length(dofs_))
        g[dofs_] = x
        return dofs_, true
    end

    function action2(node_id, dofs)
        """ If no coefficients on matrix D we can set essential boundary condition
            only on master side and impose bc weakly on slave side, i.e., remove SPC
        """
        dofs_ = intersect(dofs, all_dofs)
        length(dofs_) == 1 || return dofs_, false
        any(has_lagrange_coefficients(dofs_)) && return dofs_, false
        if is_spc(C1, dofs_) && !is_spc(C1_, dofs_)
            C1[dofs_,:] = C2[dofs_,:] = g[dofs_,:] = 0
            return dofs_, true
        elseif is_spc(C1_, dofs_) && !is_spc(C1, dofs_)
            C1_[dofs_,:] = C2_[dofs_,:] = g_[dofs_,:] = 0
            return dofs_, true
        else
            return dofs_, false
        end
    end

    actions = [action1, action2]

    function show_lambda_coefficients(dofs, C1)
        for dof in dofs
            status = dof in all_dofs ? " <-- overconstrained dof" : ""
            r = C1[:, dof]
            r[abs(r) .< 1.0e-9] = 0
            length(nonzeros(r)) != 0 || continue
            info("dof $dof: "*pretty_print_C1_row(r))
        end
    end

    function show_rows_in_constraint_matrix(dofs, C2, D; show_status=true)
        for dof in dofs
            status = (dof in all_dofs) && show_status ? " <-- overconstrained dof" : ""
            a = C2[dof,:]
            b = D[dof,:]
            a[abs(a) .< 1.0e-12] = 0
            b[abs(b) .< 1.0e-12] = 0
            if (length(nonzeros(a)) == 0) && (length(nonzeros(b)) == 0)
                continue
            end
            info("dof $dof: "*pretty_print_constraint_equation(a, b, g[dof])*status)
        end
    end

    function show_related_equations(dofs, C2, C2_, D, D_)
        related_dofs = Set{Int64}()
        G = [C2 D]
        G_ = [C2_ D_]
        for dof in dofs
            dof in all_dofs || continue
            c = find(G[dof,:])
            length(c) == 0 && continue
            push!(related_dofs, c...)
            c = find(G_[dof,:])
            length(c) == 0 && continue
            push!(related_dofs, c...)
        end
        related_dofs = setdiff(related_dofs, all_dofs)
        length(related_dofs) != 0 || return
        info("----- Related equations -----")
        show_rows_in_constraint_matrix(related_dofs, C2, D)
        show_rows_in_constraint_matrix(related_dofs, C2_, D_)
    end

    function print_summary(node_id, dofs)
        s = join(dofs, ", ")
        info()
        info("SUMMARY for node id $node_id with dofs $s:")
        info("----- Current constraint -----")
        info("lambda coefficients in C1 matrix are:")
        show_lambda_coefficients(dofs, C1_)
        info("rows in constraint matrix C2 & D")
        show_rows_in_constraint_matrix(dofs, C2_, D_)
        info("----- New constraint -----")
        info("lambda coefficients in C1 matrix are:")
        show_lambda_coefficients(dofs, C1)
        info("rows in constraint matrix C2 & D")
        show_rows_in_constraint_matrix(dofs, C2, D)
        show_related_equations(dofs, C2, C2_, D, D_)
    end

    info("System is overconstrained by $(length(all_dofs)) dofs.")
    info("Overconstrained_nodes: $(join(nodes, ", ")).")
    info("Following dofs already constrained: $(join(all_dofs, ", ")).")
    for node_id in nodes
        dofs = [2*(node_id-1)+1, 2*(node_id-1)+2]

        print_summary(node_id, dofs)

        # try to resolve issue automatically
        resolved = false
        for (i, action) in enumerate(actions)
            dofs, resolved = action(node_id, dofs)
            info("algorithm $i solved issue? $resolved")
            if resolved
                break
            end
        end

        if resolved
            info("fixed: new setting is")
            show_rows_in_constraint_matrix(dofs, C2, D; show_status=false)
            show_rows_in_constraint_matrix(dofs, C2_, D_; show_status=false)
            show_related_equations(dofs, C2, C2_, D, D_)
            info()
            continue
        end
           
        info("unable to resolve overconstrained situation, not continuing")
        throw("failed to resolve overconstraint situation")
        info()
    end
end

