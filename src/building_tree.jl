include("struct/tree.jl")

"""
Build a decision tree by solving formulation F on individual samples.
"""
function build_tree(
    x::Matrix{Float64},
    y::AbstractVector,
    D::Int64,
    classes;
    multivariate::Bool = false,
    time_limit::Int64 = -1,
    mu::Float64 = 10.0^(-4),
)
    data_count = length(y)
    feature_count = size(x, 2)
    class_count = length(classes)
    split_count = 2^D - 1
    leaf_count = 2^D

    model = Model(CPLEX.Optimizer)
    set_silent(model)

    if time_limit != -1
        set_time_limit_sec(model, time_limit)
    end

    mu_min = 1.0
    mu_max = 0.0

    if !multivariate
        # In the univariate model, CM1 uses a strict left/right separation.
        # The smallest positive spacing on each feature provides that margin in the Big-M constraints.
        mu_vector = ones(Float64, feature_count)
        for feature_id in 1:feature_count
            for first_id in 1:data_count
                for second_id in (first_id + 1):data_count
                    if abs(x[first_id, feature_id] - x[second_id, feature_id]) > 1e-4
                        mu_vector[feature_id] = min(
                            mu_vector[feature_id],
                            abs(x[first_id, feature_id] - x[second_id, feature_id]),
                        )
                    end
                end
            end
            mu_min = min(mu_min, mu_vector[feature_id])
            mu_max = max(mu_max, mu_vector[feature_id])
        end
    end

    if multivariate
        @variable(model, a[1:feature_count, 1:split_count], base_name = "a")
        @variable(model, a_abs[1:feature_count, 1:split_count], base_name = "a_abs")
        @variable(model, s[1:feature_count, 1:split_count], Bin, base_name = "s")
        @variable(model, d[1:split_count], Bin, base_name = "d")
    else
        @variable(model, a[1:feature_count, 1:split_count], Bin, base_name = "a")
    end
    @variable(model, b[1:split_count], base_name = "b")
    @variable(model, c[1:class_count, 1:(split_count + leaf_count)], Bin, base_name = "c")
    @variable(model, u_at[1:data_count, 1:(split_count + leaf_count)], Bin, base_name = "u_at")
    @variable(model, u_tw[1:data_count, 1:(split_count + leaf_count)], Bin, base_name = "u_tw")

    if multivariate
        @constraint(model, [node in 1:split_count], d[node] + sum(c[k, node] for k in 1:class_count) == 1)
        @constraint(model, [node in 1:split_count], b[node] <= d[node])
        @constraint(model, [node in 1:split_count], b[node] >= -d[node])
        @constraint(model, [node in 1:split_count], sum(a_abs[feature_id, node] for feature_id in 1:feature_count) <= d[node])
        @constraint(model, [node in 1:split_count, feature_id in 1:feature_count], a[feature_id, node] <= a_abs[feature_id, node])
        @constraint(model, [node in 1:split_count, feature_id in 1:feature_count], a[feature_id, node] >= -a_abs[feature_id, node])
        @constraint(model, [node in 1:split_count, feature_id in 1:feature_count], a[feature_id, node] <= s[feature_id, node])
        @constraint(model, [node in 1:split_count, feature_id in 1:feature_count], a[feature_id, node] >= -s[feature_id, node])
        @constraint(model, [node in 1:split_count, feature_id in 1:feature_count], s[feature_id, node] <= d[node])
        @constraint(model, [node in 1:split_count], sum(s[feature_id, node] for feature_id in 1:feature_count) >= d[node])
        @constraint(model, [node in 2:split_count], d[node] <= d[fld(node, 2)])
    else
        @constraint(model, [node in 1:split_count], sum(a[feature_id, node] for feature_id in 1:feature_count) + sum(c[k, node] for k in 1:class_count) == 1)
        @constraint(model, [node in 1:split_count], b[node] <= sum(a[feature_id, node] for feature_id in 1:feature_count))
        @constraint(model, [node in 1:split_count], b[node] >= 0)
    end

    @constraint(model, [node in (split_count + 1):(split_count + leaf_count)], sum(c[k, node] for k in 1:class_count) == 1)
    # A sample reaching a node either stops there with its true class or keeps flowing to one child.
    @constraint(model, [sample_id in 1:data_count, node in 1:split_count], u_at[sample_id, node] == u_at[sample_id, 2 * node] + u_at[sample_id, 2 * node + 1] + u_tw[sample_id, node])
    @constraint(model, [sample_id in 1:data_count, node in (split_count + 1):(split_count + leaf_count)], u_at[sample_id, node] == u_tw[sample_id, node])
    @constraint(model, [sample_id in 1:data_count, node in 1:(split_count + leaf_count)], u_tw[sample_id, node] <= c[findfirst(classes .== y[sample_id]), node])

    if multivariate
        @constraint(model, [sample_id in 1:data_count, node in 1:split_count], sum(a[feature_id, node] * x[sample_id, feature_id] for feature_id in 1:feature_count) + mu <= b[node] + (2 + mu) * (1 - u_at[sample_id, 2 * node]))
        @constraint(model, [sample_id in 1:data_count, node in 1:split_count], sum(a[feature_id, node] * x[sample_id, feature_id] for feature_id in 1:feature_count) >= b[node] - 2 * (1 - u_at[sample_id, 2 * node + 1]))
        @constraint(model, [sample_id in 1:data_count, node in 1:split_count], u_at[sample_id, 2 * node + 1] <= d[node])
    else
        @constraint(model, [sample_id in 1:data_count, node in 1:split_count], sum(a[feature_id, node] * (x[sample_id, feature_id] + mu_vector[feature_id] - mu_min) for feature_id in 1:feature_count) + mu_min <= b[node] + (1 + mu_max) * (1 - u_at[sample_id, 2 * node]))
        @constraint(model, [sample_id in 1:data_count, node in 1:split_count], sum(a[feature_id, node] * x[sample_id, feature_id] for feature_id in 1:feature_count) >= b[node] - (1 - u_at[sample_id, 2 * node + 1]))
        @constraint(model, [sample_id in 1:data_count, node in 1:split_count], u_at[sample_id, 2 * node + 1] <= sum(a[feature_id, node] for feature_id in 1:feature_count))
        @constraint(model, [sample_id in 1:data_count, node in 1:split_count], u_at[sample_id, 2 * node] <= sum(a[feature_id, node] for feature_id in 1:feature_count))
    end

    @objective(model, Max, sum(u_at[sample_id, 1] for sample_id in 1:data_count))

    start_time = time()
    optimize!(model)
    resolution_time = time() - start_time

    gap = -1.0
    tree = nothing
    objective = -1

    if primal_status(model) == MOI.FEASIBLE_POINT
        class_prediction = Vector{Int64}(undef, split_count + leaf_count)
        for node in 1:(split_count + leaf_count)
            class_id = argmax(value.(c[:, node]))
            class_prediction[node] = value(c[class_id, node]) >= 1.0 - 1e-4 ? class_id : -1
        end

        objective = objective_value(model)
        if termination_status(model) == MOI.OPTIMAL
            gap = 0.0
        else
            bound = objective_bound(model)
            gap = 100.0 * abs(objective - bound) / (objective + 1e-4)
        end

        if multivariate
            tree = Tree(D, class_prediction, round.(Int, value.(u_at)), round.(Int, value.(s)), x)
        else
            tree = Tree(D, value.(a), class_prediction, round.(Int, value.(u_at)), x)
        end
    end

    return tree, objective, resolution_time, gap
end

"""
Build a decision tree by solving one of the clustered formulations.
"""
function build_tree(
    clusters::Vector{Cluster},
    D::Int64,
    classes;
    multivariate::Bool = false,
    time_limit::Int64 = -1,
    mu::Float64 = 10.0^(-4),
    useFhS::Bool = false,
    useFeS::Bool = false,
)
    data_count = sum(length(cluster.dataIds) for cluster in clusters)
    cluster_count = length(clusters)
    feature_count = size(clusters[1].x, 2)
    class_count = length(classes)
    split_count = 2^D - 1
    leaf_count = 2^D

    model = Model(CPLEX.Optimizer)
    set_silent(model)

    if time_limit != -1
        set_time_limit_sec(model, time_limit)
    end

    mu_min = 1.0
    mu_max = 0.0

    if !multivariate
        mu_vector = ones(Float64, feature_count)
        for feature_id in 1:feature_count
            for first_id in 1:cluster_count
                for second_id in (first_id + 1):cluster_count
                    if useFhS || useFeS
                        if abs(clusters[first_id].barycenter[feature_id] - clusters[second_id].barycenter[feature_id]) > 1e-4
                            mu_vector[feature_id] = min(
                                mu_vector[feature_id],
                                abs(clusters[first_id].barycenter[feature_id] - clusters[second_id].barycenter[feature_id]),
                            )
                        end
                    else
                        left_gap = clusters[first_id].lBounds[feature_id] - clusters[second_id].uBounds[feature_id]
                        right_gap = clusters[second_id].lBounds[feature_id] - clusters[first_id].uBounds[feature_id]
                        if left_gap > 1e-4 || right_gap > 1e-4
                            mu_vector[feature_id] = min(mu_vector[feature_id], min(abs(left_gap), abs(right_gap)))
                        end
                    end
                end
            end
            mu_min = min(mu_min, mu_vector[feature_id])
            mu_max = max(mu_max, mu_vector[feature_id])
        end
    end

    if multivariate
        @variable(model, a[1:feature_count, 1:split_count], base_name = "a")
        @variable(model, a_abs[1:feature_count, 1:split_count], base_name = "a_abs")
        @variable(model, s[1:feature_count, 1:split_count], Bin, base_name = "s")
        @variable(model, d[1:split_count], Bin, base_name = "d")
    else
        @variable(model, a[1:feature_count, 1:split_count], Bin, base_name = "a")
    end
    @variable(model, b[1:split_count], base_name = "b")
    @variable(model, c[1:class_count, 1:(split_count + leaf_count)], Bin, base_name = "c")
    @variable(model, u_at[1:cluster_count, 1:(split_count + leaf_count)], Bin, base_name = "u_at")
    @variable(model, u_tw[1:cluster_count, 1:(split_count + leaf_count)], Bin, base_name = "u_tw")

    if useFeS
        @variable(model, r[1:data_count], Bin, base_name = "r")
        @constraint(model, [cluster_id in 1:cluster_count], sum(r[sample_id] for sample_id in clusters[cluster_id].dataIds) == 1)
    end

    if multivariate
        @constraint(model, [node in 1:split_count], d[node] + sum(c[k, node] for k in 1:class_count) == 1)
        @constraint(model, [node in 1:split_count], b[node] <= d[node])
        @constraint(model, [node in 1:split_count], b[node] >= -d[node])
        @constraint(model, [node in 1:split_count], sum(a_abs[feature_id, node] for feature_id in 1:feature_count) <= d[node])
        @constraint(model, [node in 1:split_count, feature_id in 1:feature_count], a[feature_id, node] <= a_abs[feature_id, node])
        @constraint(model, [node in 1:split_count, feature_id in 1:feature_count], a[feature_id, node] >= -a_abs[feature_id, node])
        @constraint(model, [node in 1:split_count, feature_id in 1:feature_count], a[feature_id, node] <= s[feature_id, node])
        @constraint(model, [node in 1:split_count, feature_id in 1:feature_count], a[feature_id, node] >= -s[feature_id, node])
        @constraint(model, [node in 1:split_count, feature_id in 1:feature_count], s[feature_id, node] <= d[node])
        @constraint(model, [node in 1:split_count], sum(s[feature_id, node] for feature_id in 1:feature_count) >= d[node])
        @constraint(model, [node in 2:split_count], d[node] <= d[fld(node, 2)])
    else
        @constraint(model, [node in 1:split_count], sum(a[feature_id, node] for feature_id in 1:feature_count) + sum(c[k, node] for k in 1:class_count) == 1)
        @constraint(model, [node in 1:split_count], b[node] <= sum(a[feature_id, node] for feature_id in 1:feature_count))
        @constraint(model, [node in 1:split_count], b[node] >= 0)
    end

    @constraint(model, [node in (split_count + 1):(split_count + leaf_count)], sum(c[k, node] for k in 1:class_count) == 1)
    @constraint(model, [cluster_id in 1:cluster_count, node in 1:split_count], u_at[cluster_id, node] == u_at[cluster_id, 2 * node] + u_at[cluster_id, 2 * node + 1] + u_tw[cluster_id, node])
    @constraint(model, [cluster_id in 1:cluster_count, node in (split_count + 1):(split_count + leaf_count)], u_at[cluster_id, node] == u_tw[cluster_id, node])
    @constraint(model, [cluster_id in 1:cluster_count, node in 1:(split_count + leaf_count)], u_tw[cluster_id, node] <= c[findfirst(classes .== clusters[cluster_id].class), node])

    if multivariate
        if useFhS
            # FhS replaces each cluster by its barycenter, while FU keeps the whole cluster box.
            @constraint(model, [cluster_id in 1:cluster_count, node in 1:split_count], sum(a[feature_id, node] * clusters[cluster_id].barycenter[feature_id] for feature_id in 1:feature_count) + mu <= b[node] + (2 + mu) * (1 - u_at[cluster_id, 2 * node]))
            @constraint(model, [cluster_id in 1:cluster_count, node in 1:split_count], sum(a[feature_id, node] * clusters[cluster_id].barycenter[feature_id] for feature_id in 1:feature_count) >= b[node] - 2 * (1 - u_at[cluster_id, 2 * node + 1]))
        elseif useFeS
            # FeS is stricter: one original sample is selected inside each cluster and must satisfy the split.
            @constraint(model, [(cluster_id, cluster) in enumerate(clusters), sample_id in cluster.dataIds, node in 1:split_count], sum(a[feature_id, node] * cluster.x[sample_id, feature_id] for feature_id in 1:feature_count) + mu <= b[node] + (2 + mu) * (2 - u_at[cluster_id, 2 * node] - r[sample_id]))
            @constraint(model, [(cluster_id, cluster) in enumerate(clusters), sample_id in cluster.dataIds, node in 1:split_count], sum(a[feature_id, node] * cluster.x[sample_id, feature_id] for feature_id in 1:feature_count) >= b[node] - 2 * (1 - u_at[cluster_id, 2 * node + 1]))
        else
            @constraint(model, [cluster_id in 1:cluster_count, node in 1:split_count, sample_id in clusters[cluster_id].dataIds], sum(a[feature_id, node] * clusters[cluster_id].x[sample_id, feature_id] for feature_id in 1:feature_count) + mu <= b[node] + (2 + mu) * (1 - u_at[cluster_id, 2 * node]))
            @constraint(model, [cluster_id in 1:cluster_count, node in 1:split_count, sample_id in clusters[cluster_id].dataIds], sum(a[feature_id, node] * clusters[cluster_id].x[sample_id, feature_id] for feature_id in 1:feature_count) >= b[node] - 2 * (1 - u_at[cluster_id, 2 * node + 1]))
        end
        @constraint(model, [cluster_id in 1:cluster_count, node in 1:split_count], u_at[cluster_id, 2 * node + 1] <= d[node])
    else
        if useFhS
            @constraint(model, [cluster_id in 1:cluster_count, node in 1:split_count], sum(a[feature_id, node] * (clusters[cluster_id].barycenter[feature_id] + mu_vector[feature_id] - mu_min) for feature_id in 1:feature_count) + mu_min <= b[node] + (1 + mu_max) * (1 - u_at[cluster_id, 2 * node]))
            @constraint(model, [cluster_id in 1:cluster_count, node in 1:split_count], sum(a[feature_id, node] * clusters[cluster_id].barycenter[feature_id] for feature_id in 1:feature_count) >= b[node] - (1 - u_at[cluster_id, 2 * node + 1]))
        elseif useFeS
            @constraint(model, [(cluster_id, cluster) in enumerate(clusters), sample_id in cluster.dataIds, node in 1:split_count], sum(a[feature_id, node] * (cluster.x[sample_id, feature_id] + mu_vector[feature_id] - mu_min) for feature_id in 1:feature_count) + mu_min <= b[node] + (1 + mu_max) * (2 - u_at[cluster_id, 2 * node] - r[sample_id]))
            @constraint(model, [(cluster_id, cluster) in enumerate(clusters), sample_id in cluster.dataIds, node in 1:split_count], sum(a[feature_id, node] * cluster.x[sample_id, feature_id] for feature_id in 1:feature_count) >= b[node] - (2 - u_at[cluster_id, 2 * node + 1] - r[sample_id]))
        else
            @constraint(model, [cluster_id in 1:cluster_count, node in 1:split_count], sum(a[feature_id, node] * (clusters[cluster_id].uBounds[feature_id] + mu_vector[feature_id] - mu_min) for feature_id in 1:feature_count) + mu_min <= b[node] + (1 + mu_max) * (1 - u_at[cluster_id, 2 * node]))
            @constraint(model, [cluster_id in 1:cluster_count, node in 1:split_count], sum(a[feature_id, node] * clusters[cluster_id].lBounds[feature_id] for feature_id in 1:feature_count) >= b[node] - (1 - u_at[cluster_id, 2 * node + 1]))
        end
        @constraint(model, [cluster_id in 1:cluster_count, node in 1:split_count], u_at[cluster_id, 2 * node + 1] <= sum(a[feature_id, node] for feature_id in 1:feature_count))
    end

    @objective(model, Max, sum(length(clusters[cluster_id].dataIds) * u_at[cluster_id, 1] for cluster_id in 1:cluster_count))

    start_time = time()
    optimize!(model)
    resolution_time = time() - start_time

    gap = -1.0
    tree = nothing
    objective = -1

    if primal_status(model) == MOI.FEASIBLE_POINT
        class_prediction = Vector{Int64}(undef, split_count + leaf_count)
        for node in 1:(split_count + leaf_count)
            class_id = argmax(value.(c[:, node]))
            class_prediction[node] = value(c[class_id, node]) >= 1.0 - 1e-4 ? class_id : -1
        end

        objective = objective_value(model)
        if termination_status(model) == MOI.OPTIMAL
            gap = 0.0
        else
            bound = objective_bound(model)
            gap = 100.0 * abs(objective - bound) / (objective + 1e-4)
        end

        if multivariate
            tree = Tree(D, class_prediction, round.(Int, value.(u_at)), round.(Int, value.(s)), clusters)
        else
            tree = Tree(D, value.(a), class_prediction, round.(Int, value.(u_at)), clusters)
        end
    end

    return tree, objective, resolution_time, gap
end

"""
Build a tree by repeatedly solving the clustered formulations and splitting clusters when needed.
"""
function iteratively_build_tree(
    clusters::Vector{Cluster},
    D::Int64,
    x::Matrix{Float64},
    y::AbstractVector,
    classes::AbstractVector;
    multivariate::Bool = false,
    time_limit::Int64 = -1,
    mu::Float64 = 10.0^(-4),
    isExact::Bool = false,
    shiftSeparations::Bool = false,
)
    start_time = time()

    last_objective = nothing
    last_feasible_tree = nothing
    gap = nothing

    cluster_split = true
    iteration_count = 0
    useFhS = !isExact
    useFeS = isExact

    while cluster_split
        iteration_count += 1

        tree, objective, solve_time, gap = build_tree(
            clusters,
            D,
            classes;
            multivariate = multivariate,
            time_limit = time_limit,
            mu = mu,
            useFhS = useFhS,
            useFeS = useFeS,
        )

        objective == -1 && break

        if shiftSeparations
            tree = naivelyShiftSeparations(tree, x, y, classes, clusters)
        end

        # The iterative scheme only refines clusters that are cut by the current tree.
        new_clusters = Cluster[]
        for cluster in clusters
            append!(new_clusters, getSplitClusters(cluster, tree))
        end

        if length(clusters) == length(new_clusters)
            cluster_split = false
        else
            clusters = new_clusters
        end

        last_feasible_tree = tree
        last_objective = objective
    end

    total_time = time() - start_time
    return last_feasible_tree, last_objective, total_time, gap, iteration_count
end
