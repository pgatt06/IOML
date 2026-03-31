using JuMP
using CPLEX
using LinearAlgebra

include("cluster.jl")

"""
Decision tree returned by the optimization models.
"""
mutable struct Tree
    D::Int64
    a::Matrix{Float64}
    b::Vector{Float64}
    c::Vector{Int64}

    function Tree()
        return new()
    end
end

"""
Create a tree directly from solver values.
"""
function Tree(D::Int64, a::Matrix{Float64}, b::Vector{Float64}, c::Vector{Int64})
    tree = Tree()
    tree.D = D
    tree.a = a
    tree.b = b
    tree.c = c
    return tree
end

"""
Create a univariate tree and recenter the split thresholds from sample flows.
"""
function Tree(D::Int64, a::Matrix{Float64}, c::Vector{Int64}, u::Matrix{Int64}, x::Matrix{Float64})
    tree = Tree()
    tree.D = D
    tree.a = a
    tree.c = c

    split_count = 2^D - 1
    leaf_count = 2^D
    sample_count = size(x, 1)
    feature_count = size(x, 2)

    upper_bounds = ones(Float64, split_count)
    lower_bounds = zeros(Float64, split_count)

    for sample_id in 1:sample_count
        node = split_count + leaf_count

        while u[sample_id, node] == 0
            node -= 1
            node == 0 && break
        end

        if node != 0
            while node > 1
                parent = fld(node, 2)
                projection = sum(a[feature_id, parent] * x[sample_id, feature_id] for feature_id in 1:feature_count)

                if node % 2 == 0
                    lower_bounds[parent] = max(lower_bounds[parent], projection)
                else
                    upper_bounds[parent] = min(upper_bounds[parent], projection)
                end

                node = parent
            end
        end
    end

    tree.b = zeros(Float64, split_count)
    for node in 1:split_count
        if c[node] == -1
            tree.b[node] = (upper_bounds[node] + lower_bounds[node]) / 2
        end
    end

    return tree
end

"""
Create a clustered univariate tree and recenter the split thresholds from cluster flows.
"""
function Tree(
    D::Int64,
    a::Matrix{Float64},
    c::Vector{Int64},
    u::Matrix{Int64},
    clusters::Vector{Cluster};
    splittableClusters::Bool = false,
)
    tree = Tree()
    tree.D = D
    tree.a = a
    tree.c = c

    split_count = 2^D - 1
    leaf_count = 2^D
    cluster_count = length(clusters)
    feature_count = size(clusters[1].x, 2)

    upper_bounds = ones(Float64, split_count)
    lower_bounds = zeros(Float64, split_count)

    for cluster_id in 1:cluster_count
        node = split_count + leaf_count

        while u[cluster_id, node] == 0
            node -= 1
            node == 0 && break
        end

        if node != 0
            while node > 1
                parent = fld(node, 2)
                projection = sum(
                    a[feature_id, parent] * clusters[cluster_id].barycenter[feature_id]
                    for feature_id in 1:feature_count
                )

                if node % 2 == 0
                    lower_bounds[parent] = max(lower_bounds[parent], projection)
                else
                    upper_bounds[parent] = min(upper_bounds[parent], projection)
                end

                node = parent
            end
        end
    end

    tree.b = zeros(Float64, split_count)
    for node in 1:split_count
        if c[node] == -1
            tree.b[node] = (upper_bounds[node] + lower_bounds[node]) / 2
        end
    end

    return tree
end

"""
Create a multivariate tree and recenter each split with a secondary max-margin model.
"""
function Tree(D::Int64, c::Vector{Int64}, u::Matrix{Int64}, s_model::Matrix{Int64}, x::Matrix{Float64})
    tree = Tree()
    tree.D = D
    tree.c = c

    split_count = 2^D - 1
    sample_count = size(x, 1)
    feature_count = size(x, 2)

    tree.a = zeros(Float64, feature_count, split_count)
    tree.b = zeros(Float64, split_count)

    for node in 1:split_count
        c[node] == -1 || continue

        left_ids = Int[]
        right_ids = Int[]

        for sample_id in 1:sample_count
            if u[sample_id, 2 * node] == 1
                push!(left_ids, sample_id)
            elseif u[sample_id, 2 * node + 1] == 1
                push!(right_ids, sample_id)
            end
        end

        left_count = length(left_ids)
        right_count = length(right_ids)

        model = Model(CPLEX.Optimizer)
        set_silent(model)

        @variable(model, a[1:feature_count], base_name = "a_j")
        @variable(model, s[1:feature_count], Bin, base_name = "s_j")
        @variable(model, b, base_name = "b")
        @variable(model, margin[1:(left_count + right_count)], base_name = "margin")
        @variable(model, min_margin, base_name = "min_margin")

        @constraint(model, -1 <= b)
        @constraint(model, b <= 1)
        @constraint(model, [feature_id in 1:feature_count], -s[feature_id] <= a[feature_id])
        @constraint(model, [feature_id in 1:feature_count], a[feature_id] <= s[feature_id])
        @constraint(
            model,
            sum(s[feature_id] for feature_id in 1:feature_count) <=
            sum(s_model[feature_id, node] for feature_id in 1:feature_count),
        )
        @constraint(model, [point_id in 1:(left_count + right_count)], margin[point_id] >= min_margin)
        @constraint(
            model,
            [point_id in 1:left_count],
            margin[point_id] == b - sum(a[feature_id] * x[left_ids[point_id], feature_id] for feature_id in 1:feature_count),
        )
        @constraint(
            model,
            [point_id in 1:right_count],
            margin[point_id + left_count] ==
            -b + sum(a[feature_id] * x[right_ids[point_id], feature_id] for feature_id in 1:feature_count),
        )

        @objective(model, Max, min_margin)
        optimize!(model)

        tree.b[node] = value(b)
        for feature_id in 1:feature_count
            tree.a[feature_id, node] = value(a[feature_id])
        end
    end

    return tree
end

"""
Create a multivariate clustered tree and recenter each split with a secondary max-margin model.
"""
function Tree(D::Int64, c::Vector{Int64}, u::Matrix{Int64}, s_model::Matrix{Int64}, clusters::Vector{Cluster})
    tree = Tree()
    tree.D = D
    tree.c = c

    split_count = 2^D - 1
    cluster_count = length(clusters)
    feature_count = length(clusters[1].lBounds)

    tree.a = zeros(Float64, feature_count, split_count)
    tree.b = zeros(Float64, split_count)

    for node in 1:split_count
        c[node] == -1 || continue

        right_data = Matrix{Float64}(undef, 0, feature_count)
        left_data = Matrix{Float64}(undef, 0, feature_count)

        for cluster_id in 1:cluster_count
            cluster_data = clusters[cluster_id].x[clusters[cluster_id].dataIds, :]

            if u[cluster_id, 2 * node] == 1
                left_data = vcat(left_data, cluster_data)
            elseif u[cluster_id, 2 * node + 1] == 1
                right_data = vcat(right_data, cluster_data)
            end
        end

        left_count = size(left_data, 1)
        right_count = size(right_data, 1)

        model = Model(CPLEX.Optimizer)
        set_silent(model)

        @variable(model, a[1:feature_count], base_name = "a_j")
        @variable(model, s[1:feature_count], Bin, base_name = "s_j")
        @variable(model, b, base_name = "b")
        @variable(model, margin[1:(left_count + right_count)], base_name = "margin")
        @variable(model, min_margin, base_name = "min_margin")

        @constraint(model, -1 <= b)
        @constraint(model, b <= 1)
        @constraint(model, [feature_id in 1:feature_count], -s[feature_id] <= a[feature_id])
        @constraint(model, [feature_id in 1:feature_count], a[feature_id] <= s[feature_id])
        @constraint(
            model,
            sum(s[feature_id] for feature_id in 1:feature_count) <=
            sum(s_model[feature_id, node] for feature_id in 1:feature_count),
        )
        @constraint(model, [point_id in 1:(left_count + right_count)], margin[point_id] >= min_margin)
        @constraint(
            model,
            [point_id in 1:left_count],
            margin[point_id] == b - sum(a[feature_id] * left_data[point_id, feature_id] for feature_id in 1:feature_count),
        )
        @constraint(
            model,
            [point_id in 1:right_count],
            margin[point_id + left_count] ==
            -b + sum(a[feature_id] * right_data[point_id, feature_id] for feature_id in 1:feature_count),
        )

        @objective(model, Max, min_margin)
        optimize!(model)

        tree.b[node] = value(b)
        for feature_id in 1:feature_count
            tree.a[feature_id, node] = value(a[feature_id])
        end
    end

    return tree
end

"""
Return the leaf reached by one sample.
"""
function leafReached(x::Vector{Float64}, tree::Tree)
    current_node = 1

    while tree.c[current_node] == -1
        if dot(tree.a[:, current_node], x) < tree.b[current_node]
            current_node = 2 * current_node
        else
            current_node = 2 * current_node + 1
        end
    end

    return current_node
end

"""
Return the leaf reached by each sample inside a cluster.
"""
function getLeavesReached(cluster::Cluster, tree::Tree)
    cluster_size = length(cluster.dataIds)
    leaves_reached = Vector{Int}(undef, cluster_size)

    for (local_id, sample_id) in enumerate(cluster.dataIds)
        leaves_reached[local_id] = leafReached(cluster.x[sample_id, :], tree)
    end

    return leaves_reached
end

"""
Split one cluster according to the leaves reached by its samples.
"""
function getSplitClusters(cluster::Cluster, tree::Tree)
    leaves_reached = getLeavesReached(cluster, tree)
    new_clusters = Cluster[]

    for local_id in 1:length(leaves_reached)
        leaves_reached[local_id] == -1 && continue

        members = findall(leaves_reached .== leaves_reached[local_id])
        push!(new_clusters, Cluster(cluster.dataIds[members], cluster.x, cluster.class))
        leaves_reached[members] .= -1
    end

    return new_clusters
end

"""
Count how many extra clusters would appear after splitting all current clusters with a tree.
"""
function getClusterSplitCount(tree::Tree, clusters::Vector)
    new_cluster_count = 0
    for cluster in clusters
        new_cluster_count += length(getSplitClusters(cluster, tree))
    end

    return new_cluster_count - length(clusters)
end
