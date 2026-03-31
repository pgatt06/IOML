include("struct/distance.jl")

function initialize_clusters(x::Matrix{Float64}, y)
    return [Cluster(sample_id, x, y) for sample_id in 1:size(x, 1)]
end

function pairwise_same_class_distances(x::Matrix{Float64}, y)
    sample_count = length(y)
    distances = Distance[]

    for first_id in 1:(sample_count - 1)
        for second_id in (first_id + 1):sample_count
            y[first_id] == y[second_id] || continue
            push!(distances, Distance(first_id, second_id, x))
        end
    end

    sort!(distances, by = distance -> distance.distance)
    return distances
end

"""
Merge samples while keeping the exact H1 hypothesis.
"""
function exactMerge(x::Matrix{Float64}, y)
    sample_count = length(y)
    clusters = initialize_clusters(x, y)
    cluster_ids = collect(1:sample_count)

    for distance in pairwise_same_class_distances(x, y)
        first_cluster_id = cluster_ids[distance.ids[1]]
        second_cluster_id = cluster_ids[distance.ids[2]]
        first_cluster_id == second_cluster_id && continue

        first_cluster = clusters[first_cluster_id]
        second_cluster = clusters[second_cluster_id]

        if canMerge(first_cluster, second_cluster, x, y)
            merge!(first_cluster, second_cluster)
            for sample_id in second_cluster.dataIds
                cluster_ids[sample_id] = first_cluster_id
            end
            empty!(clusters[second_cluster_id].dataIds)
        end
    end

    return filter(cluster -> !isempty(cluster.dataIds), clusters)
end

"""
Merge the closest samples of the same class until the requested cluster ratio is reached.
"""
function simpleMerge(x::Matrix{Float64}, y, gamma)
    sample_count = length(y)
    clusters = initialize_clusters(x, y)
    cluster_ids = collect(1:sample_count)
    distances = pairwise_same_class_distances(x, y)

    remaining_clusters = sample_count
    distance_id = 1

    while distance_id <= length(distances) && remaining_clusters > sample_count * gamma
        distance = distances[distance_id]
        first_cluster_id = cluster_ids[distance.ids[1]]
        second_cluster_id = cluster_ids[distance.ids[2]]

        if first_cluster_id != second_cluster_id
            remaining_clusters -= 1
            first_cluster = clusters[first_cluster_id]
            second_cluster = clusters[second_cluster_id]
            merge!(first_cluster, second_cluster)

            for sample_id in second_cluster.dataIds
                cluster_ids[sample_id] = first_cluster_id
            end

            empty!(clusters[second_cluster_id].dataIds)
        end

        distance_id += 1
    end

    return filter(cluster -> !isempty(cluster.dataIds), clusters)
end

"""
Check whether two clusters can be merged without violating the exact clustering hypothesis.
"""
function canMerge(first_cluster::Cluster, second_cluster::Cluster, x::Matrix{Float64}, y::Vector{Int})
    merged_lower_bounds = min.(first_cluster.lBounds, second_cluster.lBounds)
    merged_upper_bounds = max.(first_cluster.uBounds, second_cluster.uBounds)

    sample_id = 1
    merge_is_valid = true

    while sample_id <= size(x, 1) && merge_is_valid
        sample = x[sample_id, :]

        if !(sample_id in first_cluster.dataIds) &&
           !(sample_id in second_cluster.dataIds) &&
           isInABound(sample, merged_lower_bounds, merged_upper_bounds)
            merge_is_valid = false
        end

        sample_id += 1
    end

    return merge_is_valid
end

"""
Return true when a sample intersects at least one feature interval.
This matches the H1 condition used in the course slides.
"""
function isInABound(v::Vector{Float64}, lower_bounds::Vector{Float64}, upper_bounds::Vector{Float64})
    feature_id = 1
    intersects_bounds = false

    while !intersects_bounds && feature_id <= length(v)
        if lower_bounds[feature_id] <= v[feature_id] <= upper_bounds[feature_id]
            intersects_bounds = true
        end
        feature_id += 1
    end

    return intersects_bounds
end

"""
Merge `second_cluster` into `first_cluster`.
"""
function merge!(first_cluster::Cluster, second_cluster::Cluster)
    append!(first_cluster.dataIds, second_cluster.dataIds)
    first_cluster.lBounds = min.(first_cluster.lBounds, second_cluster.lBounds)
    first_cluster.uBounds = max.(first_cluster.uBounds, second_cluster.uBounds)
    first_cluster.barycenter = getBarycenter(first_cluster)
end
