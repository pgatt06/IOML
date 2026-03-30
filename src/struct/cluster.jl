using Statistics

"""
Store one cluster of samples.
"""
mutable struct Cluster
    dataIds::Vector{Int}
    lBounds::Vector{Float64}
    uBounds::Vector{Float64}
    x::Matrix{Float64}
    class::Any
    barycenter::Vector{Float64}

    function Cluster()
        return new()
    end
end

"""
Create a singleton cluster from one sample.
"""
function Cluster(id::Int, x::Matrix{Float64}, y)
    cluster = Cluster()
    cluster.x = x
    cluster.class = y[id]
    cluster.dataIds = [id]
    cluster.lBounds = Vector{Float64}(x[id, :])
    cluster.uBounds = Vector{Float64}(x[id, :])
    cluster.barycenter = getBarycenter(cluster)
    return cluster
end

"""
Create a cluster from a list of sample indices.
"""
function Cluster(ids::Vector{Int}, x::Matrix{Float64}, y)
    cluster = Cluster()
    cluster.x = x
    cluster.class = y
    cluster.dataIds = copy(ids)
    cluster.lBounds = vec(minimum(x[ids, :], dims = 1))
    cluster.uBounds = vec(maximum(x[ids, :], dims = 1))
    cluster.barycenter = getBarycenter(cluster)
    return cluster
end

"""
Return the barycenter of a cluster.
"""
function getBarycenter(cluster::Cluster)
    return vec(mean(cluster.x[cluster.dataIds, :], dims = 1))
end
