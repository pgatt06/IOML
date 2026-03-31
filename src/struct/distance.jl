using Distances

"""
Store the distance between two samples.
"""
mutable struct Distance
    distance::Float64
    ids::Vector{Int}

    function Distance()
        return new()
    end
end

"""
Create the Euclidean distance between two samples.
"""
function Distance(id1::Int, id2::Int, x::Matrix{Float64})
    distance = Distance()
    # Pairwise Euclidean distances are only used to order candidate merges before the tree optimization starts.
    distance.distance = euclidean(x[id1, :], x[id2, :])
    distance.ids = [id1, id2]
    return distance
end
