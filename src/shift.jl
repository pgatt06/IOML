"""
Try simple threshold shifts on each univariate split and keep the best training error.
"""
function naivelyShiftSeparations(
    tree::Tree,
    x::Matrix{Float64},
    y::AbstractVector,
    classes::AbstractVector,
    clusters::Vector{Cluster},
)
    shifted_tree = Tree(tree.D, copy(tree.a), copy(tree.b), copy(tree.c))

    for node in eachindex(shifted_tree.b)
        shifted_tree.c[node] == -1 || continue

        best_errors = prediction_errors(shifted_tree, x, y, classes)
        best_threshold = shifted_tree.b[node]

        for candidate_threshold in 0.0:0.1:1.0
            shifted_tree.b[node] = candidate_threshold
            current_errors = prediction_errors(shifted_tree, x, y, classes)

            if current_errors < best_errors
                best_errors = current_errors
                best_threshold = candidate_threshold
            end
        end

        shifted_tree.b[node] = best_threshold
    end

    return shifted_tree
end
