function findall_ids(f::Function, A)
    ids = findall(f, A)
    [[I[1], I[2]] for I ∈ ids]
end