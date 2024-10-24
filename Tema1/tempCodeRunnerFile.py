if det(A)!=0:
    print("\nDeterminant of A:")
    print(det(A))
    print("\nTrace of A:")
    print(trace(A))
    print("\nNorm of B")
    print(vector_Norm(B))
    print("\nTranspose of A:")
    print(transpose(A))
    print("\nSolution from inverse method:")
    A_inv = inverse(A)
    X = multiply_matrix_vector(A_inv, B)
    print(X)
    print("\nSolutions from Cramer's method:")
    print(Cramer(A,B)) 