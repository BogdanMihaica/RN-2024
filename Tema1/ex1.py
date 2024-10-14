import re
import os
here = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(here, 'sistem.in')


def get_data(file_name):
    with open(file_name, 'r') as file:
        equations = file.readlines()
    
    matrix = []
    constants = []
    
    for equation in equations:
        equation = equation.replace(" ", "")
        lhs, rhs = equation.split('=')
        constants.append(int(rhs))
        
        coefficients = [0, 0, 0] 
        matches = re.findall(r'([+-]?\d*)([xyz])', lhs)
        for match in matches:
            coef, var = match
            coef = int(coef) if coef and coef != '+' and coef != '-' else (1 if coef == '+' or coef == '' else -1)
            if var == 'x':
                coefficients[0] += coef
            elif var == 'y':
                coefficients[1] += coef
            elif var == 'z':
                coefficients[2] += coef
        matrix.append(coefficients)
    return matrix, constants


def det(matrix):
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    elif len(matrix) == 3:
        return (
            matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
            matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
            matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
        )

def trace(matrix):
    s=0
    for i in range(0,len(matrix[0])):
        s+=matrix[i][i]
    return s


def vector_Norm(B):
    s=0
    for x in B:
        s+=(x**2)
    return s**(1/2)


def transpose(matrix):
    transposed_matrix = []
    for i in range(len(matrix[0])): 
        new_row = []  
        for j in range(len(matrix)):  
            new_row.append(matrix[j][i])
        transposed_matrix.append(new_row)
    return transposed_matrix

def cofactor(matrix):
    cofactors = []
    for i in range(len(matrix)):
        row_cofactors = []
        for j in range(len(matrix)):
            minor = [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]
            minor_det = det(minor)
            sign = (-1) ** (i + j)
            row_cofactors.append(sign * minor_det)
        cofactors.append(row_cofactors)
    return cofactors

def adjugate(matrix):
    cofactors = cofactor(matrix)
    return transpose(cofactors)

def multiply_matrix_vector(A, B):
    C = []
    for row in A:
        result = sum(row[i] * B[i] for i in range(len(B)))
        C.append(result)
    return C

def inverse(matrix):
    determinant = det(matrix)
    if determinant == 0:
        raise ValueError("Inversa nu exista!(D=0)")
    
    adj = adjugate(matrix)
    inverse_matrix = [[adj[i][j] / determinant for j in range(len(adj))] for i in range(len(adj))]
    return inverse_matrix


def Cramer(A,B):
    Ax=[row[:] for row in A]
    Ay=[row[:] for row in A]
    Az=[row[:] for row in A]
    Ax[0][0]=B[0]
    Ax[1][0]=B[1]
    Ax[2][0]=B[2]
    Ay[0][1]=B[0]
    Ay[1][1]=B[1]
    Ay[2][1]=B[2]
    Az[0][2]=B[0]
    Az[1][2]=B[1]
    Az[2][2]=B[2]
    D=det(A)
    Dx=det(Ax)
    Dy=det(Ay)
    Dz=det(Az)
    [x,y,z]=[Dx/D,Dy/D,Dz/D]
    return [x,y,z]

A, B = get_data(filename)
    
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
    


    