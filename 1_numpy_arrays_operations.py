import numpy as np

# 1. Create arrays
print("1. Creating arrays:")
a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2, 3], [4, 5, 6]])
c = np.zeros((2, 3))
d = np.ones((3, 2))
e = np.arange(0, 10, 2)

print("a:", a)
print("b:\n", b)
print("c:\n", c)
print("d:\n", d)
print("e:", e)

# 2. Array operations
print("\n2. Array operations:")
f = a + 2
g = b * 3
h = np.dot(b, d)

print("f (a + 2):", f)
print("g (b * 3):\n", g)
print("h (b dot d):\n", h)

# 3. Mathematical functions
print("\n3. Mathematical functions:")
i = np.sin(a)
j = np.exp(a)
k = np.sqrt(a)

print("i (sin(a)):", i)
print("j (exp(a)):", j)
print("k (sqrt(a)):", k)

# 4. Statistical operations
print("\n4. Statistical operations:")
print("Mean of a:", np.mean(a))
print("Standard deviation of a:", np.std(a))
print("Max of b:", np.max(b))
print("Min of b:", np.min(b))

# 5. Reshaping and transposing
print("\n5. Reshaping and transposing:")
l = np.reshape(a, (5, 1))
m = b.T

print("l (a reshaped):\n", l)
print("m (b transposed):\n", m)

# 6. Boolean indexing
print("\n6. Boolean indexing:")
n = a[a > 3]
print("n (elements of a > 3):", n)

# 7. Basic linear algebra
print("\n7. Basic linear algebra:")
o = np.linalg.inv(np.array([[1, 2], [3, 4]]))
p = np.linalg.det(np.array([[1, 2], [3, 4]]))

print("o (inverse of [[1, 2], [3, 4]]):\n", o)
print("p (determinant of [[1, 2], [3, 4]]):", p)