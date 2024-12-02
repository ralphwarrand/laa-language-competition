import numpy as np
import time


def quicksort(lengths, vectors, low, high):
    """QuickSort implementation for sorting vectors by length."""
    if low < high:
        # Partition function
        def partition(low, high):
            pivot = lengths[high]
            i = low - 1

            for j in range(low, high):
                if lengths[j] <= pivot:
                    i += 1
                    lengths[i], lengths[j] = lengths[j], lengths[i]
                    vectors[i], vectors[j] = vectors[j], vectors[i]

            lengths[i + 1], lengths[high] = lengths[high], lengths[i + 1]
            vectors[i + 1], vectors[high] = vectors[high], vectors[i + 1]
            return i + 1

        pi = partition(low, high)  # Partition index

        quicksort(lengths, vectors, low, pi - 1)
        quicksort(lengths, vectors, pi + 1, high)


def run_python_tasks():
    print("=========================")
    print("Starting Python Native Tasks")
    print("=========================")

    # Task 1: Compute Eigenvalues of a Symmetric Matrix
    def task1():
        print("\n[Task 1] Computing Eigenvalues of a Symmetric Matrix")

        n = 1000  # Matrix size
        print(f"Generating a random symmetric {n} x {n} matrix")

        # Generate a random matrix
        A = np.random.rand(n, n)

        # Make the matrix symmetric
        A = (A + A.T) / 2

        # Start timing
        start_time = time.time()

        # Compute the eigenvalues
        eigvals = np.linalg.eigvalsh(A)

        # Stop timing
        end_time = time.time()
        print(f"Time to compute eigenvalues: {1000 * (end_time - start_time):.2f} ms")
        print(f"First few eigenvalues: {eigvals[:5]}")
        print("[Task 1 Completed]\n")

    # Task 2: Generate random vectors, compute lengths, and sort them
    def task2():
        print("\n[Task 2] Generating Random Vectors, Computing Lengths, and Sorting")

        n = 1000000  # Number of vectors
        vec_dim = 2  # Dimension of each vector
        print(f"Generating {n} random {vec_dim}-dimensional vectors")

        # Step 1: Generate random vectors
        vectors = np.random.rand(n, vec_dim)

        # Step 2: Compute lengths of the vectors
        start_compute = time.time()
        lengths = np.linalg.norm(vectors, axis=1)
        end_compute = time.time()
        print(f"Time to compute lengths: {1000 * (end_compute - start_compute):.2f} ms")

        # Step 3: Sort vectors by length using QuickSort
        start_sort = time.time()
        lengths_list = lengths.tolist()
        vectors_list = vectors.tolist()
        quicksort(lengths_list, vectors_list, 0, len(lengths_list) - 1)
        end_sort = time.time()
        print(f"Time to sort vectors by length using quicksort: {1000 * (end_sort - start_sort):.2f} ms")

        # Print the first 5 lengths for verification
        print(f"First few sorted lengths: {lengths_list[:5]}")
        print("[Task 2 Completed]\n")

    # Task 3: Compute the determinant of a matrix
    def task3():
        print("\n[Task 3] Computing Determinant of a Random Matrix")

        n = 500  # Matrix size
        print(f"Generating a random {n} x {n} matrix")

        # Generate a random matrix
        A = np.random.rand(n, n)

        # Start timing
        start_time = time.time()

        # Compute the determinant
        determinant = np.linalg.det(A)

        # Stop timing
        end_time = time.time()
        print(f"Time to compute determinant: {1000 * (end_time - start_time):.2f} ms")
        print(f"Determinant of the matrix: {determinant}")
        print("[Task 3 Completed]\n")

    # Execute all tasks
    task1()
    task2()
    task3()

    print("=========================")
    print("All Python Native Tasks Completed Successfully")
    print("=========================")


if __name__ == "__main__":
    run_python_tasks()