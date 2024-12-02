// Cpp Linear Algebra
#include <armadillo>

//Python C API
#include <Python.h>

// STL
#include <chrono> 
#include <iostream>
#include <cmath>
#include <algorithm>

// QuickSort Helper Function
static void quickSort(std::vector<double>& lengths, std::vector<std::vector<double>>& vectors, int low, int high)
{
    if (low < high)
    {
        // Partition function
        auto partition = [&](int low, int high) -> int {
            double pivot = lengths[high];
            int i = low - 1;

            for (int j = low; j < high; ++j)
            {
                if (lengths[j] <= pivot)
                {
                    ++i;
                    std::swap(lengths[i], lengths[j]);
                    std::swap(vectors[i], vectors[j]);
                }
            }

            std::swap(lengths[i + 1], lengths[high]);
            std::swap(vectors[i + 1], vectors[high]);

            return i + 1;
            };

        int pi = partition(low, high); // Partition index

        quickSort(lengths, vectors, low, pi - 1);
        quickSort(lengths, vectors, pi + 1, high);
    }
}

static bool runCppNativeTasks()
{
    std::cout << "=========================" << std::endl;
    std::cout << "Starting C++ Native Tasks" << std::endl;
    std::cout << "=========================" << std::endl;

    // Task 1
    {
        std::cout << "\n[Task 1] Computing Eigenvalues of a Symmetric Matrix\n";

        // Define the size of the matrix
        constexpr int n = 1000; // Adjust 'n' for desired matrix size
        std::cout << "Generating a random symmetric " << n << " x " << n << " matrix\n";

        if (!arma::arma_config::lapack)
        {
            std::cout << "LAPACK is NOT enabled!\n";
            return false;
        }

        // Generate a random matrix
        arma::mat A = arma::randu<arma::mat>(n, n);

        // Ensure the matrix is symmetric
        A = (A + A.t()) / 2;

        // Variable to store eigenvalues
        arma::vec eigval;

        // Start timing
        auto start = std::chrono::high_resolution_clock::now();

        // Compute the eigenvalues
        arma::eig_sym(eigval, A);

        // Stop timing
        auto stop = std::chrono::high_resolution_clock::now();

        // Calculate elapsed time
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Time to compute eigenvalues: " << duration.count() << " ms" << std::endl;

        // Print the first few eigenvalues for verification
        std::cout << "First few eigenvalues: " << eigval.head(5).t();

        std::cout << "[Task 1] Completed\n";
    }

    // Task 2
    {
        std::cout << "\n[Task 2] Generating Random Vectors, Computing Lengths, and Sorting" << std::endl;

        constexpr int n = 1000000; // Number of vectors
        constexpr int vec_dim = 2; // Dimension of each vector

        std::cout << "Generating " << n << " random " << vec_dim << "-dimensional vectors" << std::endl;

        // Step 1: Generate random vectors
        std::vector<std::vector<double>> vectors(n, std::vector<double>(vec_dim));
        for (auto& vec : vectors)
        {
            for (auto& element : vec)
            {
                element = static_cast<double>(rand()) / RAND_MAX; // Generate random double [0, 1]
            }
        }

        // Step 2: Compute lengths of the vectors
        std::vector<double> lengths(n);

        auto start_compute = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n; ++i)
        {
            double sum = 0.0;
            for (const auto& element : vectors[i])
            {
                sum += element * element; // Sum of squares
            }
            lengths[i] = std::sqrt(sum); // Length (Euclidean norm)
        }

        auto stop_compute = std::chrono::high_resolution_clock::now();
        auto duration_compute = std::chrono::duration_cast<std::chrono::milliseconds>(stop_compute - start_compute);
        std::cout << "Time to compute lengths: " << duration_compute.count() << " ms" << std::endl;

        // Step 3: Sort vectors by length using QuickSort
        auto start_sort = std::chrono::high_resolution_clock::now();

        quickSort(lengths, vectors, 0, n - 1);

        auto stop_sort = std::chrono::high_resolution_clock::now();
        auto duration_sort = std::chrono::duration_cast<std::chrono::milliseconds>(stop_sort - start_sort);
        std::cout << "Time to sort vectors by length using quicksort: " << duration_sort.count() << " ms" << std::endl;

        // Print the first 5 lengths for verification
        std::cout << "First few sorted lengths: ";
        for (int i = 0; i < 5 && i < n; ++i)
        {
            std::cout << lengths[i] << " ";
        }
       
        std::cout << std::endl;

        std::cout << "[Task 2] Completed\n" << std::endl;
    }

    // Task 3
    {
        std::cout << "[Task 3] Computing Determinant of a Random Matrix" << std::endl;

        constexpr int m = 500; // Define the size of the matrix

        std::cout << "Generating a random " << m << " x " << m << " matrix\n";

        // Generate a random matrix
        arma::mat B = arma::randu<arma::mat>(m, m);

        std::cout << "Matrix sample values:\n" << B.submat(0, 0, 4, 4) << std::endl;

        double cond_num = arma::cond(B);
        std::cout << "Condition number of the matrix: " << cond_num << std::endl;

        // Start timing
        auto start = std::chrono::high_resolution_clock::now();

        // Compute the determinant
        double determinant = arma::det(B);

        // Stop timing
        auto stop = std::chrono::high_resolution_clock::now();

        // Calculate elapsed time
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Time to compute determinant: " << duration.count() << " ms" << std::endl;

        // Print the determinant value
        std::cout << "Determinant of the matrix: " << determinant << std::endl;

        std::cout << "[Task 3] Completed\n" << std::endl;
    }

    std::cout << "===========================================" << std::endl;
    std::cout << "All C++ Native Tasks Completed Successfully" << std::endl;
    std::cout << "===========================================\n" << std::endl;

    return true;
}

static bool runPythonEmbeddedTasks()
{
    //ugly but it works
    _putenv("PYTHONHOME=D:\\Projects\\laa-language-competition");
    _putenv("PYTHONPATH=D:\\Projects\\laa-language-competition\\thirdparty\\python\\lib");

    Py_Initialize();
    
    // Open the Python script file
    std::string scriptPath = "main.py";
    FILE* file = nullptr;
    errno_t err = fopen_s(&file, scriptPath.c_str(), "r");
    if (err != 0 || file == nullptr) {
        std::cerr << "Could not open script: " << scriptPath << std::endl;
        Py_Finalize();
        return false;
    }

    // Execute the script
    std::cout << "Running Embedded Python script: " << scriptPath << std::endl;
    PyRun_SimpleFile(file, scriptPath.c_str());

    Py_Finalize();
    return true;
}


int main()
{
    try
    {
        if (!runCppNativeTasks())
            return EXIT_FAILURE;

        runPythonEmbeddedTasks();
    }
    catch (std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}