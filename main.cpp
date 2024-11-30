#include <iostream>
#include <armadillo>
#include <chrono> // For time measurement

int main() {
    // Define the size of the matrix
    const int n = 1000; // Adjust 'n' for desired matrix size

    if (arma::arma_config::lapack) {
        std::cout << "LAPACK is enabled!" << std::endl;
    }
    else {
        std::cout << "LAPACK is NOT enabled!" << std::endl;
    }

    try {
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
        std::cout << "First few eigenvalues: " << eigval.head(5).t() << std::endl;
    }
    catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}