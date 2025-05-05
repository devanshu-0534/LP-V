#include <iostream>
#include <vector>
#include <limits>
#include <omp.h>
#include <climits>
using namespace std;

int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;
    vector<int> data(n);
    cout << "Enter elements:\n";
    for (int i = 0; i < n; ++i) cin >> data[i];

    int minVal = INT_MAX, maxVal = INT_MIN;
    long long sum = 0;

    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        if (data[i] < minVal) minVal = data[i];
        if (data[i] > maxVal) maxVal = data[i];
        sum += data[i];
    }

    double avg = static_cast<double>(sum) / n;
    cout << "Min: " << minVal << "\nMax: " << maxVal << "\nSum: " << sum << "\nAverage: " << avg << endl;

    return 0;
}
