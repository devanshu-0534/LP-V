#include <iostream>
#include <vector>
#include <stack>
#include <omp.h>
using namespace std;

void parallelDFS(vector<vector<int>>& graph, int start) {
    int n = graph.size();
    vector<bool> visited(n, false);
    stack<int> s;
    vector<int> traversalOrder;  // New: To store the order of traversal
    s.push(start);

    while (!s.empty()) {
        int u;
        #pragma omp critical
        {
            u = s.top(); s.pop();
        }

        if (!visited[u]) {
            visited[u] = true;
            cout << "Visited: " << u << endl;
            traversalOrder.push_back(u);  // Record visit

            #pragma omp parallel for
            for (int i = 0; i < graph[u].size(); ++i) {
                int v = graph[u][i];
                if (!visited[v]) {
                    #pragma omp critical
                    s.push(v);
                }
            }
        }
    }

    // New: Print traversal order
    cout << "Traversal Order: ";
    for (int node : traversalOrder) {
        cout << node << " ";
    }
    cout << endl;
}

int main() {
    int n, e;
    cout << "Enter number of nodes and edges: ";
    cin >> n >> e;
    vector<vector<int>> graph(n);
    cout << "Enter edges (u v):\n";
    for (int i = 0; i < e; ++i) {
        int u, v;
        cin >> u >> v;
        graph[u].push_back(v);
        graph[v].push_back(u);
    }

    int start;
    cout << "Enter starting node: ";
    cin >> start;
    parallelDFS(graph, start);
    return 0;
}
