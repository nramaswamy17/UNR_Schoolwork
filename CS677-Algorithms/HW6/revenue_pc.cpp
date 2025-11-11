#include <iostream>
#include <vector>
#include <algorithm>
#include <stack>
#include <map>

using namespace std;

map<pair<int, char>, char> parent; // for path reconstruction

double findOptimal(int week, char city, int n , double T,
    vector<double>& RA, vector<double>& RB,
    vector<double>& SA, vector<double>& SB) {
    
    // Base case
    if (week == 0) {
        if (city == 'A') {
            return RA[0] - SA[0];
        } else {
            return RB[0] - SB[0];
        }
    }

    // Recursive
    double stay, change;
    if (city == 'A') {
        stay = findOptimal(week-1, 'A', n, T, RA, RB, SA, SB) + RA[week] - SA[week];
        change = findOptimal(week-1, 'B', n, T, RA, RB, SA, SB) + RA[week] - SA[week] - T;
        parent[{week, city}] = (stay >= change) ? 'A' : 'B';
        return max(stay, change);
    } else {
        stay = findOptimal(week-1, 'B', n, T, RA, RB, SA, SB) + RA[week] - SA[week];
        change = findOptimal(week-1, 'A', n, T, RA, RB, SA, SB) + RA[week] - SA[week] - T;
        parent[{week, city}] = (stay >= change) ? 'B' : 'A';
        return max(stay, change);
    }
    


}

int main() {
    
    // Define n, T, arrays
    int n = 5;
    double T = 50;

    // City A (CHANGE TO REAL NUMBRES)
    vector<double> RA = {20, 100, 120, 110, 130, 115}; 
    vector<double> RB = {10, 90, 130, 1050, 125, 120};

    // City B (CHANGE TO REAL NUMBERS)
    vector<double> SA = {10, 40, 45, 42, 48, 43};
    vector<double> SB = {10, 35, 50, 38, 45, 40};

    double optA = findOptimal(n-1, 'A', n, T, RA, RB, SA, SB);
    double optB = findOptimal(n-1, 'B', n, T, RA, RB, SA, SB);

    
    
    char best_start = (optA >= optB) ? 'A' : 'B';
    double max_revenue = max(optA, optB);
    
    cout << "Max Revenue-cost: $" << max_revenue << endl;
    
    
    vector<char> path;
    char current = best_start;
    for (int w = n-1; w >= 0; w--) {
        path.push_back(current);
        if (w > 0) {
            current = parent[{w, current}];
        }
    }
    
    cout << "Path:\n";
    for (int i = 0; i < path.size(); i++) {
        cout << path[i];
        if (i < path.size() - 1) cout << " ->";
    }
    cout << endl;
    
    return 0;
}