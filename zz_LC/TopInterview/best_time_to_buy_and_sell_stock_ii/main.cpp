#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int profit = 0;

        for (int i = 0; i < prices.size() - 1; i++) {
            if (prices[i+1] - prices[i] > 0){
                profit += prices[i+1] - prices[i];
            }
        }
        return profit;
    }
};

int main() {
    Solution solution;
    
    // Test case 1: prices = [7,1,5,3,6,4]
    // Expected: 7
    vector<int> prices1 = {7, 1, 5, 3, 6, 4};
    cout << "Test 1:" << endl;
    cout << "Input: [7,1,5,3,6,4]" << endl;
    int profit1 = solution.maxProfit(prices1);
    cout << "Output: " << profit1 << endl;
    cout << "Expected: 7" << endl << endl;
    
    // Test case 2: prices = [1,2,3,4,5]
    // Expected: 4
    vector<int> prices2 = {1, 2, 3, 4, 5};
    cout << "Test 2:" << endl;
    cout << "Input: [1,2,3,4,5]" << endl;
    int profit2 = solution.maxProfit(prices2);
    cout << "Output: " << profit2 << endl;
    cout << "Expected: 4" << endl << endl;
    
    // Test case 3: prices = [7,6,4,3,1]
    // Expected: 0
    vector<int> prices3 = {7, 6, 4, 3, 1};
    cout << "Test 3:" << endl;
    cout << "Input: [7,6,4,3,1]" << endl;
    int profit3 = solution.maxProfit(prices3);
    cout << "Output: " << profit3 << endl;
    cout << "Expected: 0" << endl << endl;
    
    // Test case 4: Single day
    vector<int> prices4 = {5};
    cout << "Test 4:" << endl;
    cout << "Input: [5]" << endl;
    int profit4 = solution.maxProfit(prices4);
    cout << "Output: " << profit4 << endl;
    cout << "Expected: 0" << endl;
    
    return 0;
}