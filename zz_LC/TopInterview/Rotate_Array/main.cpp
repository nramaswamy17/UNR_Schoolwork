#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

/*
Approach 1: Create a new array and push each element starting at k to the new array
Approach 2: Create a variable placeholder, then move each element by 1 step. Repeat k times
To move each element 1 step:
    placeholder = nums[i+1]
    nums[i+1] = nums[i]
Approach 3: Also have variable placeholder. Then move each element by however much it needs to move.
Set the placeholder equal to the value it's going to replace and reepat. 

[1,2,3] => [1,1,3] => 
[3,1,2]
[2,3,1]
[1,2,3]



move each one by one and store extra in a variable? 


*/


class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        // Your solution goes here
        // Remember: rotate the array to the right by k steps
        int p;
        for (int i = nums.size() - 1; i >= 0; i--){
            if (i == (nums.size() - 1)) {
                p = nums[i];
                cout << "P = " << p;
            }
            if (i == 0){
                nums[i] = p;
            }
            nums[i] = nums[i-1];
        }
        
    }
    
    // You can add additional helper methods here if needed
    
};

// Helper function to print array
void printArray(const vector<int>& nums) {
    cout << "[";
    for (int i = 0; i < nums.size(); i++) {
        cout << nums[i];
        if (i < nums.size() - 1) cout << ",";
    }
    cout << "]" << endl;
}

int main() {
    Solution solution;
    
    // Test case 1: nums = [1,2,3,4,5,6,7], k = 3
    // Expected: [5,6,7,1,2,3,4]
    vector<int> nums1 = {1, 2, 3, 4, 5, 6, 7};
    cout << "Test 1:" << endl;
    cout << "Input: nums = ";
    printArray(nums1);
    cout << "k = 3" << endl;
    solution.rotate(nums1, 3);
    cout << "Output: ";
    printArray(nums1);
    cout << "Expected: [5,6,7,1,2,3,4]" << endl << endl;
    
    // Test case 2: nums = [-1,-100,3,99], k = 2
    // Expected: [3,99,-1,-100]
    vector<int> nums2 = {-1, -100, 3, 99};
    cout << "Test 2:" << endl;
    cout << "Input: nums = ";
    printArray(nums2);
    cout << "k = 2" << endl;
    solution.rotate(nums2, 2);
    cout << "Output: ";
    printArray(nums2);
    cout << "Expected: [3,99,-1,-100]" << endl << endl;
    
    // Test case 3: Edge case - k larger than array size
    vector<int> nums3 = {1, 2};
    cout << "Test 3 (k > array size):" << endl;
    cout << "Input: nums = ";
    printArray(nums3);
    cout << "k = 3" << endl;
    solution.rotate(nums3, 3);
    cout << "Output: ";
    printArray(nums3);
    cout << "Note: k=3 should be equivalent to k=1 for array of size 2" << endl << endl;
    
    // Test case 4: Single element
    vector<int> nums4 = {1};
    cout << "Test 4 (single element):" << endl;
    cout << "Input: nums = ";
    printArray(nums4);
    cout << "k = 1" << endl;
    solution.rotate(nums4, 1);
    cout << "Output: ";
    printArray(nums4);
    cout << endl;
    
    return 0;
}