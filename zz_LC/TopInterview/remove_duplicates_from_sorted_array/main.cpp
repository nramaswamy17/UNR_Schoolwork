/*
This is sorted in non-decreasing order. 
Variable: 
(1) store the current number
(2) Count the number of distanct numbers in the array (whenever the # var changes)

This will require a pass through all elements (O(N))

func (vector nums)
    current_number = nums[0];
    distinct_nums = 0;

for every element
    if nums[i] != nums[0]
        then current_number = nums[i]
        and distainct_nums += 1
*/

#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:

    // Two pointer solution
    int removeDuplicates(vector<int>& nums) {
        int writeIndex = 0;
        int readIndex;
        int distinct_nums = 1; // first element is always distinct

        for (readIndex = 1; readIndex < nums.size(); readIndex++){
            if (nums[readIndex] != nums[writeIndex]) {
                writeIndex++;
                nums[writeIndex] = nums[readIndex];
                distinct_nums++;
            }
        }
        return distinct_nums;
    }

    // Slow version (first attempt)
    int removeDuplicates2(vector<int>& nums) {
        // Your solution goes here
        int current_number;
        int distinct_nums = 0;

        for (int i = 0; i < nums.size(); i++){
            if (nums[i] != current_number){
                current_number = nums[i];
                distinct_nums += 1;
            } // end if
            else {
                nums.erase(nums.begin() + i);
                i--; // adjust index after erasing
            }
        } // end for
        return distinct_nums;
    } // end removeDuplicates()
};

// Helper function to print array (first k elements)
void printArray(const vector<int>& nums, int k) {
    cout << "[";
    for (int i = 0; i < k; i++) {
        cout << nums[i];
        if (i < k - 1) cout << ", ";
    }
    cout << "]" << endl;
}

int main() {
    Solution solution;
    
    // Test case 1: nums = [1,1,2]
    // Expected: return 2, nums = [1,2,_]
    vector<int> nums1 = {1, 1, 2};
    cout << "Test 1:" << endl;
    cout << "Input: [1,1,2]" << endl;
    int k1 = solution.removeDuplicates(nums1);
    cout << "Output: k = " << k1 << ", nums = ";
    printArray(nums1, k1);
    cout << "Expected: k = 2, nums = [1,2]" << endl << endl;
    
    // Test case 2: nums = [0,0,1,1,1,2,2,3,3,4]
    // Expected: return 5, nums = [0,1,2,3,4,_,_,_,_,_]
    vector<int> nums2 = {0, 0, 1, 1, 1, 2, 2, 3, 3, 4};
    cout << "Test 2:" << endl;
    cout << "Input: [0,0,1,1,1,2,2,3,3,4]" << endl;
    int k2 = solution.removeDuplicates(nums2);
    cout << "Output: k = " << k2 << ", nums = ";
    printArray(nums2, k2);
    cout << "Expected: k = 5, nums = [0,1,2,3,4]" << endl << endl;
    
    // Test case 3: Single element
    vector<int> nums3 = {1};
    cout << "Test 3:" << endl;
    cout << "Input: [1]" << endl;
    int k3 = solution.removeDuplicates(nums3);
    cout << "Output: k = " << k3 << ", nums = ";
    printArray(nums3, k3);
    cout << "Expected: k = 1, nums = [1]" << endl;
    
    return 0;
}