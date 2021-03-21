package past

import (
	"container/heap"
	"context"
	"fmt"
	"regexp"
	"sort"
	"time"
)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

var tree = TreeNode{Val: 1, Right: &TreeNode{Val: 3, Left: &TreeNode{Val: 6}, Right: &TreeNode{Val: 7}}}
var tree1 = TreeNode{Val: 1, Left: &TreeNode{Val: 2, Left: &TreeNode{Left: &TreeNode{Val: 4}, Right: &TreeNode{Val: 5}}}, Right: &TreeNode{Val: 3, Left: &TreeNode{Val: 6}, Right: &TreeNode{Val: 7}}}

func numIdenticalPairs(nums []int) int {
	var p1, p2 *int
	s := 0
	for i := 0; i < len(nums); i++ {
		p1 = &nums[i]
		for k := i + 1; k < len(nums); k++ {
			p2 = &nums[k]
			if *p1 == *p2 {
				s++
			}
		}
	}
	return s
}

func kidsWithCandies(candies []int, extraCandies int) []bool {
	re := make([]bool, len(candies))
	var max int
	max = 0
	for _, c := range candies {
		if c > max {
			max = c
		}
	}
	fmt.Println(max)
	for i, c := range candies {
		if max > c+extraCandies {
			re[i] = false
		} else {
			re[i] = true
		}
	}
	return re
}

func isNumber(s string) bool {
	res := regexp.MustCompile(`^(([+\-]?[0-9]+(\.[0-9]*)?)|([+\-]?\.?[0-9]+))([eE][+\-]?[0-9]+)?$`).FindString(s)
	//	res:= regexp.MustCompile(`^ *[-+]?[0-9]+\.?[0-9]* *$|^ *[-+]?[0-9]*\.?[0-9]+ *$|^ *[-+]?[0-9]+\.?[0-9]*[Ee][-+]?[0-9]+ *$|^ *[-+]?[0-9]*\.?[0-9]+[Ee][-+]?[0-9]+ *$`).FindString(s)
	fmt.Println(res)
	if res == `` {
		return false
	}
	return true
}

func PredictTheWinner(nums []int) bool {
	return total(nums, 0, len(nums)-1, 1) >= 0
}

func total(nums []int, start, end int, turn int) int {
	if start == end {
		return nums[start] * turn
	}
	scoreStart := nums[start]*turn + total(nums, start+1, end, -turn)
	scoreEnd := nums[end]*turn + total(nums, start, end-1, -turn)
	return max(scoreStart*turn, scoreEnd*turn) * turn
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

func maxProfit(prices []int) int {
	all := 0
	nowI := prices[0]
	for i := 1; i < len(prices); i++ {
		if prices[i] < nowI {
			nowI = prices[i]
		} else if i < len(prices)-1 {
			if prices[i] < prices[i+1] {
				continue
			} else {
				all += prices[i] - nowI
				nowI = prices[i+1]
			}
		} else {
			all += prices[i] - nowI
		}
	}
	return all
}

func onesCount(x int) (c int) {
	for ; x > 0; x /= 2 {
		c += x % 2
	}
	return
}

func sortByBits(a []int) []int {
	sort.Slice(a, func(i, j int) bool {
		x, y := a[i], a[j]
		cx, cy := onesCount(x), onesCount(y)
		return cx < cy || cx == cy && x < y
	})
	return a
}

func validMountainArray(a []int) bool {
	i, n := 0, len(a)

	// 递增扫描
	for ; i+1 < n && a[i] < a[i+1]; i++ {
	}

	// 最高点不能是数组的第一个位置或最后一个位置
	if i == 0 || i == n-1 {
		return false
	}

	// 递减扫描
	for ; i+1 < n && a[i] > a[i+1]; i++ {
	}

	return i == n-1
}

func intersection(nums1 []int, nums2 []int) []int {
	var res []int
	for _, num1 := range nums1 {
		for _, num2 := range nums2 {
			if num2 == num1 {
				k := 0
				for _, r := range res {
					if r == num1 {
						k = 1
						break
					}
				}
				if k == 0 {
					res = append(res, num1)
				}
			}
		}
	}
	return res
}

func Proc(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			fmt.Println(`结束了`)
			return
		default:
			fmt.Println(123456)
			//do
		}
	}
}

func dfs(root *TreeNode, prevSum int) int {
	if root == nil {
		return 0
	}
	sum := prevSum*10 + root.Val
	if root.Left == nil && root.Right == nil {
		return sum
	}
	return dfs(root.Left, sum) + dfs(root.Right, sum)
}

func sumNumbers(root *TreeNode) int {
	return dfs(root, 0)
}

func gen() <-chan int {
	ch := make(chan int)
	go func() {
		var n int
		for {
			ch <- n
			n++
			time.Sleep(time.Second)
		}
	}()
	return ch
}

func topKFrequent(nums []int, k int) []int {
	occurrences := map[int]int{}
	for _, num := range nums {
		occurrences[num]++
	}
	h := &IHeap{}
	heap.Init(h)
	for key, value := range occurrences {
		heap.Push(h, [2]int{key, value})
		if h.Len() > k {
			heap.Pop(h)
		}
	}
	ret := make([]int, k)
	for i := 0; i < k; i++ {
		ret[k-i-1] = heap.Pop(h).([2]int)[0]
	}
	return ret
}

type IHeap [][2]int

func (h IHeap) Len() int           { return len(h) }
func (h IHeap) Less(i, j int) bool { return h[i][1] < h[j][1] }
func (h IHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *IHeap) Push(x interface{}) {
	*h = append(*h, x.([2]int))
}

func (h *IHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

func levelOrderBottom(root *TreeNode) [][]int {
	//深度优先遍历(dfs)
	result := make([][]int, 0)
	level := 0
	if root == nil {
		return result
	}

	orderBottom(root, &result, level)

	//数组翻转
	resultLength := len(result)
	left := 0
	right := resultLength - 1
	for left < right {
		temp := result[left]
		result[left] = result[right]
		result[right] = temp

		left++
		right--
	}

	return result
}

func orderBottom(root *TreeNode, result *[][]int, level int) {
	if root == nil {
		return
	}
	fmt.Println(len(*result), level)

	if len(*result) > level {
		(*result)[level] = append((*result)[level], root.Val)
	} else {
		*result = append(*result, []int{root.Val})
	}

	orderBottom(root.Left, result, level+1)
	orderBottom(root.Right, result, level+1)
}

func sumOfLeftLeaves(root *TreeNode) int {

	stack := make([]*TreeNode, 0)
	var res int
	for root != nil || len(stack) > 0 {
		if root != nil {
			stack = append(stack, root)
			root = root.Left
			if root != nil && root.Left == nil && root.Right == nil {
				res += root.Val
			}
		} else {
			node := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			root = node.Right
		}
	}
	return res

}
func mergeTrees(t1, t2 *TreeNode) *TreeNode {
	if t1 == nil {
		return t2
	}
	if t2 == nil {
		return t1
	}
	t1.Val += t2.Val
	t1.Left = mergeTrees(t1.Left, t2.Left)
	t1.Right = mergeTrees(t1.Right, t2.Right)
	return t1
}

func invertTree(root *TreeNode) *TreeNode {
	orgRoot := root
	recursion(root)
	return orgRoot
}

func recursion(root *TreeNode) {
	if root == nil {
		return
	} else {
		left := root.Left
		root.Left = root.Right
		root.Right = left
		recursion(root.Right)
		recursion(root.Left)
	}
}

func traverse(root *TreeNode) {
	if root == nil {
		return
	} else {
		fmt.Println(root.Val)
		traverse(root.Right)
		traverse(root.Left)
	}
}

func generate(numRows int) [][]int {

	if numRows == 0 {
		return [][]int{}
	}

	if numRows == 1 {
		return [][]int{{1}}
	}

	if numRows == 2 {
		return [][]int{{1}, {1, 1}}
	}
	res := [][]int{{1}, {1, 1}}

	for i := 2; i < numRows; i++ {
		for k := 0; k <= i; k++ {
			if k == 0 {
				res = append(res, []int{1})
			} else if k == i {
				res[i] = append(res[i], 1)
			} else {
				res[i] = append(res[i], res[i-1][k]+res[i-1][k-1])
			}
		}
	}
	return res
}

func countPrimes(n int) int {
	if n < 2 {
		return 0
	}
	res := 0
	for i := 2; i < n; i++ {
		if checkN(i) {
			res++
		}
	}
	return res
}

func checkN(n int) bool {
	for i := 2; i < n; i++ {
		if n%i == 0 {
			return false
		}
	}
	return true
}

func moveZeroes(nums []int) {
	var c0 []int
	for i := 0; i < len(nums); i++ {
		if nums[i] == 0 {
			//nums = append(nums[:i],append(nums[i+1:],0)...)
			nums = append(nums[:i], nums[i+1:]...)
			i--
			c0 = append(c0, 0)
		}
	}
	nums = append(nums, c0...)
}

func searchRange1(nums []int, target int) []int {
	leftmost := sort.SearchInts(nums, target)
	if leftmost == len(nums) || nums[leftmost] != target {
		return []int{-1, -1}
	}
	rightmost := sort.SearchInts(nums, target+1) - 1
	return []int{leftmost, rightmost}
}

func searchRange(nums []int, target int) []int {
	res := []int{-1, -1}
	for i := 0; i < len(nums); i++ {
		if nums[i] == target {
			res[0] = i
			break
		}

	}
	for i := len(nums) - 1; i >= 0; i-- {
		if nums[i] == target {
			res[1] = i
			break
		}
	}
	return res
}

func relativeSortArray(arr1 []int, arr2 []int) []int {
	var res []int
	for i := 0; i < len(arr2); i++ {
		for k := 0; k < len(arr1); k++ {
			if arr2[i] == arr1[k] {
				res = append(res, arr1[k])
				arr1 = append(arr1[:k], arr1[k+1:]...)
				k--
			}
		}
	}

	sort.Ints(arr1)
	res = append(res, arr1...)
	return res
}

func canCompleteCircuit(gas []int, cost []int) int {
	res := 0
	for ; res < len(gas); res++ {
		var jud = func() bool {
			nowGas := 0
			for i := res; i < len(gas); i++ {
				nowGas = nowGas + gas[i] - cost[i]
				if nowGas < 0 {
					return false
				}
			}
			for i := 0; i < res; i++ {
				nowGas = nowGas + gas[i] - cost[i]
				if nowGas < 0 {
					return false
				}
			}
			return true
		}
		if jud() == true {
			return res
		}
	}
	return -1
}
