package past

import (
	"container/heap"
	"context"
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

var tree = TreeNode{Val: 1, Right: &TreeNode{Val: 3, Left: &TreeNode{Val: 6}, Right: &TreeNode{Val: 7}}}
var tree1 = TreeNode{Val: 1, Left: &TreeNode{Val: 2, Left: &TreeNode{Left: &TreeNode{Val: 4}, Right: &TreeNode{Val: 5}}}, Right: &TreeNode{Val: 3, Left: &TreeNode{Val: 6}, Right: &TreeNode{Val: 7}}}

func climbStairs(n int) int {
	if 0 == n {
		return 1
	}
	if n <= 2 {
		return n
	}
	a, b, tmp := 1, 2, 0
	for i := 3; i <= n; i++ {
		tmp = a + b
		a = b
		b = tmp
	}
	return tmp
}

func mySqrt(x int) int {
	return int(math.Sqrt(float64(x)))
}
func lengthOfLastWord(s string) int {
	a := 0
	l := 0
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] == 32 && a == 0 {
			continue
		} else if s[i] == 32 {
			return l
		} else {
			a = 1
			l++
		}
	}
	return l
}
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil {
		return l2
	} else if l2 == nil {
		return l1
	} else if l1.Val < l2.Val {
		l1.Next = mergeTwoLists(l1.Next, l2)
		return l1
	} else {
		l2.Next = mergeTwoLists(l1, l2.Next)
		return l2
	}
}

func removeElement(nums []int, val int) int {
	left, right := 0, len(nums)
	for left < right {
		if nums[left] == val {
			nums[left] = nums[right-1]
			right--
		} else {
			left++
		}
	}
	return left
}

func toD(i int) int {
	cd := 30
	nc := 0
	var f int
	f = 5
	ha := (i - 1) / 8
	if i-ha*8 > 4 {
		f = 4
	}
	dest := int(math.Abs(float64(20 * ((i - ha*8) - f))))
	nc = dest + (-(i-24)/8)*20
	nd := nc + cd
	return nd
}

func searchInsert(nums []int, target int) int {
	if target > nums[len(nums)] {
		return len(nums)
	}
	for i := 0; i < len(nums); i++ {
		if nums[i] == target {
			return i
		} else if i < len(nums) {
			if nums[i] < target && nums[i+1] > target {
				return i + 1
			}
		}
	}

	return 0
}
func removeDuplicates(nums []int) int {
	i1 := 1
	for i := 1; i < len(nums); i++ {
		if nums[i] != nums[i-1] {
			nums = append(nums[:i1], nums[i:]...)
			i1 = i - (i - 1 - i1)
			i = i1 - 1
		}
	}
	if len(nums) >= 1 {
		nums = nums[:i1]
	}
	return len(nums)
}

func isValid(s string) bool {
	var queue []int32
	for _, b := range s {
		if len(queue) != 0 {
			switch queue[len(queue)-1] {
			case 40:
				if b == 41 {
					queue = queue[:len(queue)-1]
				} else {
					queue = append(queue, b)
				}
			case 91:
				if b == 93 {
					queue = queue[:len(queue)-1]
				} else {
					queue = append(queue, b)
				}

			case 123:
				if b == 125 {
					queue = queue[:len(queue)-1]
				} else {
					queue = append(queue, b)
				}
			}
		} else {
			queue = append(queue, b)
		}
	}
	return len(queue) == 0
}

func setZeroes(matrix [][]int) {
	var zeros []point
	for y, line := range matrix {
		for x, k := range line {
			if k == 0 {
				fmt.Println(x, y)
				zeros = append(zeros, point{
					x: x,
					y: y,
				})
			}
		}
	}
	fmt.Println(zeros)
	for _, zero := range zeros {
		for i := 0; i < len(matrix[zero.y]); i++ {
			matrix[zero.y][i] = 0
		}
		for i := 0; i < len(matrix); i++ {
			matrix[i][zero.x] = 0
		}
	}
}

type point struct {
	x int
	y int
}

func evalRPN(tokens []string) int {
	var queue []int
	for _, token := range tokens {
		switch token {
		case `+`:
			t := queue[len(queue)-2] + queue[len(queue)-1]
			queue = queue[:len(queue)-2]
			queue = append(queue, t)
		case `-`:
			t := queue[len(queue)-2] - queue[len(queue)-1]
			queue = queue[:len(queue)-2]
			queue = append(queue, t)
		case `*`:
			t := queue[len(queue)-2] * queue[len(queue)-1]
			queue = queue[:len(queue)-2]
			queue = append(queue, t)

		case `/`:
			t := queue[len(queue)-2] / queue[len(queue)-1]
			fmt.Println(t)
			queue = queue[:len(queue)-2]
			queue = append(queue, t)
		}
		k, _ := strconv.Atoi(token)
		queue = append(queue, k)
		fmt.Println(queue)
	}
	return queue[0]
}

type ListNode struct {
	Val  int
	Next *ListNode
}

func reverseBetween(head *ListNode, left int, right int) *ListNode {
	if head.Next == nil || left == right {
		return head
	}
	i := 1
	var leftleftnode *ListNode
	var lastnode *ListNode
	var leftnode *ListNode
	var rightnode *ListNode
	trav := func(node *ListNode) {}
	trav = func(node *ListNode) {
		if node.Next == nil {
			return
		}
		if i == left-1 {
			leftleftnode = node
		}
		if i == left {
			lastnode = node
			leftnode = node
		}
		if i == right {
			rightnode = node
			leftnode.Next = node.Next
		}
		fmt.Println(node.Val, i)
		if i > left && i <= right {
			n := node.Next
			node.Next = lastnode
			lastnode = node
			i++
			trav(n)
			return
		}
		i++
		trav(node.Next)
	}
	trav(head)
	if leftleftnode != nil {
		leftleftnode.Next = rightnode
	}
	return head
}

func generateMatrix(n int) [][]int {
	var res [][]int
	for i := 0; i < n; i++ {
		res = append(res, make([]int, n))
	}
	x, y, d := 0, 0, 0
	for i := 1; i <= n*n; i++ {
		res[y][x] = i
		switch d {
		case 0:
			if len(res[y])-1 == x || res[y][x+1] != 0 {
				y++
				d++
			} else {
				x++
			}
		case 1:
			if len(res)-1 == y || res[y+1][x] != 0 {
				x--
				d++
			} else {
				y++
			}
		case 2:
			if x == 0 || res[y][x-1] != 0 {
				y--
				d++
			} else {
				x--
			}
		case 3:
			if y == 0 || res[y-1][x] != 0 {
				x++
				d = 0
			} else {
				y--
			}
		}
	}
	return res
}

type MyHashMap struct {
	data map[int]int
}

/** Initialize your data structure here. */
func Constructor() MyHashMap {
	return MyHashMap{
		data: make(map[int]int),
	}
}

/** value will always be non-negative. */
func (this *MyHashMap) Put(key int, value int) {
	this.data[key] = value
}

/** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
func (this *MyHashMap) Get(key int) int {
	v, ok := this.data[key]
	if !ok {
		return -1
	}
	return v
}

/** Removes the mapping of the specified value key if this map contains a mapping for the key */
func (this *MyHashMap) Remove(key int) {
	delete(this.data, key)
}

/**
 * Your MyHashMap object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Put(key,value);
 * param_2 := obj.Get(key);
 * obj.Remove(key);
 */

func findDisappearedNumbers(nums []int) (ans []int) {
	n := len(nums)
	for _, v := range nums {
		v = (v - 1) % n
		nums[v] += n
	}
	for i, v := range nums {
		if v <= n {
			ans = append(ans, i+1)
		}
	}
	return
}

func rotate(matrix [][]int) {
	N := len(matrix)
	for i := 1; i < N; i++ { //对称一波
		for k := 0; k < i; k++ {
			b := matrix[i][k]
			matrix[i][k] = matrix[k][i]
			matrix[k][i] = b
		}
	}
	for i := 0; i < N; i++ { //镜像翻转一波
		for k := 0; k < N/2; k++ {
			b := matrix[i][k]
			matrix[i][k] = matrix[i][N-1-k]
			matrix[i][N-1-k] = b
		}
	}
}
func firstUniqChar(s string) int {
	a := make(map[string]int)
	for i := 0; i < len(s); i++ {
		a[s[i:i+1]] += 1
	}
	fmt.Println(a)
	return 1
}

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
func GuessingGame() {
	var s string
	fmt.Printf("Pick an integer from 0 to 100.\n")
	answer := sort.Search(100, func(i int) bool {
		fmt.Printf("Is your number <= %d? ", i)
		_, err := fmt.Scanf("%s", &s)
		if err != nil {
			return false
		}
		fmt.Println(s)
		return s != "" && s[0] == 'y'
	})
	fmt.Printf("Your number is %d.\n", answer)
}

func shipWithinDays(weights []int, D int) int {
	// 确定二分查找左右边界
	left, right := 0, 0
	for _, w := range weights {
		if w > left {
			left = w
		}
		right += w
	}
	fmt.Println(left, right)
	return left + sort.Search(right-left, func(x int) bool {
		fmt.Println(`x`, x)
		x += left
		day := 1 // 需要运送的天数
		sum := 0 // 当前这一天已经运送的包裹重量之和
		for _, w := range weights {
			if sum+w > x {
				day++
				sum = 0
			}
			sum += w
		}
		fmt.Println(day)
		return day <= D
	})
}

func minDiffInBST(root *TreeNode) int {
	var v int
	var min int
	reserve := func(n *TreeNode) {}
	reserve = func(n *TreeNode) {
		if n == nil {
			return
		}
		if v-n.Val > min {
			min = v - n.Val
		}
		fmt.Println(n.Val)
		reserve(n.Left)
		reserve(n.Right)
	}
	reserve(root)
	return min
}

func nthUglyNumber(n int) int {
	if n == 1 {
		return 1
	}
	k := 1
	for i := 2; ; i++ {
		if k == n {
			return i
		}
		for b := 2; b < i; b++ {
			//fmt.Println(i%b,i,b,k)
			if i%b == 0 && b != 2 && b != 3 && b != 5 { //不是丑数
				fmt.Println(i, b)
				goto this
			}
		}
		k++
	this:
	}
}
func merge(nums1 []int, m int, nums2 []int, n int) {
	if m == len(nums1) {
		return
	}
	idx1, idx2 := 0, 0
	for k := 0; k < m+n; k++ {
		if len(nums2) > idx2 {
			if nums1[idx1] >= nums2[idx2] {
				for z := len(nums1) - 1; z > idx1; z-- {
					nums1[z] = nums1[z-1]
				}
				nums1[idx1] = nums2[idx2]
				idx2++
			}
		}
		if nums1[idx1] == 0 {
			nums1[idx1] = nums2[idx2]
			idx2++
		}
		idx1++
	}
}

func searchMatrix(matrix [][]int, target int) bool {
	//last := -999999999
	for i := 0; i < len(matrix); i++ {
		for r := 0; r < len(matrix[i]); r++ {
			if matrix[i][r] == target {
				return true
			}
			if matrix[i][r] > target {
				return false
			}
		}
	}
	return false
}

func reverseBits(n uint32) (rev uint32) {
	for i := 0; i < 32 && n > 0; i++ {
		fmt.Println(rev, n)
		rev |= n & 1 << (31 - i)
		n >>= 1
	}

	return
}

func plusOne(digits []int) []int {
	if digits[len(digits)-1] != 9 {
		digits[len(digits)-1] += 1
		return digits
	}
	digits[len(digits)-1] = 0
	p := true
	for i := len(digits) - 2; i >= 0; i-- {
		fmt.Println(digits)
		if p == true && digits[i] == 9 {
			digits[i] = 0
			continue
		}
		if p == true {
			p = false
			digits[i] += 1
		}
	}
	if p == true {
		digits = append([]int{1}, digits...)
	}
	return digits
}

func deleteDuplicates(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}

	cur := head
	for cur.Next != nil {
		if cur.Next.Val == cur.Val {
			cur.Next = cur.Next.Next
		} else {
			cur = cur.Next
		}
	}
	return head
}

func deleteDuplicates1(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}

	dummy := &ListNode{0, head} //哑巴节点

	cur := dummy
	for cur.Next != nil && cur.Next.Next != nil {
		if cur.Next.Val == cur.Next.Next.Val {
			x := cur.Next.Val
			for cur.Next != nil && cur.Next.Val == x {
				cur.Next = cur.Next.Next
			}
		} else {
			cur = cur.Next
		}
	}

	return dummy.Next
}

func countNegatives(grid [][]int) int {
	l := len(grid[0])
	r := 0
	for _, ints := range grid {
		for i := l; i > 0; i-- {
			if ints[i-1] < 0 {
				r++
			} else {
				break
			}
		}
	}
	return r
}

func reverseString(s []byte) {
	s1 := s
	a := len(s1) - 1
	fmt.Println(s1)
	for _, b := range s1 {
		fmt.Println(a, b)
		s[a] = b
		a--
	}
}

func countBits(n int) (res []int) {
	for i := 0; i < n; i++ {
		bin := DecConvertToX(i, 2)
		res = append(res, strings.Count(bin, "1"))
	}
	return
}

//268. 丢失的数字
func missingNumber(nums []int) int {
	arr := make([]bool, len(nums))
	for _, num := range nums {
		arr[num] = true
	}
	for i, b := range arr {
		if b == false {
			return i
		}
	}
	return 0
}

//258. 各位相加
func addDigits(num int) int {
	for {
		arr := getBit(num)
		num = 0
		for _, i := range arr {
			num += i
		}
		if num < 10 {
			return num
		}
	}
}

//242. 有效的字母异位词
func isAnagram(s string, t string) bool {
	m := map[int32]int{}
	for _, i := range s {
		m[i]++
	}
	for _, i := range t {
		m[i]--
	}
	for _, i := range m {
		if i != 0 {
			return false
		}
	}
	return true
}

//231. 2 的幂
func isPowerOfTwo(n int) bool {
	switch n {
	case 1:
		return true
	case 2:
		return true
	case 4:
		return true
	case 8:
		return true
	case 16:
		return true
	case 32:
		return true
	case 64:
		return true
	case 128:
		return true
	case 256:
		return true
	case 512:
		return true
	case 1024:
		return true
	case 2048:
		return true
	case 4096:
		return true
	case 8192:
		return true
	case 16384:
		return true
	case 32768:
		return true
	case 65536:
		return true
	case 131072:
		return true
	case 262144:
		return true
	case 524288:
		return true
	case 1048576:
		return true
	case 2097152:
		return true
	case 4194304:
		return true
	case 8388608:
		return true
	case 16777216:
		return true
	case 33554432:
		return true
	case 67108864:
		return true

	case 134217728:
		return true

	case 268435456:
		return true

	case 536870912:
		return true
	case 1073741824:
		return true
	}
	return false
}

//228. 汇总区间
func summaryRanges(nums []int) (res []string) {
	if len(nums) == 0 {
		return
	}
	k := 0
	for i := 0; i < len(nums)-1; i++ {
		if nums[i+1]-nums[i] != 1 {
			//生成单个
			r := strconv.Itoa(nums[i])
			if k != i {
				r = strconv.Itoa(nums[k]) + "->" + r
			}

			res = append(res, r)
			k = i + 1
		}
	}
	if k == len(nums)-1 {
		res = append(res, strconv.Itoa(nums[len(nums)-1]))
	} else {
		res = append(res, strconv.Itoa(nums[k])+"->"+strconv.Itoa(nums[len(nums)-1]))
	}
	return
}

type MyStack struct {
	arr []int
}

//225. 用队列实现栈
func Constructor() MyStack {
	return MyStack{}
}

func (s *MyStack) Push(x int) {
	s.arr = append(s.arr, x)
}

func (s *MyStack) Pop() int {
	a := s.arr[len(s.arr)]
	s.arr = s.arr[:len(s.arr)-1]
	return a
}

func (s *MyStack) Top() int {
	return s.arr[len(s.arr)]
}

func (s *MyStack) Empty() bool {
	return len(s.arr) == 0
}

//219. 存在重复元素 II
func containsNearbyDuplicate(nums []int, k int) bool {
	m := make(map[int]int)
	for i, num := range nums {
		if i-m[num] <= k && m[num] != 0 {
			return true
		}
		m[num] = i
	}
	return false
}

func myPow(x float64, n int) float64 {
	return math.Pow(x, float64(n))
}

func majorityElement(nums []int) int {
	m := make(map[int]int)
	for _, num := range nums {
		m[num]++
		if m[num] > (len(nums) / 2) {
			return num
		}
	}
	return 0
}

//168. Excel表列名称
func convertToTitle(columnNumber int) string {
	ans := []byte{}
	for columnNumber > 0 {
		a0 := (columnNumber-1)%26 + 1
		ans = append(ans, 'A'+byte(a0-1))
		columnNumber = (columnNumber - a0) / 26
	}
	for i, n := 0, len(ans); i < n/2; i++ {
		ans[i], ans[n-1-i] = ans[n-1-i], ans[i]
	}
	return string(ans)
}

//1996. 游戏中弱角色的数量
func numberOfWeakCharacters(properties [][]int) (ans int) {
	sort.Slice(properties, func(i, j int) bool {
		p, q := properties[i], properties[j]
		return p[0] < q[0] || p[0] == q[0] && p[1] > q[1]
	})
	var st []int
	for _, p := range properties {
		for len(st) > 0 && st[len(st)-1] < p[1] {
			st = st[:len(st)-1]
			ans++
		}
		st = append(st, p[1])
	}
	return
}

//171. Excel 表列序号
func titleToNumber(columnTitle string) (res int) {

	l := len(columnTitle)
	k := 1
	for i := l; i > 0; i-- {
		res += int(columnTitle[i-1]-64) * k
		k *= 26
	}

	return
}

//202. 快乐数
func isHappy(n int) bool {
	m := make(map[int]bool)
	for {
		arr := getBit(n)
		n = 0
		for _, i := range arr {
			n += i * i
		}

		if m[n] == true {
			return false
		}
		m[n] = true

		if n <= 3 {
			if n == 1 {
				return true
			} else {
				return false
			}
		}
	}
}

//217. 存在重复元素
func containsDuplicate(nums []int) bool {
	if len(nums) > 0 {
		for k, v := range nums {
			for _, vv := range nums[k+1:] {
				if v == vv {
					return true
				}
			}
		}
	}
	return false
}

//461. 汉明距离
func hammingDistance(x int, y int) (res int) {
	n := y
	b := x
	if y < x {
		n = x
		b = y
	}

	for ; n > 0; n /= 2 {
		if n%2 != b%2 {
			res++
		}
		b /= 2
	}
	return
}

//459. 重复的子字符串
func repeatedSubstringPattern(s string) bool {
	l := len(s)
	for i := 1; i <= l/2; i++ {
		if l%i == 0 {
			subs := s[:i]
			t := true
			for k := 0; k < l/i; k++ {
				if s[k*i:k*i+i] != subs {
					t = false
					break
				}
			}
			if t {
				return true
			}
		}
	}
	return false
}

//453. 最小操作次数使数组元素相等
func minMoves(nums []int) (res int) {
	ln := len(nums)
	if len(nums) == 1 {
		return 0
	}
	for {
		//检查一致
		n := nums[0]
		t := true
		for _, num := range nums {
			if num != n {
				t = false
				break
			}
		}
		if t {
			return
		}

		//获取max
		maxIdx := 0
		max := nums[0]
		for i := 0; i < ln; i++ {
			if max < nums[i] {
				maxIdx = i
				max = nums[i]
			}
		}

		//加1
		for i := 0; i < ln; i++ {
			if i != maxIdx {
				nums[i]++
			}
		}

		res++

	}
}

//1447. 最简分数
func simplifiedFractions(n int) (res []string) {
	for i := 1; i < n; i++ {
		for k := n; k > 1; k-- {
			if i >= k {
				break
			}
			//判断化简
			for f := 2; f <= i; f++ {
				if i%f == 0 && k%f == 0 {
					goto this
				}
			}
			res = append(res, strconv.Itoa(i)+"/"+strconv.Itoa(k))
		this:
		}
	}
	return
}

func inorderTraversal(root *TreeNode) (res []int) {
	if root == nil {
		return res
	}

	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}

		dfs(node.Left)
		res = append(res, node.Val)
		dfs(node.Right)
	}
	dfs(root)

	return res
}

type CQueue struct {
	queue []int
}

func Constructor() CQueue {
	return CQueue{}
}

func (this *CQueue) AppendTail(value int) {
	this.queue = append(this.queue, value)
}

func (this *CQueue) DeleteHead() int {
	if len(this.queue) == 0 {
		return -1
	}
	res := this.queue[1]
	this.queue = this.queue[1:]
	return res
}

func search(nums []int, target int) int {
	i := sort.Search(len(nums), func(i int) bool {
		return nums[i] >= target
	})
	if len(nums) > i && nums[i] == target {
		return i
	}
	return -1

}
func longestWord(words []string) string {
	sort.Strings(words)
	fmt.Println(words)
	l := 0
	var arr []string
	for _, word := range words {
		if l == len(word) {
			arr = append(arr, word)
			continue
		}
		l = len(word)

	}

	if len(arr) != 0 {
		return arr[0]
	}
	return ""
}

func lengthOfLongestSubstring(s string) int {
	if len(s) == 0 {
		return 0
	}
	res := 1
	k := 0
	for i := 1; i <= len(s); i++ {
		chong := -1
		for i3, i2 := range s[k : i-1] {
			//	fmt.Println(i3, s[i-1], uint8(i2))
			if s[i-1] == uint8(i2) {
				chong = i3 + 1
				break
			}
		}
		if chong != -1 {
			k = k + chong
		}

		if i-k > res {
			res = i - k
		}
		//	fmt.Println(s[k:i], i, k)
	}
	return res
}

func maximumProduct(nums []int) int {
	res := 1
	sort.Ints(nums)
	l := len(nums) - 1
	for i := l; i >= l-2; i-- {
		//	fmt.Println(nums[i])
		res *= nums[i]
	}
	return res
}

type AllOne struct {
	m   map[string]int
	max string
	min string
}

func Constructor() AllOne {

	return AllOne{
		max: "",
		min: "",
		m:   make(map[string]int),
	}
}

func (this *AllOne) check(key string) {
	//fmt.Println("check", key, this)
	if this.min == "" {
		//	fmt.Println("min", key)
		this.min = key
	}
	if this.max == "" {
		//fmt.Println("max", key)
		this.max = key
	}
	if this.m[key] > this.m[this.max] {
		this.max = key
		max, min, maxstr, minstr := -1, math.MaxInt, "", ""
		for s, i := range this.m {
			if i > max {
				maxstr = s
				max = i
			}
			if i < min {
				minstr = s
				min = i
			}
		}
		this.max = maxstr
		this.min = minstr
	}
	if this.m[key] < this.m[this.min] {
		this.min = key
		max, min, maxstr, minstr := -1, math.MaxInt, "", ""
		for s, i := range this.m {
			if i > max {
				maxstr = s
				max = i
			}
			if i < min {
				minstr = s
				min = i
			}
		}
		this.max = maxstr
		this.min = minstr
	}
}

func (this *AllOne) Inc(key string) {
	this.m[key]++

	this.check(key)

}

func (this *AllOne) Dec(key string) {
	if this.m[key] == 1 {
		delete(this.m, key)
		if len(this.m) > 0 {
			max, min, maxstr, minstr := -1, math.MaxInt, "", ""
			for s, i := range this.m {
				if i > max {
					maxstr = s
					max = i
				}
				if i < min {
					minstr = s
					min = i
				}
			}
			this.max = maxstr
			this.min = minstr
		} else {
			this.min = ""
			this.max = ""
		}
		return
	}
	this.m[key]--

	this.check(key)
}

func (this *AllOne) GetMaxKey() string {
	return this.max
}

func (this *AllOne) GetMinKey() string {
	return this.min
}

//819. 最常见的单词
func mostCommonWord(paragraph string, banned []string) string {
	bandMap := make(map[string]bool)
	for _, s := range banned {
		bandMap[s] = true
	}
	paragraph = strings.ToLower(paragraph)
	s1 := strings.NewReplacer(",", " ", ".", " ", "!", " ", "?", " ", "'", " ", ";", " ").Replace(paragraph)
	s1 = strings.ReplaceAll(s1, "  ", " ")
	//fmt.Println(s1)
	strarr := strings.Split(s1, " ")
	wordMap := make(map[string]int)
	for _, s := range strarr {
		if bandMap[s] {
			continue
		}
		wordMap[s]++
	}
	max := 0
	var res string
	for s, i := range wordMap {
		if i > max {
			max = i
			res = s
		}
	}
	//	fmt.Println(wordMap)
	return res
}

//804. 唯一摩尔斯密码词
func uniqueMorseRepresentations(words []string) int {
	mosi := []string{".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--", "-.", "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."}
	m := make(map[string]bool)
	for _, word := range words {
		s := ""
		for _, i := range word {
			i -= 'a'
			s += mosi[i]
		}
		m[s] = true
		//		fmt.Println(s)
	}

	return len(m)
}

//806. 写字符串需要的行数
func numberOfLines(widths []int, s string) []int {
	count := 0
	line := 1
	for _, i := range s {
		i -= 'a'
		count += widths[i]
		if count > 100 {
			count = widths[i]
			line++
		} else if count == 100 {
			count = 0
			line++
		}
		fmt.Println(i)
	}
	return []int{line, count}
}
func countMaxOrSubsets(nums []int) int {
	res := 0
	max := 0
	//var arr []int
	var dfs func(int, int)

	dfs = func(k int, l int) {

		if l == max {
			res++
		} else if l > max {
			res = 0
			max = l
			k = 0
		}

		for i := k; i < len(nums); i++ {
			//arr = append(arr, nums[i])
			dfs(i+1, l|nums[i])
			//arr = arr[:len(arr)-1]
		}
	}
	dfs(0, 0)

	//fmt.Println(ziji, max)
	return res
}

func climbStairs(n int) int {
	var res int
	var hs func()
	i := 0
	step := []int{1, 2}
	hs = func() {
		if i >= n {
			fmt.Println(i)
			if i == n {
				res++
			}
			return
		}

		for _, s := range step {
			i += s
			hs()
			i -= s
		}

	}
	hs()
	return res
}

func subsets(nums []int) [][]int {
	sort.Ints(nums)
	var res [][]int
	var arr []int

	var dfs func(k int, f int)
	//floor := 0
	dfs = func(k int, f int) {
		//fmt.Println(arr)
		temp := make([]int, len(arr))
		copy(temp, arr)
		res = append(res, temp)

		for i := k; i < len(nums); i++ {
			fmt.Println(arr, f)
			//if i > 0 && nums[i] == nums[i-1] && len(arr) == f {
			//	continue
			//}

			arr = append(arr, nums[i])
			dfs(i+1, len(arr))
			arr = arr[:len(arr)-1]

		}
	}
	dfs(0, 0)
	return res
}

//46. 全排列
func permute(nums []int) (res [][]int) {
	var dfs func()
	visted := make(map[int]bool)
	var arr []int
	dfs = func() {
		if len(arr) == len(nums) {
			fmt.Println(arr)

			a := make([]int, len(arr))
			copy(a, arr)
			res = append(res, a)
			return
		}

		for _, n := range nums {
			if visted[n] {
				continue
			}
			arr = append(arr, n)
			visted[n] = true
			dfs()
			visted[n] = false
			arr = arr[:len(arr)-1]
		}
	}
	dfs()
	return
}

func bestRotation(nums []int) int {
	res := 0
	max := 0
	l := len(nums)
	for i := 0; i < l; i++ {
		score := 0
		idx := 0
		for k := i; k < l+i; k++ {
			countScore := func(n int) {
				//fmt.Print(n)
				if n <= idx {
					score++
				}
			}

			if k >= l {
				countScore(nums[k-l])
			} else {
				countScore(nums[k])
			}
			idx++
		}

		if score > max {
			res = i
			max = score
		}
		//fmt.Print("\n")
		//fmt.Println("score", score, "i", i)
	}
	return res
}

func gogogo() {
	opencast, err := os.Open("LBMA-GOLD.csv")
	if err != nil {
		log.Println("csv文件打开失败！")
	}

	reader := csv.NewReader(opencast)

	file, _ := reader.ReadAll()

	var res [][]string

	opencast.Close()

	tArr := geneTimeArr()
	for _, t := range *tArr {
		res = append(res, []string{t.Format("1/2/06")})
	}

	//fmt.Println(res)
	//16.9.12-21.9.10

	for i := 0; i < len(file); i++ {
		if len(file[i][0]) > 2 {

			if file[i][0][:2] == "20" {
				//fmt.Println(file[i][0])
				k, _ := strconv.Atoi(file[i][0][2:4])
				if k < 10 {
					file[i][0] = file[i][0][3:]
				} else {
					file[i][0] = file[i][0][2:]
				}
			}
		}

		//var layout string = "1/2/06"
		//var timeStr string = "9/3/21"
		//timeIdx, _ := time.Parse(layout, timeStr)
	}

	//fmt.Println(res[0])
	//fmt.Println(res[1])

	k := 0
	key := file[k][1]
	for i := 0; i < len(res); i++ {
		if k < len(file) && len(file[k][0]) == 0 {
			k++
		}
		if k < len(file) && res[i][0] == file[k][0] {

			res[i] = append(res[i], file[k][1])
			key = file[k][1]
			k++
		} else {
			res[i] = append(res[i], key)
		}

		var layout string = "1/2/06"
		t, _ := time.Parse(layout, res[i][0])
		res[i][0] = t.Format("06/1/2")

		//fmt.Println(res[i])
	}
	//fmt.Println(file)

	//OpenFile读取文件，不存在时则创建，使用追加模式
	f, err := os.OpenFile("res1.csv", os.O_RDWR|os.O_APPEND|os.O_CREATE, 0666)
	if err != nil {
		log.Println("文件打开失败！")
	}
	defer f.Close()

	//创建写入接口
	WriterCsv := csv.NewWriter(f)

	//写入一条数据，传入数据为切片(追加模式)
	err1 := WriterCsv.WriteAll(res)
	if err1 != nil {
		log.Println("WriterCsv写入文件失败")
	}
	for _, re := range res {
		fmt.Println(re)
	}
	//fmt.Println(res)
	WriterCsv.Flush() //刷新，不刷新是无法写入的
	log.Println("数据写入成功...")

}

func geneTimeArr() *[]time.Time {
	var layout string = "2006-01-02"
	var timeStr string = "2016-09-12"
	var timeStr1 string = "2021-09-10"
	timeIdx, _ := time.Parse(layout, timeStr)
	timeEnd, _ := time.Parse(layout, timeStr1)

	var timeArr []time.Time
	for {
		timeArr = append(timeArr, timeIdx)
		if timeIdx.Unix() > timeEnd.Unix() {
			break
		}
		timeIdx = timeIdx.Add(time.Hour * 24)
	}
	return &timeArr
}

//492. 构造矩形
func constructRectangle(area int) (res []int) {
	res = make([]int, 2)
	for i := 1; i <= int(math.Sqrt(float64(area))); i++ {
		if area%i == 0 {
			res[0] = area / i
			res[1] = i
		}
	}
	return
}

//495. 提莫攻击
func findPoisonedDuration(timeSeries []int, duration int) (ans int) {
	expired := 0
	for _, t := range timeSeries {
		if t >= expired {
			ans += duration
		} else {
			ans += t + duration - expired
		}
		expired = t + duration
	}
	return
}

//496. 下一个更大元素 I
func nextGreaterElement(nums1 []int, nums2 []int) (res []int) {
	for _, i := range nums1 {
		g := false

		for _, i3 := range nums2 {
			if i == i3 {
				g = true
			}
			//fmt.Println(g, i3)
			if g && i < i3 {
				res = append(res, i3)
				goto this
			}
		}
		res = append(res, -1)
	this:
	}
	return
}

//500. 键盘行
func findWords(words []string) (res []string) {
	keybord := []string{"qwertyuiopQWERTYUIOP", "asdfghjklASDFGHJKL", "zxcvbnmZXCVBNM"}
	findHang := func(w int32) int {
		for i, s := range keybord {
			for _, i2 := range s {
				if w == i2 {
					return i
				}
			}
		}
		return 0
	}

	for _, word := range words {
		h := findHang(int32(word[0]))
		for _, i := range word {
			//
			if findHang(i) != h {
				goto this
			}
		}
		res = append(res, word)
	this:
	}
	return
}

//504. 七进制数
//func convertToBase7(num int) string {
//	return DecConvertToX(num, 7)
//}

//1189. “气球” 的最大数量
func maxNumberOfBalloons(text string) (res int) {
	str := "balloon"
	m := make(map[int32]int, 0)
	for _, i2 := range text {
		m[i2]++
	}
	res = m[int32(str[0])]
	//fmt.Println(m)
	for _, i := range str {
		if i == 108 || i == 111 {
			//fmt.Println(i, m[i]/2)

			if m[i]/2 < res {
				res = m[i] / 2
			}
		} else {
			if m[i] < res {
				res = m[i]
			}
		}
	}
	return
}

//506. 相对名次
func findRelativeRanks(score []int) (res []string) {
	k := 1
	arr := make([]int, len(score))
	copy(arr, score)
	sort.Ints(arr)
	fmt.Println(arr, score)
	m := make(map[int]int)
	for i := len(score) - 1; i >= 0; i-- {
		m[arr[i]] = k
		k++
	}

	for _, i := range score {
		switch m[i] {
		case 1:
			res = append(res, "Gold Medal")
		case 2:
			res = append(res, "Silver Medal")
		case 3:
			res = append(res, "Bronze Medal")
		default:
			res = append(res, strconv.Itoa(m[i]))
		}
	}

	return
}

//507. 完美数
func checkPerfectNumber(num int) bool {
	if num == 1 {
		return false
	}
	sum := 1
	for i := 2; i <= int(math.Sqrt(float64(num))); i++ {
		if num%i == 0 {
			sum += i
			sum += num / i
		}
	}
	return sum == num
}

//509. 斐波那契数
func fib(n int) (res int) {
	if n == 0 {
		return
	}
	a := 0
	res = 1
	for i := 0; i < n-1; i++ {
		b := res
		res += a
		a = b
	}
	return
}

func hasAlternatingBits(n int) bool {
	if n == 1 {
		return true
	}
	if n == 2 {
		return true
	}
	if n == 5 {
		return true
	}
	if n == 10 {
		return true
	}
	if n == 21 {
		return true
	}
	if n == 42 {
		return true
	}
	if n == 85 {
		return true
	}
	if n == 170 {
		return true
	}
	if n == 341 {
		return true
	}
	if n == 682 {
		return true
	}
	if n == 1365 {
		return true
	}
	if n == 2730 {
		return true
	}
	if n == 5461 {
		return true
	}
	if n == 10922 {
		return true
	}
	if n == 21845 {
		return true
	}
	if n == 43690 {
		return true
	}
	if n == 87381 {
		return true
	}
	if n == 174762 {
		return true
	}
	if n == 349525 {
		return true
	}
	if n == 699050 {
		return true
	}
	if n == 1398101 {
		return true
	}
	if n == 2796202 {
		return true
	}
	if n == 5592405 {
		return true
	}
	if n == 11184810 {
		return true
	}
	if n == 22369621 {
		return true
	}
	if n == 44739242 {
		return true
	}
	if n == 89478485 {
		return true
	}
	if n == 178956970 {
		return true
	}
	if n == 357913941 {
		return true
	}
	if n == 715827882 {
		return true
	}
	if n == 1431655765 {
		return true
	}
	return false
}

func findMaxConsecutiveOnes(nums []int) int {
	count := 0
	m := 0
	for _, num := range nums {
		if num == 1 {
			count++
		} else {
			if count > m {
				m = count
			}
			count = 0
		}
	}
	if count > m {
		m = count
	}
	return m
}

func tribonacci(n int) int {
	if n == 0 {
		return 0
	}
	if n == 1 {
		return 0
	}
	if n == 2 {
		return 0
	}

	n1 := 0
	n2 := 1
	n3 := 1
	for i := 3; i < n; i++ {
		temp3 := n1 + n2 + n3
		n1 = n2
		n2 = n3
		n3 = temp3
	}
	return n3
}

func distributeCandies(candies int, num_people int) []int {
	res := make([]int, num_people)
	n := 1
	for {
		for i := 0; i < len(res); i++ {
			res[i] += n
			candies -= n
			if candies <= 0 {
				res[i] += candies
				return res
			}
			n++
		}
	}
}

func largestSumAfterKNegations(nums []int, k int) int {
	sort.Ints(nums)
	for i := 0; i < k; i++ {
		nums[i] = -nums[i]
	}
	res := 0
	for _, num := range nums {
		res += num
	}
	return res
}

func searchBST(root *TreeNode, val int) *TreeNode {
	var res *TreeNode
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		fmt.Println(node.Val)
		if node.Val == val {
			res = node
			return
		}

		dfs(node.Left)
		dfs(node.Right)
	}

	dfs(root)
	return res
}

func largestRectangleArea(heights []int) int {

	m := math.MinInt
	l := len(heights)

	for i := 0; i < l; i++ {
		il := i
		ir := i
		for il > 0 && heights[il-1] >= heights[i] {
			il--
		}
		for ir < l-1 && heights[ir+1] >= heights[i] {
			//fmt.Println(heights[ir], heights[i])
			ir++
		}
		area := (ir - il + 1) * heights[i]
		if area > m {
			m = area
		}
		//fmt.Println(il, ir, area)
	}
	return m

}

func convert(s string, numRows int) string {
	arr := make([][]uint8, numRows)
	j := 0
	down := true
	for i := 0; i < len(s); i++ {
		arr[j] = append(arr[j], s[i])

		if numRows == 1 {
			continue
		}

		switch down {
		case true:
			j++
		case false:
			j--
		}
		if j == 0 {
			down = true
		}
		if j == numRows-1 {
			down = false
		}
		//fmt.Println(arr)
	}
	res := ""
	for _, ints := range arr {
		for _, i := range ints {
			res += string(i)
		}
	}
	return res
}
func countNum(str string) {
	var stack []uint8

	for i := 0; i < len(str); i++ {
		if len(stack) == 0 {
			stack = append(stack, str[i])
		} else {
			if stack[len(stack)-1] == str[i] {
				stack = append(stack, str[i])
			} else {
				if len(stack) > 2 {
					delLen := len(stack) - 2
					str = str[:i-delLen] + str[i:]
					i -= delLen
				}
				stack = []uint8{str[i]}

			}
		}
		//fmt.Println(stack)
	}

	if len(stack) > 2 {
		i := len(str)
		delLen := len(stack) - 2
		str = str[:i-delLen] + str[i:]
		i -= delLen
	}
	stack = stack[:0]
	//fmt.Println(str)

	for i := 0; i < len(str); i++ {
		if len(stack) == 0 {
			stack = append(stack, str[i])
		} else {
			if len(stack) == 4 {
				str = str[:i-1] + str[i:]
				i--
				stack = []uint8{}
			} else if stack[len(stack)-1] == str[i] || len(stack) == 2 {
				stack = append(stack, str[i])
			} else {
				stack = []uint8{str[i]}
			}
		}
	}
	i := len(str)
	if len(stack) == 4 {
		str = str[:i-1] + str[i:]
		i--
		stack = []uint8{}
	}

	fmt.Println(str)

}

func max(nums ...int) int {
	m := nums[0]
	for _, i2 := range nums {
		if i2 > m {
			m = i2
		}
	}
	return m

}

func min(nums ...int) int {
	m := nums[0]
	for _, i2 := range nums {
		if i2 < m {
			m = i2
		}
	}

	return m
}

func missingRolls(rolls []int, mean int, n int) (res []int) {
	regexp.MustCompile("")
	l := len(rolls)
	sum := mean * (n + l)
	for _, roll := range rolls {
		sum -= roll
	}
	s := float32(sum) / float32(n)
	//fmt.Println(s)
	if s > 6 || s < 1 {
		return
	}
	s1 := int(s)
	for i := 0; i < n; i++ {
		sum -= s1
		res = append(res, s1)
	}
	fmt.Println(sum)
	i := 0
	for ; sum > 0; sum-- {
		res[i] += 1
		i++
	}
	return
}

func isBalanced(root *TreeNode) bool {
	res := true
	var dfs func(node *TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		left := dfs(node.Left)
		right := dfs(node.Right)
		if left-right > 1 || left-right < -1 {
			res = false
			return 0
		}
		return max(left, right) + 1
	}
	dfs(root)
	return res
}

func isSubsequence(t string, s string) bool {
	i1 := 0
	for i := 0; i < len(s); i++ {
		if i1 == len(t) {
			return true
		}
		//fmt.Println(string(s[i]), string(t[i1]))
		if s[i] == t[i1] {
			i1++
		}
	}
	//fmt.Println(i1)
	return len(t) == i1
}

func trap(height []int) int {
	res := 0
	l := len(height)
	maxLeft := make([]int, l)
	maxRight := make([]int, l)
	maxLeft[0] = height[0]
	maxRight[l-1] = height[l-1]
	for i := 1; i < l; i++ {
		maxLeft[i] = max(maxLeft[i-1], height[i])
	}
	for i := l - 2; i >= 0; i-- {
		//fmt.Println(l)
		maxRight[i] = max(maxRight[i+1], height[i])
	}
	//fmt.Println(maxLeft, maxRight)

	for i := 1; i < l-1; i++ {

		temp := min(maxRight[i], maxLeft[i]) - height[i]
		if temp < 0 {
			temp = 0
		}
		res += temp
		//fmt.Println(maxLeft, maxRight)
	}
	return res
}

func rob(nums []int) int {
	l := len(nums)
	if l == 0 {
		return 0
	}
	if l == 1 {
		return nums[0]
	}

	arr := make(map[int]int)
	arr[0] = nums[0]
	arr[1] = max(nums[0], nums[1])
	var dp func(k int) int
	dp = func(k int) int {
		//fmt.Println(k)
		n, ok := arr[k]
		if ok {
			return n
		}

		r := max(nums[k]+dp(k-2), dp(k-1))
		fmt.Println(k, nums[k]+dp(k-2), dp(k-1))
		arr[k] = r
		return r
	}
	dp(l - 1)
	return arr[l-1]
}

func nextPermutation(nums []int) {
	m := math.MinInt
	mIdx := -1
	for i, num := range nums {
		if num > m {
			m = num
			mIdx = i
		}
	}
	if mIdx == 0 {
		sort.Ints(nums)
		return
	}
	nums[mIdx], nums[mIdx-1] = nums[mIdx-1], nums[mIdx]

}

func longestPalindrome(s string) string {
	var arr [][]bool
	for i := 0; i < len(s); i++ {
		arr = append(arr, make([]bool, len(s)))
	}

	var dp func(i, j int) bool
	dp = func(i, j int) bool {
		ans := false
		fmt.Println(i, j)
		if i == j {
			ans = true
			arr[i][j] = ans
			return ans
		}

		if dp(i+1, j-1) == true && s[i] == s[j] {
			ans = true
		}
		arr[i][j] = ans
		return ans
	}
	dp(0, len(s)-1)
	fmt.Println(arr)

	return ""

	//check := func(str string) bool {
	//	i2 := len(str) - 1
	//	for i1 := 0; i1 < len(str); i1++ {
	//		if str[i1] != str[i2] {
	//			return false
	//		}
	//		i2--
	//	}
	//	return true
	//}
	//
	//for i := len(s); i > 0; i-- {
	//	for i1 := 0; i1 < len(s)-i+1; i1++ {
	//		tmpstr := s[i1 : i1+i]
	//		//fmt.Println(tmpstr, i1)
	//		if check(tmpstr) {
	//			return tmpstr
	//		}
	//	}
	//}
	//return ""
}

//dp[i] i的最大乘积
//dp[i] = max(dp[i-j]*j,(i-j)*j) 0<j<i
func integerBreak(n int) int {
	arr := make([]int, n+1)
	arr[2] = 1
	var dp func(k int) int
	dp = func(k int) int {
		//fmt.Println(k)
		if arr[k] != 0 {
			return arr[k]
		}

		m := math.MinInt
		for i := 1; i < k-1; i++ {
			n1 := (k - i) * i
			n2 := dp(k-i) * i
			m = max(m, n1, n2)
		}
		arr[k] = m
		return m

	}

	return dp(n)
}

func calPoints(ops []string) int {
	var rec []int
	for i := 0; i < len(ops); i++ {
		switch ops[i] {
		case "+":
			rec = append(rec, rec[len(rec)]+rec[len(rec)-1])
		case "D":
			rec = append(rec, rec[len(rec)]*2)
		case "C":
			rec = rec[:len(rec)-1]
		default:
			n, _ := strconv.Atoi(ops[i])
			rec = append(rec, n)
		}
	}
	res := 0
	for _, i := range rec {
		res += i
	}
	return res
}
func sortedSquares(nums []int) []int {
	for i := 0; i < len(nums); i++ {
		if nums[i] < 0 {
			nums[i] = -nums[i]
		}
	}
	sort.Ints(nums)
	for i := 0; i < len(nums); i++ {
		nums[i] = nums[i] * nums[i]
	}
	return nums
}
func uniquePathsWithObstacles(obstacleGrid [][]int) int {
	leny := len(obstacleGrid)
	lenx := len(obstacleGrid[0])
	if obstacleGrid[leny-1][lenx-1] == 1 || obstacleGrid[0][0] == 1 {
		return 0
	}

	var arr [][]int
	for i := 0; i < leny; i++ {
		arr = append(arr, make([]int, lenx))
	}
	if lenx == 1 && leny == 1 {
		return 1
	}
	if lenx > 1 {
		arr[0][1] = 1
	}
	if leny > 1 {
		arr[1][0] = 1
	}

	var dp func(y, x int) int
	dp = func(y, x int) int {
		//fmt.Println(arr)
		//fmt.Println(y, x)
		if arr[y][x] != 0 {
			return arr[y][x]
		}

		d := 0
		//边界条件
		up := true
		left := true
		if y == 0 {
			up = false
		}
		if x == 0 {
			left = false
		}
		if up && obstacleGrid[y-1][x] == 1 {
			up = false
		}
		if left && obstacleGrid[y][x-1] == 1 {
			left = false
		}
		if !up && !left {
			return 0
		}
		if up && left {
			d = dp(y-1, x) + dp(y, x-1)
		} else if up {
			d = dp(y-1, x)
		} else {
			d = dp(y, x-1)
		}

		arr[y][x] = d

		return d
	}
	dp(leny-1, lenx-1)
	return arr[leny-1][lenx-1]

}

func uniquePaths(m int, n int) int {
	if m == 1 && n == 1 {
		return 0
	}

	var arr [][]int
	for i := 0; i < m; i++ {
		arr = append(arr, make([]int, n))
	}
	if m > 1 {
		arr[1][0] = 1
	}
	if n > 1 {
		arr[0][1] = 1
	}

	var dp func(y, x int) int
	dp = func(y, x int) int {
		//fmt.Println(arr, y, x)
		if arr[y][x] != 0 {
			return arr[y][x]
		}

		d := 0
		if y == 0 {
			d = dp(y, x-1)
		} else if x == 0 {
			d = dp(y-1, x)
		} else {
			d = dp(y-1, x) + dp(y, x-1)
		}
		arr[y][x] = d

		return d
	}
	dp(m-1, n-1)
	return arr[m-1][n-1]
}

func minCostClimbingStairs(cost []int) int {
	cost = append(cost, 0)
	m := make(map[int]int)
	m[0] = cost[0]
	m[1] = cost[1]

	var dp func(k int) int
	dp = func(k int) int {
		n, ok := m[k]
		if ok {
			return n
		}

		d := min(dp(k-1), dp(k-2)) + cost[k]
		m[k] = d
		return d
	}
	return dp(len(cost) - 1)
}

func louti() {
	m := make(map[int]int)
	m[1] = 1
	m[2] = 2
	var dp func(k int) int
	dp = func(k int) int {
		i, ok := m[k]
		if ok {
			return i
		}
		d := dp(k-2) + dp(k-1)
		m[k] = d

		return d
	}

	fmt.Println(dp(44))
}

func trailingZeroes(n int) int {
	res := 0
	for i := 0; i < n; i++ {
		if i/5 == 0 {
			res++
		}
	}
	return res
}

func findAnagrams(s string, p string) (res []int) {
	if len(p) > len(s) {
		return
	}

	m := make(map[uint8]int)

	for i := 0; i < len(p); i++ {
		m[p[i]]++
	}
	l := len(p)

	wm := make(map[uint8]int)
	for i := 0; i < l; i++ {
		wm[s[i]]++
	}

	checkSame := func(n int) {
		same := true
		for k, v := range m {
			if wm[k] != v {
				same = false
			}
		}
		if same {
			res = append(res, n)
		}
	}

	for i := l; i < len(s); i++ {
		//fmt.Println(wm, m)
		checkSame(i - l)
		wm[s[i]]++
		wm[s[i-l]]--

		//s[i-l : l]
	}

	checkSame(len(s) - l)

	return

}

func maxSlidingWindow(nums []int, k int) (res []int) {
	//优先队列
	var queue []int
	for i := 0; i < len(nums); i++ {
		if len(queue) > 0 && i-k >= 0 && queue[0] == nums[i-k] {
			queue = queue[1:]
		}

		for len(queue) > 0 && queue[len(queue)-1] < nums[i] {
			queue = queue[:len(queue)-1]
		}

		queue = append(queue, nums[i])

		if i >= k-1 {
			res = append(res, queue[0])
		}

		//fmt.Println(queue)
	}
	return
}

func groupAnagrams(strs []string) [][]string {
	m := make(map[string][]string)
	var res [][]string
	for _, str := range strs {
		sbyte := []byte(str)
		sort.Slice(sbyte, func(i, j int) bool {
			return sbyte[i] < sbyte[j]
		})
		//fmt.Println(res, str)
		m[string(sbyte)] = append(m[string(sbyte)], str)
	}
	//fmt.Println(m)
	for _, v := range m {
		res = append(res, v)
	}
	return res
}

func middleNode(head *ListNode) *ListNode {
	var arr []*ListNode
	for head != nil {
		arr = append(arr, head)
		head = head.Next
	}
	return arr[len(arr)/2]
}

func isPalindrome(s string) bool {

	s = strings.ToLower(s)
	s1 := ""
	for i := 0; i < len(s); i++ {
		if (s[i] >= 'a' && s[i] <= 'z') || (s[i] >= '0' && s[i] <= '9') {
			s1 += string(s[i])
		}
	}
	if len(s1) == 0 {
		return true
	}
	//fmt.Println(s1)
	i1, i2 := 0, len(s1)-1
	for i2 > i1 {
		//fmt.Println(i1, i2)
		if s1[i1] != s1[i2] {
			return false
		}
		i1++
		i2--
	}

	//fmt.Println(s1)
	return true
}
func imageSmoother(img [][]int) [][]int {
	var res [][]int

	leny := len(img) - 1
	lenx := len(img[0]) - 1
	for y, ints := range img {
		arr := make([]int, lenx+1)
		for x := range ints {
			count := 0
			sum := 0
			for i := 0; i < 9; i++ {
				mx := x + i%3 - 1
				my := y + i/3 - 1
				//fmt.Println(my, mx)
				if mx > lenx || mx < 0 || my > leny || my < 0 {
					continue
				}
				count++
				sum += img[my][mx]
				//fmt.Println(my, mx, img[my][mx])
			}
			arr[x] = sum
		}
		res = append(res, arr)
	}
	//fmt.Println(res)
	return res
}
func mergeKLists(lists []*ListNode) *ListNode {
	pre := &ListNode{}
	head := pre
	for {
		m := 100000
		mIdx := -1
		for i, list := range lists {
			fmt.Println(list)
			if list == nil {
				continue
			}
			if list.Val < m {
				m = list.Val
				mIdx = i
			}
		}
		if mIdx != -1 {
			lists[mIdx] = lists[mIdx].Next

		}

		if m == 1000000 {
			return head.Next
		}
		pre.Next = &ListNode{
			Val: m,
		}
		pre = pre.Next

	}
}
func preorderTraversal(root *TreeNode) (res []int) {
	if root == nil {
		return
	}

	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		res = append(res, node.Val)

		dfs(node.Left)
		dfs(node.Right)
	}
	dfs(root)
	return

}

func diameterOfBinaryTree(root *TreeNode) int {
	res := 1

	var dfs func(node *TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}

		l := dfs(node.Left)
		r := dfs(node.Right)

		max := 0
		if l > r {
			max = l
		} else {
			max = r
		}
		if l+r+1 > res {
			res = max
		}
		return max + 1
	}
	dfs(root)
	return res - 1
}

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	pre := &ListNode{}
	head := pre
	jin := false
	for {
		if l1 == nil && l2 == nil {
			if jin {
				pre.Next = &ListNode{
					Val: 1,
				}
			}
			return head.Next
		}
		n1 := 0
		n2 := 0
		if l1 != nil {
			n1 = l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			n2 = l2.Val
			l2 = l2.Next
		}
		if jin {
			n1++
		}
		n := n1 + n2
		if n >= 10 {
			jin = true
			n -= 10
		} else {
			jin = false
		}
		pre.Next = &ListNode{
			Val: n,
		}
		pre = pre.Next
	}
}

func findKthNumber(n int, k int) int {

	//var arr []int
	var dfs func(k int)
	prefix := 10

	dfs = func(k int) {
		num := prefix + k
		fmt.Println(num)
		if num < n {
			return
		}

		for i := 1; i <= 10; i++ {
			prefix = k * prefix
			dfs(i)
			prefix = prefix / k
		}
	}
	for i := 0; i < 10; i++ {
		dfs(i)
	}
	return 10
}

type MyLinkedList struct {
	val  int
	next *MyLinkedList
	head *MyLinkedList
}

func Constructor() MyLinkedList {
	l := MyLinkedList{}
	l.head = &l
	return l
}

func (this *MyLinkedList) Get(index int) int {
	return 0
}

func (this *MyLinkedList) AddAtHead(val int) {
	l := MyLinkedList{
		val:  val,
		next: this.head,
	}
	this.head = &l
}

func (this *MyLinkedList) AddAtTail(val int) {
	fmt.Println(this.head)
	temp := this.head
	for temp.next != nil {
		temp = temp.next
	}
	temp.next = &MyLinkedList{
		val: val,
	}
}

func (this *MyLinkedList) AddAtIndex(index int, val int) {

}

func (this *MyLinkedList) DeleteAtIndex(index int) {

}

func wordBreak(s string, wordDict []string) bool {
	sort.Strings(wordDict)
	for _, s2 := range wordDict {
		s = strings.ReplaceAll(s, s2, "")
		fmt.Println(s)
	}
	return len(s) == 0
}

func getIntersectionNode(headA, headB *ListNode) *ListNode {
	addMap := make(map[*ListNode]bool)
	for {
		if addMap[headA] == true {
			return headA
		}
		if addMap[headB] == true {
			return headB
		}
		if headB == headA {
			return headB
		}

		if headA != nil {
			addMap[headA] = true
			headA = headA.Next
		}
		if headB != nil {
			addMap[headB] = true
			headB = headB.Next
		}

		if headB == nil && headA == nil {
			return nil
		}

	}
}

func deleteNode(node *ListNode) {
	for {
		node.Val = node.Next.Val
		if node.Next.Next == nil {
			node.Next = nil
			return
		}
		node = node.Next
	}
}

func firstUniqChar(s string) int {
	m := make(map[uint8]int, 0)
	for i := 0; i < len(s); i++ {
		m[s[i]]++
	}
	for i := 0; i < len(s); i++ {
		//fmt.Println(m, s[i])
		if m[s[i]] == 1 {
			return i
		}
	}
	return -1
}
func fizzBuzz(n int) []string {
	var res []string
	for i := 0; i < n; i++ {
		if i%3 == 0 && i%5 == 0 {
			res = append(res, "FizzBuzz")
			continue
		}
		if i%3 == 0 {
			res = append(res, "Fizz")
			continue
		}
		if i%5 == 0 {
			res = append(res, "Buzz")
			continue
		}
		res = append(res, strconv.Itoa(i+1))
	}
	return res
}

func flatten(root *TreeNode) {
	if root == nil {
		return
	}
	var queue []*TreeNode

	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		queue = append(queue, node)
		fmt.Println(node.Val)
		dfs(node.Left)
		dfs(node.Right)
	}
	dfs(root)
	for i := 0; i < len(queue); i++ {
		queue[i].Left = nil
		queue[i].Right = queue[i+1]
	}

}
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	nums1 = append(nums1, nums2...)
	sort.Ints(nums1)
	l := len(nums1)
	fmt.Println(nums1)
	if l%2 == 0 {
		return float64(nums1[l/2])
	} else {
		fmt.Println(nums1[l/2], nums1[l/2+1], float64(nums1[l/2]+nums1[l/2+1]))

		return float64(nums1[l/2]+nums1[l/2+1]) / 2
	}
}

func isMatch(s string, p string) bool {
	return regexp.MustCompile(`^` + p + `$`).MatchString(s)
}

func winnerOfGame(colors string) bool {
	i1 := 0
	acount := 0
	bcount := 0

	if len(colors) < 3 {
		return false
	}
	a := true
	if colors[0] != 'A' {
		a = false
	}

	for i2 := 1; i2 < len(colors); i2++ {
		var b uint8
		if a {
			b = 'A'
		} else {
			b = 'B'
		}
		//fmt.Println(i1, i2, b, colors[i1:i2], colors[i2])
		if colors[i2] != b {
			i1 = i2
			a = !a
		} else {
			if i2-i1 > 1 {
				if a {
					acount++
				} else {
					bcount++
				}
			}
		}
	}
	fmt.Println(acount, bcount)
	return acount > bcount
}

func multiply(num1 string, num2 string) string {
	if num1 == "0" || num2 == "0" {
		return "0"
	}

	if len(num1) < len(num2) {
		num1, num2 = num2, num1
	}

	//fmt.Println(num1[1])

	z := 0
	sum := ""
	for i1 := len(num2) - 1; i1 >= 0; i1-- {
		i3 := 0
		line := ""
		for i2 := len(num1) - 1; i2 >= 0; i2-- {
			i3 = int((num2[i1]-48)*(num1[i2]-48)) + i3
			//fmt.Println(i3, num2[i1]-48, num1[i2]-48)
			if i3 >= 10 {
				line = strconv.Itoa(i3%10) + line
			} else {
				line = strconv.Itoa(i3) + line
			}
			i3 = i3 / 10
			//fmt.Println(i3, num2[i1]-48, num1[i2]-48)
		}

		if i3%10 != 0 {
			line = strconv.Itoa(i3%10) + line
		}

		for iz := 0; iz < z; iz++ {
			line = line + "0"
		}

		tempSum := ""
		l1 := len(sum)
		cha := len(line) - l1
		//fmt.Println(cha)
		k := 0
		jin := false
		for i := len(line) - 1; i >= 0; i-- {
			t := 0
			if jin {
				t++
				jin = false
			}

			if k < len(sum) {
				t += int((line[i] - 48) + sum[i-cha] - 48)
			} else {
				t += int(line[i] - 48)
			}

			if t >= 10 {
				jin = true
				t = t - 10
			}
			tempSum = strconv.Itoa(t) + tempSum

			//tempSum = strconv.Itoa() + tempSum

			//fmt.Println(sum, line, tempSum)
			k++
		}
		if jin {
			tempSum = "1" + tempSum
		}
		sum = tempSum

		fmt.Println(line, tempSum)
		z++
	}

	return sum
}

func combinationSum(candidates []int, target int) (ans [][]int) {
	comb := []int{}
	var dfs func(target, idx int)
	dfs = func(target, idx int) {
		if idx == len(candidates) {
			return
		}
		if target == 0 {
			ans = append(ans, append([]int(nil), comb...))
			return
		}
		fmt.Println(comb)
		// 选择当前数
		if target-candidates[idx] >= 0 {
			comb = append(comb, candidates[idx])
			dfs(target-candidates[idx], idx)
			comb = comb[:len(comb)-1]
		}
		// 直接跳过
		dfs(target, idx+1)

	}
	dfs(target, 0)
	return
}

func getRow(rowIndex int) []int {
	res := []int{1}
	for i := 0; i < rowIndex; i++ {
		temp := make([]int, len(res))
		copy(temp, res)
		for k := 1; k < i+1; k++ {
			res[k] = temp[k-1] + temp[k]
			//	fmt.Println(res, k)
		}
		res = append(res, 1)
		//fmt.Println(res)
	}

	return res
}

func hasPathSum(root *TreeNode, targetSum int) bool {
	res := false
	num := 0
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		if num == targetSum && node.Left == nil && node.Right == nil {
			res = true
			return
		}

		num += node.Val
		dfs(node.Left)
		num -= node.Val

		num += node.Val

		dfs(node.Right)
		num -= node.Val

	}
	dfs(root)
	return res
}

func findTarget(root *TreeNode, k int) bool {
	res := false
	m := make(map[int]bool)
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		if res {
			return
		}
		_, ok := m[k-node.Val]
		if ok {
			res = true
			return
		}
		m[root.Val] = true

		dfs(node.Left)
		dfs(node.Right)
	}
	dfs(root)
	return res
}

func minDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	res := 0
	var queue []*TreeNode
	queue = append(queue, root)
	f := 1
	fmt.Println(queue)
	for len(queue) > 0 {
		l := len(queue)
		for i := 0; i < l; i++ {
			node := queue[0]
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
			if node.Left == nil && node.Right == nil {
				res = f
				return res
			}
			f++

		}
	}
	return res
}

type MinStack struct {
	arr []int
	min int
}

func (this *MinStack) Push(val int) {
	if val < this.min {
		this.min = val
	}
	this.arr = append(this.arr, val)
}

func (this *MinStack) Top() int {

	return this.arr[len(this.arr)]

}

func (this *MinStack) GetMin() int {
	return this.min
}

//53. 最大子数组和
func maxSubArray(nums []int) int {

	max := -999999

	for i := 1; i <= len(nums); i++ {

		for j := 0; j < len(nums)+1-i; j++ {
			sum := 0
			for k := j; k < i+j; k++ {
				sum += nums[k]
			}
			if sum > max {
				max = sum
			}
			//fmt.Println(sum, j, i+j)
		}

	}
	return max
}

//155. 最小栈

//100. 相同的树
func isSameTree(p *TreeNode, q *TreeNode) bool {
	if p == nil && q == nil {
		return true
	}
	if p != nil && q == nil {
		return false
	}
	if p == nil && q != nil {
		return false
	}

	if p.Val != q.Val {
		return false
	}

	t1 := isSameTree(p.Left, q.Left)
	t2 := isSameTree(p.Right, q.Right)
	if t1 && t2 {
		return true
	}
	return false

}

func rotate(nums []int, k int) {

	numsB := make([]int, len(nums))
	copy(numsB, nums)

	for k > len(nums) {
		k -= len(nums)
	}
	//fmt.Println(k)
	j := len(nums) - k

	for i := 0; i < len(nums); i++ {
		if j == len(nums) {
			j = 0
		}
		nums[i] = numsB[j]
		j++
	}
}

func orchestraLayout(num int, xPos int, yPos int) int {
	//注意,这里我把原点变为了(1,1),而不是(0,0)
	i := xPos + 1
	j := yPos + 1
	n := num
	mi := min(i, min(j, min(n-i+1, n-j+1)))
	var ans int
	if i <= j {
		ans = mi*(4*n-4*mi) + 6*mi - 4*n - 3 + i + j
	} else {
		ans = mi*(4*n-4*mi) + 2*mi + 1 - i - j
	} //模拟过程
	if ans%9 == 0 {
		return 9
	} else {
		return ans % 9
	}

}

func romanToInt(s string) int {
	res := 0
	l := len(s)
	for i := 0; i < l; i++ {
		sw := func() {
			switch s[i] {
			case 'I':
				res += 1
			case 'V':
				res += 5
			case 'X':
				res += 10
			case 'L':
				res += 50
			case 'C':
				res += 100
			case 'D':
				res += 500
			case 'M':
				res += 1000
			}
		}

		if i < l-1 {
			fmt.Println(s[i : i+2])
			switch s[i : i+2] {
			case "IV":
				res += 4
			case "IX":
				res += 9
			case "XL":
				res += 40
			case "XC":
				res += 90
			case "CD":
				res += 400
			case "CM":
				res += 900
			default:
				sw()
				i--
			}
			i++
		} else {
			sw()
		}

	}
	return res
}

type ListNode struct {
	Val  int
	Next *ListNode
}

func reverseList(head *ListNode) *ListNode {
	curr := head
	var last *ListNode
	for curr != nil {
		next := curr.Next
		curr.Next = last
		last = curr
		curr = next
	}
	return curr
}

func minimumTotal(triangle [][]int) int {
	l := len(triangle)
	res := 0
	//var arr []int
	for i := l; i > 1; i-- {
		//arr = []int{}
		for k := 0; k < len(triangle[i])-1; k++ {
			triangle[i-1][k] = triangle[i-1][k] + min(triangle[i][k], triangle[i][k+1])
			//arr = append(arr, min(triangle[i][k], triangle[i][k+1]))
		}
		fmt.Println(triangle)
	}
	return res
}

func tree2str(root *TreeNode) string {
	if root == nil {
		return ""
	}
	if root.Left == nil && root.Right == nil {
		return strconv.Itoa(root.Val)
	}
	if root.Left != nil && root.Right == nil {
		return strconv.Itoa(root.Val) + "(" + tree2str(root.Left) + ")"
	}
	return strconv.Itoa(root.Val) + "(" + tree2str(root.Left) + ")(" + tree2str(root.Right) + ")"

}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func isSymmetric(root *TreeNode) bool {
	if root == nil {
		return true
	}
	l := root
	r := root

	res := true
	var dfs func(n1, n2 *TreeNode)
	dfs = func(n1, n2 *TreeNode) {
		if n1 == nil && n2 != nil {
			res = false
			return
		}
		if n1 != nil && n2 == nil {
			res = false
			return
		}
		if n1 == nil && n2 == nil {
			return
		}
		if n1.Val != n2.Val {
			res = false
			return
		}
		mutex := sync.Mutex{}
		mutex.Lock()
		dfs(n1.Left, n2.Right)
		dfs(n1.Right, n2.Left)

	}
	dfs(l, r)
	return res
}

func getLeastNumbers(arr []int, k int) []int {
	sort.Ints(arr)
	res := make([]int, k)
	res = arr[:k]
	return res
}

type LRUCache struct {
	queue []int
	size  int
	cache map[int]int
}

func (this *LRUCache) Get(key int) int {
	//	fmt.Println(this.queue)
	v, ok := this.cache[key]
	if !ok {
		return -1
	}

	for i, i1 := range this.queue {
		if key == i1 {
			this.queue = append(this.queue[:i], this.queue[i+1:]...)
			break
		}
	}
	this.queue = append(this.queue, key)
	//	fmt.Println(this.queue)

	return v
}

func (this *LRUCache) Put(key int, value int) {
	_, ok := this.cache[key]
	if ok {
		for i, i1 := range this.queue {
			if key == i1 {
				this.queue = append(this.queue[:i], this.queue[i+1:]...)
				break
			}
		}
	}

	if len(this.queue) == this.size {
		delete(this.cache, this.queue[0])
		this.queue = this.queue[1:]
	}
	this.queue = append(this.queue, key)
	this.cache[key] = value

}

func kaochang(n int, arr [][]int) {
	res := 0
	var resPeople []int
	for len(arr) > 0 {
		l := len(arr)
		manCount := make(map[int]int)
		womanCount := make(map[int]int)

		for i := 0; i < l; i++ {
			manCount[arr[i][0]]++
			womanCount[arr[i][1]]++
		}
		maxMan := 0
		maxManId := 0
		maxWoman := 0
		maxWomanId := 0
		for k, v := range manCount {
			if v > maxMan {
				maxMan = v
				maxManId = k
			}
		}
		for k, v := range womanCount {
			if v > maxWoman {
				maxWoman = v
				maxWomanId = k
			}
		}
		if maxMan >= maxWoman {
			for i := 0; i < len(arr); i++ {
				if arr[i][0] == maxManId {
					//fmt.Println(arr[i])
					arr = append(arr[:i], arr[i+1:]...)
					i--
				}
			}
			resPeople = append(resPeople, maxManId)
		} else {
			for i := 0; i < len(arr); i++ {
				if arr[i][1] == maxWomanId {
					arr = append(arr[:i], arr[i+1:]...)
					i--
				}
			}
			resPeople = append(resPeople, maxWomanId)
		}
		res++
	}
	fmt.Println(res)
	for _, person := range resPeople {
		fmt.Print(person)
	}

	//fmt.Println(arr, maxManId, maxWomanId, maxMan, maxWoman)

}

func hechang(arr []int) {

	//for i := 0; i < l; i++ {
	//	k := i
	//	var temp []int
	//	temp = append(temp, arr[i])
	//	if i < l-1 && arr[i] > arr[i+1] {
	//		for i < l-1 && arr[i] >= arr[i+1] {
	//			temp = append(temp, arr[i+1])
	//			//fmt.Println(arr[i], arr[i+1])
	//			i++
	//		}
	//	}
	//	for i > k && arr[i] == arr[i-1] {
	//		temp = temp[:len(temp)-1]
	//		//fmt.Println(i)
	//		i--
	//	}
	//	//fmt.Println(temp)
	//
	//	res++
	//
	//	//fmt.Println("go", temp)
	//}
	//fmt.Println(res)
}

func three(arr [][]int64) {
	res := 0
	for i := 0; i < len(arr); i++ {
		a, b, c := arr[i][0], arr[i][1], arr[i][2]
		for k := 0; k < len(arr); k++ {
			if k == i {
				continue
			}
			a1, b1, c1 := arr[k][0], arr[k][1], arr[k][2]
			if a < a1 && b < b1 && c < c1 {
				//fmt.Println(a, b, c, "  ", a1, b1, c1)
				res++
			}
		}
	}
	fmt.Println(res)
}
func threeSum(nums []int) (res [][]int) {
	sort.Ints(nums)
	fmt.Println(nums)
	for i := 0; i < len(nums)-2; i++ {
		n := nums[i]
		if n > 0 {
			break
		}

		if i > 0 && nums[i] == nums[i-1] {
			continue
		}

		lp, rp := i+1, len(nums)-1

		for lp < rp {
			l := nums[lp]
			r := nums[rp]
			fmt.Println(n, l, r)

			sum := n + l + r
			if sum == 0 {
				res = append(res, []int{n, l, r})
				for lp < rp && nums[lp] == nums[lp+1] {
					lp++
				}
				for lp < rp && nums[rp] == nums[rp-1] {
					rp--
				}
				lp++
				rp--
			} else if sum < 0 {
				lp++
			} else {
				rp--
			}
		}

	}
	return
}

func selfDividingNumbers(left int, right int) (res []int) {
	checkNumber := func(num int) bool {
		n := num
		for num > 0 {
			//fmt.Println(num % 10)
			b := num % 10
			if b == 0 {
				return false
			}
			if n%b != 0 {
				return false
			}
			num /= 10
		}
		return true
	}

	for i := left; i <= right; i++ {
		if checkNumber(i) {
			res = append(res, i)
		}
	}
	return
}

type server struct {
	task      int
	countDown int
}

func busiestServers(k int, arrival []int, load []int) []int {
	servers := make([]server, k)
	for t := 1; ; t++ {
		done := true
		for i, s := range servers {
			if s.countDown >= 1 {
				done = false
				if s.countDown == 1 {
					servers[i].task++
				}
				servers[i].countDown--
			}
		}
		if t != 1 && done {
			break
		}

		for i, at := range arrival {
			if t == at {
				for i2, s := range servers {
					if s.countDown == 0 {
						servers[i2].countDown += load[i]
						break
					}
				}
				break
			}
		}
		fmt.Println(servers)
	}
	var res []int
	m := 0
	for _, s := range servers {
		if s.task > m {
			m = s.task
		}
	}
	for i, s := range servers {
		if s.task == m {
			res = append(res, i)
		}
	}
	return res
}

//双指针+贪心
func core(s string) {
	arr := []uint8{'A', 'B', 'C'}
	lens := [][]int{{0, 0}, {0, 0}, {0, 0}}
	maxLen := 0
	//maxB := 0

	//获取abc距离最远值
	for is, b := range arr {
		var front, end int
		for i := 0; i < len(s); i++ {
			if b == s[i] {
				front = i
				break
			}
		}
		for i := len(s) - 1; i >= 0; i-- {
			if b == s[i] {
				end = i
				break
			}
		}
		l := end - front
		if l > maxLen {
			//maxB = is
			maxLen = l
		}
		lens[is] = []int{end, front}
		//fmt.Println(front, end, l)
	}

	//fmt.Println(maxB, maxLen, lens)

}

func ti2() {
	var m int
	fmt.Scan(&m)
	for i := 0; i < m; i++ {
		var num int
		fmt.Scan(&num)
		countNum(num)
	}
}

func countNum(num int) {

	n := int(math.Sqrt(float64(num)))
	//fmt.Println(n)
	res := 1
	for i := 1; i <= n; i++ {
		if num%i == 0 {
			k := num / i
			for j := 1; j <= k; j++ {
				if j%i == 0 && k%i == 0 {
					//fmt.Println(j, k)
					goto this
				}
			}
			res++
		this:
			//fmt.Println(i, num/i)
		}
	}
	fmt.Println(res)
}

func reverse(i int) int {
	s := strconv.Itoa(i)
	s1 := ""
	for k := len(s) - 1; k >= 0; k-- {
		s1 += string(s[k])
	}
	res, _ := strconv.Atoi(s1)
	fmt.Println(res)
	return res
}

func lastStoneWeight(stones []int) int {
	for len(stones) <= 1 {
		sort.Ints(stones)
		l := len(stones)
		a := stones[l-1] - stones[l-2]
		stones = stones[:l-2]
		if a != 0 {
			stones = append(stones, a)
		}
		fmt.Println(stones)
	}
	if len(stones) == 1 {
		return stones[0]
	} else {
		return 0
	}
}

func countSegments(s string) int {
	if len(s) == 0 {
		return 0
	}
	res := 0
	for i := 0; i < len(s); i++ {
		if s[i] != ' ' {
			for i < len(s) && s[i] != ' ' {
				i++
			}
			res++
		}
	}
	return res
}

func flipAndInvertImage(image [][]int) [][]int {
	//ly := len(image)
	lx := len(image[0]) - 1

	for y := range image {
		var k int
		if lx%2 == 0 {
			k = lx/2 + 1
		} else {
			k = lx / 2
		}
		for x := 0; x < k; x++ {
			image[y][x], image[y][lx-x] = image[y][lx-x], image[y][x]
			fmt.Println(image[y], lx, x, k)
		}
	}
	for y := range image {
		for x, i := range image[y] {
			if i == 1 {
				image[y][x] = 0
			} else {
				image[y][x] = 1
			}
		}
	}
	return image
}

func longestOnes(nums []int, k int) int {
	c := k
	j := 0
	m := 0
	for i := 0; i < len(nums); i++ {
		if nums[i] == 0 {
			if c > 0 {
				c--
			} else {
				//fmt.Println("fuck")
				if nums[j] == 0 {
					j++
				} else {
					for nums[j] != 0 {
						j++
					}
					j++
				}
			}
		}
		m1 := i - j + 1
		if m1 > m {
			m = m1
		}
		fmt.Println(nums[j:i], c, j, i)
	}
	return m
}
func run(arr []int) {
	min := math.MaxInt32
	for _, i := range arr {
		if i < min {
			min = i
		}
	}
	all := 0
	for _, i := range arr {
		all += i - min
	}
	fmt.Println(all)
	if all%2 == 0 {
		fmt.Println("woman")
	} else {
		fmt.Println("man")
	}

}

func canConstruct(ransomNote string, magazine string) bool {
	if len(ransomNote) > len(magazine) {
		return false
	}
	m := make(map[uint8]int)
	for i := 0; i < len(magazine); i++ {
		m[magazine[i]]++
	}
	for i := 0; i < len(ransomNote); i++ {
		count, ok := m[ransomNote[i]]

		if !ok || count == 0 {
			return false
		}
		m[ransomNote[i]]--
	}
	return true
}
func maxSubArray(nums []int) int {
	l := len(nums)
	arr := make([]int, l)
	arr[0] = nums[0]
	for i := 1; i < l; i++ {
		arr[i] = max(nums[i], arr[i-1]+nums[i])
	}
	m := math.MinInt32
	for _, i := range arr {
		if i > m {
			m = i
		}
	}
	return m
}
func max(i, j int) int {
	if i > j {

		return i
	}
	return j
}

func findContentChildren(g []int, s []int) int {
	sort.Ints(g)
	sort.Ints(s)
	j := 0
	i := 0
	ls := len(s)
	lg := len(g)
	if ls == 0 {
		return 0
	}
	for j <= ls-1 && i <= lg-1 {
		for g[i] <= s[j] {
			j++
		}
	}
	return len(g)
}
func firstMissingPositive(nums []int) int {
	sort.Ints(nums)
	first := true
	i := 1
	for _, num := range nums {
		if num > 0 {
			if !first {
				fmt.Println(num, i)

				if num == i {
					continue
				} else if num == i+1 {
					i++
					continue
				} else {
					return i + 1
				}
			} else {
				if num != 1 {
					return 1
				}
				first = false
			}
		}
	}
	if first {
		return i
	}
	return i + 1
	//m := make(map[int]bool)
	//for _, num := range nums {
	//	if num > 0 {
	//		m[num] = true
	//	}
	//}
	//for i := 1; ; i++ {
	//	_, ok := m[i]
	//	if !ok {
	//		return i
	//	}
	//}
}
func longestValidParentheses(s string) int {
	var stack []uint8
	res := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '(' {
			stack = append(stack, '(')
		} else {
			if len(stack) != 0 {
				res++
				stack = stack[:len(stack)-1]
			} else {
				res = 0
			}
		}
		fmt.Println(stack)
	}
	return res * 2

}

func search(nums []int, target int) int {
	for i, num := range nums {
		if num == target {
			return i
		}
	}
	return -1

	//k := -1
	//for i := 0; i < len(nums)-1; i++ {
	//	if nums[i] > nums[i+1] {
	//		k = i
	//		break
	//	}
	//}
	//sort.Ints(nums)
	//i := sort.SearchInts(nums, target)
	//fmt.Println(nums, k, i)
	//if i == -1 {
	//	return -1
	//}
	//fmt.Println(k)
}
