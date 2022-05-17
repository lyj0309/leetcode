package past

import (
	"fmt"
	"math"
	"math/bits"
	"math/rand"
	"sort"
	"strconv"
	"strings"
	"sync"
)

type Solution struct {
	m map[int][]int
}

func Constructor(nums []int) Solution {
	m := make(map[int][]int)
	for i, num := range nums {
		m[num] = append(m[num], i)
	}
	return Solution{m: m}
}

func (this *Solution) Pick(target int) int {

	return this.m[target][rand.Int()%len(this.m[target])]
}

func solve(a []*ListNode) *ListNode {
	var arr []int

	node1 := a[0]
	for node1 != nil {
		arr = append(arr, node1.Val)
		node1 = node1.Next
	}

	for i, node := range a {
		if i == 0 {
			continue
		}

		//var temp []int
		app := false
		for node != nil {
			if app {
				arr = append(arr, node.Val)
			}
			if node.Val == arr[len(arr)-1] {
				app = true
			}
			//temp = append(temp, node.Val)
			node = node.Next
		}
	}
	m := make(map[int]bool)
	min := 9999999999
	minIdx := -1
	for k, i := range arr {
		if m[i] == true {
			arr = arr[:k]
			break
		}
		if i < min {
			min = i
			minIdx = k
		}
		m[i] = true
	}

	//fmt.Println(arr, min, minIdx)
	var minarr1 []int
	for i := minIdx; i < len(arr); i++ {
		minarr1 = append(minarr1, arr[i])
	}
	for i := 0; i < i; i++ {
		minarr1 = append(minarr1, arr[i])
	}
	var minarr2 []int
	for i := minIdx; i >= 0; i-- {
		minarr2 = append(minarr2, arr[i])
	}
	for i := len(arr) - 1; i > minIdx; i-- {
		minarr2 = append(minarr2, arr[i])
	}
	//fmt.Println(minarr1, minarr2)

	minarr := minarr1
	for i := 0; i < len(minarr1); i++ {
		if minarr1[i] < minarr2[i] {
			break
		} else if minarr1[i] > minarr2[i] {
			minarr = minarr2
			break
		}
	}

	minNode := &ListNode{}
	head := minNode
	//fmt.Println(minarr)
	for _, i := range minarr {
		minNode.Next = &ListNode{
			Val: i,
		}
		minNode = minNode.Next
	}
	//c := head
	//for c != nil {
	//	fmt.Println(c.Val)
	//	c = c.Next
	//}

	return head.Next
}
func gongfang(arr string) {
	l := len(arr)
	//fmt.Println(arr)
	min := math.MaxInt32
	for i := 0; i <= l; i++ {
		gsum := 0
		fsum := 0
		for i1 := 0; i1 < i; i1++ {
			if arr[i1] == '0' {
				gsum += i1 + 1
			}
		}
		for i1 := i; i1 < l; i1++ {
			if arr[i1] == '1' {
				fsum += i1 + 1
			}
		}
		n := abs(gsum - fsum)
		if n < min {
			min = n
		}

		//fmt.Println(gsum, fsum)

	}
	fmt.Println(min)
}
func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

//getNumber([]int{3, 1, 1, 4, 5, 6})

func getNumber(a []int) int {
	zhimap := make(map[int]bool)

	iszhi := func(k int) bool {
		if k == 1 {
			return false
		}
		for i := 2; i < k; i++ {
			if k%i == 0 {
				return false
			}
		}
		return true
	}
	l := len(a)
	for i := 0; i < l; i++ {
		if iszhi(i + 1) {
			zhimap[i+1] = true
		}
	}

	for len(a) != 1 {
		var del []int
		for i := 0; i < len(a); i++ {
			if !zhimap[i+1] {
				del = append(del, i)
			}
		}
		suffix := 0
		//fmt.Println(del)
		for _, i := range del {
			a = append(a[:i+suffix], a[i+1+suffix:]...)
			suffix--
		}
		//fmt.Println(a)

	}
	//fmt.Println(a[0])

	return a[0]

}

func shudu(strs []string) {
	ly := len(strs)
	lx := len(strs[0])

	//fmt.Println(strs)
	var arrs []int
	for x := 0; x < lx; x++ {
		s := ""
		for y := 0; y < ly; y++ {
			s += string(strs[y][x])
		}
		i, _ := strconv.Atoi(s)
		arrs = append(arrs, i)
	}
	sort.Ints(arrs)
	for _, arr := range arrs {
		fmt.Print(arr, " ")
	}
}

//k 0,分，1，时
func parseCorn(s string, k int) (ans map[string]bool) {
	ans = make(map[string]bool)

	var sumTime []int
	switch k {
	case 0:
		sumTime = []int{0, 59}
	case 1:
		sumTime = []int{0, 23}
	case 2:
		sumTime = []int{1, 31}
	case 3:
		sumTime = []int{1, 12}
	case 4:
		return

	}
	fmt.Println(sumTime)

	if strings.Index(s, ",") != -1 {
		arr := strings.Split(s, ",")
		for _, i2 := range arr {
			ans[i2] = true
		}

	} else if strings.Index(s, "-") != -1 {
		sarr := strings.Split(s, "-")
		start, _ := strconv.Atoi(sarr[0])
		end, _ := strconv.Atoi(sarr[1])
		for i := start; i <= end; i++ {
			ans[strconv.Itoa(i)] = true
		}

	} else if strings.Index(s, "/") != -1 {
		sarr := strings.Split(s, "/")
		interval, _ := strconv.Atoi(sarr[1])
		if strings.Index(s, "*") != -1 {
			for i := sumTime[0]; i <= sumTime[1]; i += interval {
				ans[strconv.Itoa(i)] = true

			}
		} else {
			sarr := strings.Split(s, "-")
			start, _ := strconv.Atoi(sarr[0])
			end, _ := strconv.Atoi(sarr[1])
			for i := start; i <= end; i += interval {
				ans[strconv.Itoa(i)] = true

			}
		}
	} else if strings.Index(s, "*") != -1 {
		for i := sumTime[0]; i <= sumTime[1]; i++ {
			ans[strconv.Itoa(i)] = true

		}
	} else {
		ans[s] = true

	}
	return ans
}

func toGoatLatin(sentence string) string {
	arr := strings.Split(sentence, " ")
	for i, s := range arr {
		if s[0] == 'a' || s[0] == 'e' || s[0] == 'i' || s[0] == 'o' || s[0] == 'u' ||
			s[0] == 'A' || s[0] == 'E' || s[0] == 'I' || s[0] == 'O' || s[0] == 'U' {
			arr[i] += "ma"
		} else {
			arr[i] = s[1:] + s[:1] + "ma"
		}
		for i2 := 0; i2 <= i; i2++ {
			arr[i] += "a"
		}
	}
	fmt.Println(arr)

	res := ""
	for _, s := range arr {
		res += s
	}
	return res
}

type IntHeap []int

func (h *IntHeap) Push(x any) {
	p := *h
	p = append(p, x.(int))
}

func (h *IntHeap) Pop() any {
	p := *h
	l := len(p) - 1
	r := p[l]
	*h = p[:l]
	return r
}

func (h IntHeap) Len() int {
	return len(h)
}

func (h IntHeap) Less(i, j int) bool {
	return h[i] > h[j]
}

func (h IntHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func ti2() {
	worktime([]int{1, 2, 3, 1, 2, 3, 1}, 8)
	return
	var n, m int
	fmt.Scan(&n, &m)
	workTime := make([]int, n)
	for i := 0; i < n; i++ {
		fmt.Scan(&workTime[i])
	}
	for i := 0; i < m; i++ {
		var t int
		fmt.Scan(&t)
		worktime(workTime, t)
	}

}

func worktime(arr []int, t int) {
	//fmt.Println(arr)
	l := len(arr)
	res := 0
	for i := 0; i < l; i++ {
		if t-arr[i] <= 0 {
			fmt.Println(res)
			return
		}
		t -= arr[i]
		res++
	}
	fmt.Println(res)
}

func ti3() {
	zhezhi([][]int{{2, 6, 5, 4}, {1, 5, 7, 6}, {9, 8, 8, 7}, {1, 4, 7, 8}})
	//zhezhi([][]int{{1, 2}, {3, 4}})
	return
	var n int
	fmt.Scan(&n)
	var arr [][]int
	for i := 0; i < n; i++ {
		arr = append(arr, make([]int, n))
	}

	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			fmt.Scan(&arr[y][x])
		}

	}
	zhezhi(arr)
}

func zhezhi(arr [][]int) {
	l := len(arr)
	m := 0
	t := 1
	//算出2的多少次
	for t != l {
		t *= 2
		m++
	}

	fmt.Println(m)
	for t := 0; t <= m; t++ {
		for y := 0; y < l; y++ {
			k := l - 1
			for x := 0; x < l/2; x++ {
				arr[y][x] += arr[y][k]
				k--
			}
		}
		fmt.Println(arr, l)

		for y := 0; y < l/2; y++ {
			k := l - 1
			for x := 0; x < l/2; x++ {
				arr[y][x] += arr[k][x]
			}
			k = -1
			fmt.Println(arr, l)

		}

		l /= 2
	}
	fmt.Println(arr[0][0])

}

//典型比较233 ，23 | 21，2| 21，22
func compare(i1, i2 int) bool {
	s1 := strconv.Itoa(i1)
	s2 := strconv.Itoa(i2)

	b1 := s1[0]
	b2 := s2[0]
	l1 := len(s1)
	l2 := len(s2)
	if l1 < l2 {
		for j := 0; j < l2; j++ {
			var temp uint8
			if j > l1-1 {
				temp = b1
			} else {
				temp = s1[j]
			}

			if temp == s2[j] {
				continue
			}
			return temp > s2[j]
		}
	} else if l1 > l2 {
		for j := 0; j < l1; j++ {
			var temp uint8
			if j > l2-1 {
				temp = b2
			} else {
				temp = s1[j]
			}

			if temp == s1[j] {
				continue
			}
			return temp < s1[j]
		}
	} else {
		return i1 > i2
	}
	return false
}

//l 100 max 10000
//func solve(nums []int) string {
//	res := ""
//
//	//sort.Slice(nums, func(i, j int) bool {
//	//	return compare(nums[i], nums[j])
//	//})
//	//fmt.Println(nums)
//
//	//冒泡
//	for i := 0; i < len(nums); i++ {
//		for k := 0; k < len(nums)-1-i; k++ {
//			if !compare(nums[k], nums[k+1]) {
//				nums[k], nums[k+1] = nums[k+1], nums[k]
//			}
//		}
//		//fmt.Println(nums)
//	}
//
//	for _, num := range nums {
//		res += strconv.Itoa(num)
//	}
//	//fmt.Println(res)
//	return res
//
//}

func maximumWealth(accounts [][]int) int {
	max := 0
	var wg sync.WaitGroup
	for _, account := range accounts {
		wg.Add(1)
		account := account
		go func() {
			defer wg.Done()
			sum := 0
			for _, i := range account {
				sum += i
			}
			if sum > max {
				max = sum
			}
		}()
	}
	wg.Wait()
	return max
}

type RandomizedSet struct {
	m map[int]bool
}

//func Constructor() RandomizedSet {
//	return RandomizedSet{
//		m: make(map[int]bool),
//	}
//}

func (this *RandomizedSet) Insert(val int) bool {
	_, ok := this.m[val]
	if !ok {
		this.m[val] = true
		return true
	}
	return false
}

func (this *RandomizedSet) Remove(val int) bool {
	_, ok := this.m[val]
	if !ok {
		return false
	}
	delete(this.m, val)
	return true
}

func (this *RandomizedSet) GetRandom() int {
	for v := range this.m {
		return v
	}
	return 0
}

func deleteNode(head *ListNode, val int) *ListNode {
	node := head
	last := &ListNode{}
	last.Next = head
	for node != nil {
		if node.Val == val {
			last.Next = node.Next
			break
		}
		last = node
		node = node.Next
	}
	return last.Next
}

func hammingWeight(num uint32) int {
	return bits.OnesCount(uint(num))
}
func minArray(numbers []int) int {
	if len(numbers) == 1 {
		return numbers[0]
	}
	for i := 0; i < len(numbers)-1; i++ {
		if numbers[i] > numbers[i+1] {
			return numbers[i+1]
		}
	}
	return numbers[len(numbers)-1]
}

func countNumbersWithUniqueDigits(n int) int {
	str := ""
	var dfs func()
	dfs = func() {
		fmt.Println(str)

		if len(str) == n {
			return
		}
		for i := 1; i <= 9; i++ {
			if len(str) == n-1 {
				break
			}
			k := strconv.Itoa(i)
			str += k + k
			dfs()
			str = str[:len(str)-2]

		}
	}
	dfs()
	return 1

}
func reachingPoints(sx int, sy int, tx int, ty int) bool {

	for {
		//fmt.Println(tx, ty)
		if tx == ty {
			return sx == tx && sy == ty
		}
		if sx == tx && sy == ty {
			return true
		}

		if tx < 0 || ty < 0 {
			return false
		}

		if tx > ty {
			tx -= ty
		} else {
			ty -= tx
		}
	}
}

type ListNode struct {
	Val  int
	Next *ListNode
}

func reverseKGroup(head *ListNode, k int) *ListNode {
	node := head
	var arr []int
	for node != nil {
		arr = append(arr, node.Val)
		node = node.Next
	}
	for i := 0; i < len(arr); i += k {

		r := i + k
		for l := 0; l < k/2; l++ {
			arr[i+l], arr[r] = arr[r], arr[i+l]
			r--
		}
	}
	node = head
	for _, i := range arr {
		node.Val = i
		node.Next = node
	}
	//fmt.Println(arr)
	return head
}

type Node struct {
	Val      int
	Children []*Node
}

func levelOrder(root *Node) (res [][]int) {
	if root == nil {
		return
	}
	var queue []*Node
	queue = append(queue, root)

	for len(queue) != 0 {
		l := len(queue)

		var temp []int
		for _, node := range queue {
			temp = append(temp, node.Val)
		}
		res = append(res, temp)

		for i := 0; i < l; i++ {
			node := queue[0]
			queue = queue[1:]
			for _, child := range node.Children {
				if child != nil {
					queue = append(queue, child)
				}
			}
		}
	}
	return
}
