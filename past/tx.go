package past

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

//回溯
func numsOfStrings(n int, k int) int {

	//可能的切割
	//var temp []int
	kind := 0
	var dfs func()
	dfs = func() {

		if k == 0 {
			if n == 0 {
				kind++
				//fmt.Println(temp)
			}
			return
		} else if k < 0 || n < 0 {
			return
		}

		for i := 2; i <= n; i++ {
			n -= i
			k--
			//temp = append(temp, n)
			dfs()
			n += i
			k++
			//temp = temp[:len(temp)-1]
		}
	}
	dfs()
	//
	//fmt.Println(k)
	sum := 1
	for i := 0; i < k-1; i++ {
		sum *= 26
		sum %= 1000000
	}
	sum *= (26 - k)

	sum *= kind
	sum %= 1000000

	return sum
}

func minCnt(s string) int {
	res := 0
	l := len(s)
	for i := l - 1; i > 0; i-- {
		//fmt.Println(string(s[i]))
		if s[i] == '1' {
			res++
		}
	}
	return res
}

//最大堆（知道但不会用）
func minMax(a []int, k int, x int) int {
	sort.Ints(a)
	l := len(a)
	for i := 0; i < k; i++ {
		//fmt.Println(a)

		a[l-1] -= x

		//排序
		last := a[l-1]
		//idx := l-1
		if a[0] >= last {
			for i2 := l - 1; i2 > 0; i2-- {
				a[i2] = a[i2-1]
			}
			a[0] = last
			continue
		}

		for i1 := l - 2; i1 >= 0; i1-- {
			if a[i1] <= last {
				//fmt.Println(a[i1])
				for i2 := l - 1; i2 > i1; i2-- {
					a[i2] = a[i2-1]
				}
				a[i1+1] = last
				break
			}
		}
	}
	//fmt.Println(a)

	return a[l-1]
}

func intToRoman(num int) string {
	ints := []int{1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000}
	strs := []string{"I", "IV", "V", "IX", "X", "XL", "L", "XC", "C", "CD", "D", "CM", "M"}
	str := ""
	l := len(ints)
	m := l - 1
	for num > 0 {
		for i := m; i >= 0; i-- {
			if num >= ints[i] {
				num -= ints[i]
				str += strs[i]
				m = i
				break
			}
		}
		//	fmt.Println(num, str)
	}
	//for i := 0; i < len(ints); i++ {
	//	fmt.Println(ints[i], strs[i])
	//}
	return str
}

func rotateString(s string, goal string) bool {
	if len(goal) != len(s) {
		return false
	}
	s += s
	return strings.Contains(s, goal)
}

func ti2() {
	killTree([][]int{{3, 3, 3}, {3, 1, 1}, {3, 1, 1}}, 4)
	return
	var n, m, C int
	fmt.Scan(&n, &m, &C)

	var arr [][]int
	for i := 0; i < n; i++ {
		arr = append(arr, make([]int, m))
	}
	for y := 0; y < n; y++ {
		for x := 0; x < m; x++ {
			fmt.Scan(&arr[y][x])
		}
	}
	killTree(arr, C)
}
func killTree(arr [][]int, c int) {
	lenX := len(arr[0])
	lenY := len(arr)
	var l int
	if lenY > lenX {
		l = lenX
	} else {
		l = lenY
	}

	bianli := func(a, b, ll int) bool {
		temp := c
		for y := a; y < a+ll; y++ {
			for x := b; x < b+ll; x++ {
				temp -= arr[y][x]
				if temp < 0 {
					return false
				}
			}
		}
		return true
	}
	//fmt.Println("len", l)

	for i := l; i >= 1; i-- {
		for y := 0; y <= lenY-i; y++ {
			for x := 0; x <= lenX-i; x++ {
				//fmt.Println(arr[y][x], y, x, i)
				if bianli(y, x, i) {
					//	fmt.Println(i)
					return
				}
			}
		}
	}
	fmt.Println(0)
}

func ti1() {
	var n int
	fmt.Scan(&n)
	for i := 0; i < n; i++ {
		var target uint64
		fmt.Scan(&target)
		magic(target)
	}
}
func magic(target uint64) {
	res := 0
	for target > 1 {
		if target%2 == 0 {
			target /= 2
		} else {
			target--
		}
		//fmt.Println(target)
		res++
	}
	fmt.Println(res)
}

type ListNode struct {
	Val  int
	Next *ListNode
}

type MapNode struct {
	Val  int
	Next []*MapNode
	//Last []*MapNode
}

func findMinHeightTrees(n int, edges [][]int) (res []int) {
	var nodes []*MapNode
	for i := 0; i < n; i++ {
		nodes = append(nodes, &MapNode{Val: i})
	}
	for _, edge := range edges {
		nodes[edge[0]].Next = append(nodes[edge[0]].Next, nodes[edge[1]])
		nodes[edge[1]].Next = append(nodes[edge[1]].Next, nodes[edge[0]])
		//nodes[edge[0]].Last = append(nodes[edge[0]].Last, &nodes[edge[1]])
		//nodes[edge[1]].Last = append(nodes[edge[1]].Last, &nodes[edge[0]])
	}

	//层次遍历
	bfs := func(n *MapNode) int {
		var queue []*MapNode
		queue = append(queue, n)
		visited := make(map[*MapNode]bool)
		visited[n] = true

		floor := 0
		for len(queue) != 0 {
			floor++
			l := len(queue)
			for i := 0; i < l; i++ {
				node := queue[0]

				queue = queue[1:]
				//fmt.Println(queue)
				for _, mapNode := range node.Next {
					if mapNode != nil && !visited[mapNode] {
						visited[node] = true
						queue = append(queue, mapNode)
					}
				}
			}
			//fmt.Println(floor)

			//for _, node := range queue {
			//	fmt.Print(node)
			//}
			//fmt.Println()

		}
		return floor
	}
	minFloor := math.MaxInt32
	for _, node := range nodes {
		//fmt.Println("val", node.Val)
		f := bfs(node)
		//fmt.Println("floor", f)

		if f < minFloor {
			res = []int{node.Val}
			minFloor = f
		} else if f == minFloor {
			res = append(res, node.Val)
		}
	}

	//for _, node := range nodes {
	//	fmt.Println(node)
	//}
	return
}

func reversePrint(head *ListNode) (res []int) {
	for head != nil {
		res = append(res, head.Val)
		head = head.Next
	}
	k := len(res)
	for i := 0; i < len(res)/2; i++ {
		res[i], res[k] = res[k], res[i]
		k--
	}
	return res
}

func fib(n int) int {
	if n == 0 {
		return 0
	}
	a := 0
	b := 1
	for i := 0; i < n-1; i++ {
		a, b = b, (a+b)%1000000007

	}
	//fmt.Println(b)
	return b
}

func replaceSpace(s string) string {
	return strings.ReplaceAll(s, " ", "%20")
}

func isIn(arr []int, target int) bool {
	if len(arr) == 0 {
		return false
	}
	left, right := 0, len(arr)
	for {
		idx := (left + right) / 2
		if arr[idx] == target {
			return true
		}

		fmt.Println(left, right, idx, arr[left:right])
		if left >= right-1 {
			return false
		}
		if arr[idx] > target {
			right = idx
		} else {
			left = idx
		}
	}
}

func findNumberIn2DArray(matrix [][]int, target int) bool {
	for _, ints := range matrix {
		if isIn(ints, target) {
			return true
		}

	}
	return false
}
func findRepeatNumber(nums []int) int {
	m := make(map[int]bool)
	for _, num := range nums {
		_, ok := m[num]
		if ok {
			return num
		}
		m[num] = true
	}
	return 0
}

func countPrimeSetBits(left int, right int) int {
	checkZhi := func(n int) bool {
		if n == 1 {
			return false
		}
		for i := 2; i*i < n; i++ {
			if n%i == 0 {
				return false
			}
		}
		return true
	}
	get1 := func(n int) int {
		c := 0
		for n != 0 {
			if n%2 == 1 {
				c++
			}
			n /= 2
		}
		return c
	}
	res := 0
	for i := left; i <= right; i++ {
		if checkZhi(get1(i)) {
			//fmt.Println(i, get1(i))
			res++
		}
	}
	return res
}
func searchInsert(nums []int, target int) int {
	l := len(nums)
	for i := 0; i < l; i++ {
		//fmt.Println(nums[i])
		if target == nums[i] {
			return i
		}
		if target < nums[i] {
			return i
		}
	}
	return l
}

func solveNQueens(n int) (res [][]string) {
	//var arr [][]int
	//for i := 0; i < n; i++ {
	//	arr = append(arr, make([]int, n))
	//}
	var allPosition [][]int

	visited := make(map[int]bool)
	var temp []int
	var dfs func()
	dfs = func() {
		if len(temp) == n {
			allPosition = append(allPosition, append([]int{}, temp...))
			//fmt.Println(temp)
			return
		}

		for i := 0; i < n; i++ {
			if visited[i] {
				continue
			}
			no := false
			l := len(temp)
			//fmt.Println(i, temp)
			for i1 := 0; i1 < l; i1++ {
				if i == temp[i1]-l+i1 || i == temp[i1]+l-i1 {
					no = true
					break
				}
			}
			if no {
				continue
			}

			//if len(temp) > 0 && (i == temp[len(temp)-1]+1 || i == temp[len(temp)-1]-1) {
			//	continue
			//}
			temp = append(temp, i)
			visited[i] = true
			dfs()
			visited[i] = false
			temp = temp[:len(temp)-1]
		}
	}
	dfs()
	for _, ints := range allPosition {
		var tempStrs []string
		for _, i := range ints {
			s := ""
			for i1 := 0; i1 < n; i1++ {
				if i1 == i {
					s += "Q"
				} else {
					s += "."
				}
			}
			tempStrs = append(tempStrs, s)
		}
		res = append(res, tempStrs)
	}

	//fmt.Println(allPosition, res)
	return

}

type NumArray struct {
	arr   []int
	cache []int
}

func Constructor(nums []int) NumArray {
	var cache []int
	sum := nums[0]
	for i := 0; i < len(nums); i++ {
		cache = append(cache, sum)
		sum += nums[i]
		//fmt.Println(sum, nums[i], cache)
	}
	cache = append(cache, sum)
	return NumArray{arr: nums, cache: cache}
}

func (this *NumArray) Update(index int, val int) {
	fmt.Println(this.cache)
	sub := this.arr[index] - val
	this.arr[index] = val
	for i := index; i < len(this.arr); i++ {
		this.cache[i] -= sub
	}
	fmt.Println(this.cache)

}

func (this *NumArray) SumRange(left int, right int) int {
	return this.cache[right] - this.cache[left]
}

func isIsomorphic(s string, t string) bool {
	l := len(s)
	m1 := make(map[byte]byte)
	m2 := make(map[byte]byte)
	for i := 0; i < l; i++ {
		b1, ok1 := m1[s[i]]
		if !ok1 {
			m1[s[i]] = t[i]
		} else if b1 != t[i] {
			return false
		}
		b2, ok2 := m2[t[i]]
		if !ok2 {
			m2[t[i]] = s[i]
		} else if b2 != s[i] {
			return false
		}
		//	fmt.Println(m1, m2)

	}
	return true
}

func reverseStr(s string, k int) string {
	l := len(s)
	res := ""
	for i := k * 2; i <= l; i += k * 2 {
		for i1 := 1; i1 <= k; i1++ {
			res += string(s[i-k-i1])
			//fmt.Println(string(s[i-k-i1]))
		}
		res += s[i-k : i]
		//fmt.Println(res)
	}
	yu := l % (2 * k)
	if yu >= k {
		for i1 := 1; i1 <= k; i1++ {
			res += string(s[l-yu+k-i1])
			//fmt.Println(string(s[i-k-i1]))
		}
		res += s[l-yu+k : l]
	} else {
		for i1 := 1; i1 <= yu; i1++ {
			res += string(s[l-i1])
			//fmt.Println(string(s[i-k-i1]))
		}
	}

	return res
}

func myAtoi(s string) int {
	l := len(s)
	fu := false
	zheng := false
	var n int
	for i := 0; i < l; i++ {
		if s[i] == '-' {
			if zheng {
				break
			}
			fu = true
			continue
		}
		if s[i] == '+' {
			if fu {
				break
			}
			zheng = true
			continue
		}

		if s[i] == ' ' && !zheng && !fu {
			continue
		}

		k := 0
		for i < l && s[i] >= '0' && s[i] <= '9' {
			i++
			k++
		}
		if k != 0 {
			n, _ = strconv.Atoi(s[i-k : i])
			//fmt.Println(s[i-k:i], fu, n)
			break
		} else {
			break
		}
	}
	num := int(math.Pow(2, 31))
	if fu {
		n = -n
	}
	//fmt.Println(n)

	if n < -num {
		n = -num
	}
	if n > num-1 {
		n = num - 1
	}

	return n

}

func nextGreatestLetter(letters []byte, target byte) byte {
	l := len(letters)
	for i := target + 1; i <= 'z'; i++ {
		for i1 := 0; i1 < l; i1++ {
			if letters[i1] == i {
				return i
			}
		}
	}
	return 'a'
}
