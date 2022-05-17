package main

import (
	"fmt"
	"math"
	"strconv"
)

//双指针
func main() {

}

func inorderSuccessor(root *TreeNode, p *TreeNode) *TreeNode {
	var res *TreeNode
	f := false
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}

		dfs(node.Left)
		if node == p {
			f = true
		}
		if f {
			f = false
			res = node
		}

		dfs(node.Right)
	}
	dfs(root)
	return res
}

func nuliren() {
	var n, x, y int
	n = 5
	x = 2
	y = 2
	//fmt.Scan(&n, &x, &y)
	arr := make([]int, n)
	//for i := 0; i < n; i++ {
	//	fmt.Scan(&arr[i])
	//}
	arr = []int{10, 8, 2, 7, 1}

	for i := 0; i < n; i++ {
		ruo := false
		for j := 1; j <= x; j++ {
			if i-j < 0 {
				break
			}
			if arr[i-j] < arr[i] {
				ruo = true
				break
			}
		}
		if !ruo {
			for j := 1; j <= y; j++ {
				if i+j >= n {
					break
				}
				if arr[i+j] < arr[i] {
					ruo = true
					break
				}
				//fmt.Println(arr[i+j], arr[i])
			}
		}
		if !ruo {
			fmt.Println(i + 1)
			return
		}

	}
}

//lf >= ls
func oneEditAway(first string, second string) bool {
	if len(first) < len(second) {
		first, second = second, first
	}

	l1 := len(first)
	l2 := len(second)
	//fmt.Println(first, second, l1, l2)

	var tihuan bool
	if l1 == l2 {
		tihuan = true
	} else if l1 == l2+1 {
		tihuan = false
	} else {
		return false
	}

	chance := true
	i1, i2 := 0, 0
	for i2 < l2 {
		//fmt.Println(string(first[i1]), string(second[i2]))
		if first[i1] == second[i2] {
			i1++
			i2++
		} else {
			if !chance {
				return false
			}

			if tihuan {
				i1++
				i2++
			} else {
				i1++
			}
			chance = false

		}
	}
	return true
}

func minDeletionSize(strs []string) int {
	res := 0
	xl := len(strs[0])
	yl := len(strs)
	for x := 0; x < xl; x++ {
		for y := 0; y < yl-1; y++ {
			fmt.Println(string(strs[y][x]), string(strs[y+1][x]), strs[y][x]-strs[y+1][x])
			if strs[y][x] < strs[y+1][x] {
				res++
				break
			}
		}
	}
	return res
}

func diStringMatch(s string) (res []int) {
	i1 := 0
	i2 := len(s)
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case 'I':
			res = append(res, i1)
			i1++
		case 'D':
			res = append(res, i2)
			i2--
		}
	}
	res = append(res, i1+1)
	return
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func eating(tree *TreeNode) int {
	//node := tree
	res1 := 0
	res2 := 0
	var queue []*TreeNode

	queue = append(queue, tree)

	f := 0
	for len(queue) > 0 {
		l := len(queue)
		for i := 0; i < l; i++ {
			node := queue[0]
			if f%2 == 0 {
				res1 += node.Val
			}
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		f++
	}

	queue = []*TreeNode{tree}

	f = 1
	for len(queue) > 0 {
		l := len(queue)
		for i := 0; i < l; i++ {
			node := queue[0]
			if f%2 == 0 {
				res2 += node.Val
			}
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		f++
	}
	if res1 > res2 {
		return res1
	} else {
		return res2
	}

}

//
//func eating(tree *TreeNode) int {
//	res := 0
//	s := 0
//	var dfs func(node *TreeNode) int
//	dfs = func(node *TreeNode) int {
//		if node == nil {
//			return 0
//		}
//
//		//吃或不吃
//		var lv, rv, i int
//		i = 1
//		s += i
//		lv = dfs(node.Left)
//		s -= i
//		s += i
//		rv = dfs(node.Right)
//		s -= i
//		chi := lv + rv
//
//		i = 0
//		s += i
//		lv = dfs(node.Left)
//		s -= i
//		s += i
//		rv = dfs(node.Right)
//		s -= i
//
//		fmt.Println(lv+rv, chi)
//		m := 0
//		if rv+lv > chi {
//			m = rv + lv
//		} else {
//			m = chi
//		}
//
//		if s%2 == 0 {
//			return node.Val + m
//		} else {
//			return m
//		}
//	}
//	m := dfs(tree)
//	fmt.Println(m)
//	return res
//}

//暴力遍历, 找到两个最近的点，再把距离加起来
func getMinLength(pearls [][]int) int {
	res := math.MaxInt32
	l := len(pearls)
	for i := 0; i < l; i++ {
		//找到第一近的点的距离
		n1idx := 0
		d1 := math.MaxInt32
		for i1 := 0; i1 < l; i1++ {
			if i1 == i {
				continue
			}
			//距离
			d := abs(pearls[i1][0]-pearls[i][0]) + abs(pearls[i1][1]-pearls[i][1])
			if d < d1 {
				n1idx = i1
				d1 = d
			}
		}
		d2 := math.MaxInt32
		//找到第二近的点的距离
		for i1 := 0; i1 < l; i1++ {
			if i1 == i || i1 == n1idx {
				continue
			}
			d := abs(pearls[i1][0]-pearls[i][0]) + abs(pearls[i1][1]-pearls[i][1])
			if d < d2 {
				d2 = d
			}
		}
		if d1+d2 < res {
			res = d1 + d2
		}
		fmt.Println(d1, d2)
	}
	return res
}

func abs(a int) int {
	if a > 0 {
		return a
	}
	return -a
}

type RecentCounter struct {
	Queue []int
}

func Constructor() RecentCounter {
	return RecentCounter{}
}

func (this *RecentCounter) Ping(t int) int {
	this.Queue = append(this.Queue, t)
	l := len(this.Queue)
	res := 0
	pt := t - 3000
	for i := l; i >= 0; i-- {
		if this.Queue[i] <= pt {
			break
		}
		res++
	}
	return res
}

func smallestRangeI(nums []int, k int) int {
	max := -1
	min := 100000
	for _, num := range nums {
		if num > max {
			max = num
		}
		if num < min {
			min = num
		}
	}
	r := max - k - (min + k)
	if r < 0 {
		return 0
	}
	return r
}
func movingCount(m int, n int, k int) int {
	var arr [][]bool
	for i := 0; i < m; i++ {
		arr = append(arr, make([]bool, n))
	}
	sum := 0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			is := strconv.Itoa(i)
			js := strconv.Itoa(j)
			var s int
			for a := 0; a < len(is); a++ {
				//fmt.Println(is[a])
				s += int(is[a] - 48)
			}
			for a := 0; a < len(js); a++ {
				//fmt.Println(js[a])
				s += int(js[a] - 48)
			}
			fmt.Println(i, j, s)

			if s <= k {
				if j > 0 && arr[i][j-1] == false {
					continue
				}
				if i > 0 && arr[i-1][j] == false {
					continue
				}
				arr[i][j] = true
				sum++
			}
		}
	}
	//fmt.Println(sum)
	return sum
}

//dp dp[i] 是i长度的最大乘积
//dp[i] = max(dp[i],max(j*(i-j),j*dp[i-j]))
func cuttingRope(n int) int {
	max := func(a, b int) int {
		if a > b {
			return a
		}
		return b
	}
	n++
	dp := make([]int, n)
	dp[2] = 1
	for i := 3; i < n; i++ {
		for j := 2; j < i; j++ {
			dp[i] = max(dp[i], max(j*(i-j), j*dp[i-j]))
		}
	}
	return dp[n-1]
}
