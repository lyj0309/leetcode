package past

import (
	"fmt"
	"math"
)

func meituanfujia() {
	var n int
	fmt.Scan(&n)
	var arr [][]int
	for i := 0; i < n; i++ {
		arr = append(arr, make([]int, n))
	}
	for i := 0; i < n; i++ {
		for i1 := 0; i1 < n; i1++ {
			fmt.Scan(&arr[i][i1])
		}
	}
	//fmt.Println(arr)
	roMatrix(arr)
}

func roMatrix(arr [][]int) {
	res := 0
	n := len(arr)
	//var arr1 [][]int
	for epoc := 0; epoc < n/2; epoc++ {
		//截取一圈四条边的n-1个数
		bianArr := make([][]int, n-epoc*2-1)
		//var bianArr [][]int

		for c := 1; c <= 4; c++ {
			for i := 0; i < n-epoc*2-1; i++ {
				switch c {
				case 1:
					t := arr[epoc][i]
					bianArr[i] = append(bianArr[i], t)
					//fmt.Print(arr[epoc][i])
				case 2:
					t := arr[i][n-1-epoc]
					bianArr[i] = append(bianArr[i], t)
					//fmt.Print(arr[i][n-1-epoc])
				case 3:
					t := arr[n-1-epoc][n-1-i]
					bianArr[i] = append(bianArr[i], t)
					//fmt.Print(arr[n-1-epoc][n-1-i])
				case 4:
					t := arr[n-1-i][epoc]
					bianArr[i] = append(bianArr[i], t)
					//fmt.Print(arr[n-1-i][epoc])
				}

			}
		}

		for _, ints := range bianArr {
			avg := 0.0
			for _, i := range ints {
				avg += float64(i)
			}
			avg /= float64(len(ints))
			//fmt.Println(avg)

			nearestNum := 0
			near := math.MaxFloat64
			for _, i2 := range ints {
				s := math.Abs(float64(i2) - avg)
				if s < near {
					nearestNum = i2
					near = s
				}
			}
			//fmt.Println("nearestNum", nearestNum, avg, ints)
			for _, i := range ints {
				//fmt.Println(i, nearestNum)
				if i-nearestNum > 0 {
					res += i - nearestNum
				} else {
					res += nearestNum - i
				}
			}

		}
		//fmt.Println(bianArr)
	}

	fmt.Println(res)
	//fmt.Println(arr)
}
func timu4() {
	threeCpu([]int{5, 4, 6, 6, 8, 3, 7})
	return
	var n int
	arr := make([]int, n)
	for i := 0; i < n; i++ {
		fmt.Scan(&arr[i])
	}
	threeCpu(arr)
}

func threeCpu(arr []int) {

}
func timu3() {
	title("acac", "ac")
	return
	var s1, s2 string
	fmt.Scan(&s1)
	fmt.Scan(&s2)
	title(s1, s2)
}

func title(s1, s2 string) {
	res := 0
	ls1 := len(s1)
	ls2 := len(s2)
	//截取长度
	for l := ls2; l <= ls1; l++ {
		for i := 0; i <= ls1-l; i++ {
			subStr := s1[i : i+l]
			//使用双指针判断是否连续子序列
			i2 := 0
			for i1 := 0; i1 < len(subStr); i1++ {
				if subStr[i1] == s2[i2] {
					i2++
				}
				if i2 == ls2 {
					res++
					break
				}
			}
			//	fmt.Println(s1[i:i+l], i2)
		}
	}
	fmt.Println(res)
}

func timu2() {
	var n int
	fmt.Scan(&n)
	arr := make([]int, n)
	for i := 0; i < n; i++ {
		fmt.Scan(&arr[i])
	}
	danarr(arr)
}

func danarr(arr []int) {
	l := len(arr)
	min := math.MaxInt32

	for base := 0; base < l; base++ {
		tempArr := make([]int, l)
		copy(tempArr, arr)
		//需要加的数
		temp := 0
		//从前往后
		for i := 0; i < base-1; i++ {
			for tempArr[i] >= tempArr[i+1] {
				tempArr[i+1]++
				temp++
			}
		}
		//从后往前
		for i := l - 1; i >= base+1; i-- {
			for tempArr[i] >= tempArr[i-1] {
				tempArr[i-1]++
				temp++
			}
		}
		if temp < min {
			min = temp
		}
		//fmt.Println(tempArr, temp)
	}
	fmt.Print(min)
}

func doubleBall(str string) (r int, b int) {

	for i := 0; i < len(str)-1; i++ {
		if str[i] == str[i+1] {
			switch str[i] {
			case 'r':
				r++
			case 'b':
				b++
			}
		}
	}
	return
}
