package past

import "fmt"

func chijiin() {
	var t int
	fmt.Scan(&t)
	var arr [][]int
	for i := 0; i < t; i++ {
		arr = append(arr, make([]int, 4))
	}
	for i := 0; i < t; i++ {
		for i1 := 0; i1 < 4; i1++ {
			fmt.Scan(&arr[i][i1])
		}
	}
	for i := 0; i < t; i++ {
		chiji(arr[i])
	}
}

func chiji(arr []int) {
	res := arr[3]
	res += arr[1] / 2
	//2人
	if arr[1]%2 == 1 {
		arr[1] = 1
	}

	//3人
	if arr[2] <= arr[0] {
		arr[0] -= arr[2]
		res += arr[2]
	} else {
		res += arr[0]
		arr[0] = 0
	}

	if arr[1] == 1 && arr[0] >= 2 {
		res++
		arr[0] -= 2
	}

	res += arr[0] / 4

	//fmt.Println(arr)
	fmt.Println(res)

}

type Queue struct {
	Arr []int
}

func (q Queue) sum() int {
	sum := 0
	for _, i := range q.Arr {
		sum += i
	}
	return sum
}

func (q Queue) Add(i int) {
	for i := 1; i < 7; i++ {
		q.Arr[i-1] = q.Arr[i]
	}
	q.Arr[6] = i

}

func queConstruct() *Queue {
	return &Queue{
		Arr: make([]int, 7),
	}
}

func bangyi(arr [][]int) {
	res := 0
	var queues []*Queue
	for i := 0; i < len(arr[0]); i++ {
		queues = append(queues, queConstruct())
	}
	//最后一个是小明的
	xmQue := queConstruct()
	for _, ints := range arr {
		for i, que := range queues {
			que.Add(ints[i])
		}
		//获取最大值
		max := 0
		for _, que := range queues {
			t := que.sum()
			if t > max {
				max = t
			}
		}
		xmsum := xmQue.sum()
		xmsum -= xmQue.Arr[0]
		var pay int
		if xmsum <= max {
			pay = max + 1 - xmsum
		}
		res += pay
		xmQue.Add(pay)
		//fmt.Println(queues[0], xmQue)
		//fmt.Println(max, xmsum, pay)

	}

	fmt.Print(res)
}

func ti1() {
	bangyi([][]int{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}})
	return
	var n, m int
	fmt.Scan(&n, &m)
	var arr [][]int
	for i := 0; i < n; i++ {
		arr = append(arr, make([]int, m))
	}
	for y, ints := range arr {
		for x := range ints {
			fmt.Scan(&arr[y][x])
		}
	}
	bangyi(arr)
	fmt.Println(arr)
}
