package past

import "fmt"

func fenshua(str string) {
	//res:=0
	var a, b, c int
	dest := len(str) / 3
	for i := 0; i < len(str); i++ {
		switch str[i] {
		case 'A':
			a++
		case 'B':
			b++
		case 'C':
			c++
		}
	}
	difA := dest - a
	difB := dest - b
	difC := dest - c

	if difA == 0 && difB == 0 && difC == 0 {
		fmt.Println(0)
		return
	}
	l := len(str)
	if difA > 0 {
		for i := 0; i < l; i++ {
			tmpC := difC
			tmpB := difB
			ti := i
			for (tmpC == 0 && str[i+1] == 'C') || (tmpB == 0 && str[i+1] == 'B') || str[i+1] == 'A' {
				if i == l-1 {
					break
				}
				switch str[i] {
				case 'C':
					tmpC--
				case 'B':
					tmpB--
				}
				i++

			}
			if tmpC == 0 && tmpB == 0 {
				fmt.Println(1)
				return
			}
			i = ti
		}
	} else if difB > 0 {
		for i := 0; i < l; i++ {
			if i == l-1 {
				break
			}

			tmpA := difA
			tmpC := difC
			ti := i
			for (tmpA == 0 && str[i+1] == 'A') || (tmpC == 0 && str[i+1] == 'C') || str[i+1] == 'B' {
				switch str[i] {
				case 'A':
					tmpA--
				case 'C':
					tmpC--
				}
				i++

			}
			if tmpA == 0 && tmpC == 0 {
				fmt.Println(1)
				return
			}
			i = ti
		}

	} else {
		//寻找包含ab串的长度
		for i := 0; i < l; i++ {
			if i == l-1 {
				break
			}

			tmpA := difA
			tmpB := difB
			ti := i
			for (tmpA == 0 && str[i+1] == 'A') || (tmpB == 0 && str[i+1] == 'B') || str[i+1] == 'C' {
				switch str[i] {
				case 'A':
					tmpA--
				case 'B':
					tmpB--
				}
				i++

			}
			if tmpA == 0 && tmpB == 0 {
				fmt.Println(1)
				return
			}
			i = ti
		}
	}

	//fmt.Println(difA, difB, difC)
	fmt.Println(3)
}

func ti1() {
	var n, x int
	fmt.Scan(&n, &x)
	var str string
	fmt.Scan(&str)
	workTree(x, str)
}

func workTree(x int, str string) {
	goUp := func(n, floor int) int {
		first := pow(2, floor-1)
		des := pow(2, floor-2) + ((n - first) / 2)
		//fmt.Println("up", floor-2, pow(2, floor-2))
		return des
	}
	goLeft := func(n, floor int) int {
		first := pow(2, floor-1)
		return pow(2, floor) + (n-first)*2
	}
	goRight := func(n, floor int) int {
		first := pow(2, floor-1)
		return pow(2, floor) + (n-first)*2 + 1
	}

	//k := 7
	var floor int
	for i := 1; i < 30; i++ {
		end := pow(2, i) - 1
		if end >= 2 {
			floor = i
			//fmt.Println(i, pow(2, i))
			break
		}
	}
	fmt.Println(x, floor)

	for i := 0; i < len(str); i++ {
		switch str[i] {
		case 'U':
			x = goUp(x, floor)
			floor--
		case 'R':
			x = goRight(x, floor)
			floor++
		case 'L':
			x = goLeft(x, floor)
			floor++
		}
		fmt.Println(x, floor)
	}
}

func pow(x, y int) int {
	if y == 0 {
		return 1
	}

	p := x
	for i := 0; i < y-1; i++ {
		x *= p
	}
	return x
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func geneTree() {

}
