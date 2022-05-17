package main

import (
	"fmt"
	"github.com/Shopify/sarama"
	"log"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
)

func myAtoi(s string) int {
	s = strings.ReplaceAll(s, " ", "")
	k := len(s)
	for i := 1; i < len(s); i++ {
		fmt.Println(s[i : i+1])
		_, e := strconv.Atoi(s[i : i+1])
		if e != nil {
			k = i
			break
		}
	}
	//fmt.Println(s[:k])
	res, _ := strconv.Atoi(s[:k])

	if res < -2147483648 {
		return -2147483648
	}
	if res > 2147483648 {
		return 2147483647
	}
	return res
}

func main() {
	fmt.Println(myAtoi("4193 with words"))
	return
	config := sarama.NewConfig()

	config.Producer.Return.Successes = true
	config.Producer.Partitioner = sarama.NewRandomPartitioner

	client, err := sarama.NewClient([]string{"49.233.7.107:9092"}, config)
	defer client.Close()
	if err != nil {
		panic(err)
	}
	producer, err := sarama.NewAsyncProducerFromClient(client)
	if err != nil {
		panic(err)
	}

	// Trap SIGINT to trigger a graceful shutdown.
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, os.Interrupt)

	var (
		wg                          sync.WaitGroup
		enqueued, successes, errors int
	)

	wg.Add(1)
	// start a groutines to count successes num
	go func() {
		defer wg.Done()
		for range producer.Successes() {
			successes++
		}
	}()

	wg.Add(1)
	// start a groutines to count error num
	go func() {
		defer wg.Done()
		for err := range producer.Errors() {
			log.Println(err)
			errors++
		}
	}()

ProducerLoop:
	for {
		message := &sarama.ProducerMessage{Topic: "my_topic", Value: sarama.StringEncoder("testing 123")}
		select {
		case producer.Input() <- message:
			enqueued++

		case <-signals:
			producer.AsyncClose() // Trigger a shutdown of the producer.
			break ProducerLoop
		}
	}

	wg.Wait()

	log.Printf("Successfully produced: %d; errors: %d\n", successes, errors)
}
