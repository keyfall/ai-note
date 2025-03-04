#### 内存溢出：
看情况
- 列表原因，就可以换生成器来迭代数据
- 大文件或者数据集，可以考虑分割小块，分块处理
- 手动调用`gc.collect()`来立即进行垃圾回收，或设置垃圾回收阈值增大触发频率，阈值是如果你设置了第0代的阈值为100，那么每当Python程序中新创建并丢弃的对象数量达到100时，就会触发一次针对该代以及所有比它年轻的代（即更低代数的代）的垃圾回收。
```
import gc

# 查看当前的垃圾回收阈值
print(gc.get_threshold())

# 设置新的阈值，参数分别为三代对象的阈值
# 这里我们将它们设置得更低以使垃圾回收更频繁发生
gc.set_threshold(100, 10, 5)

# 再次查看阈值确认是否修改成功
print(gc.get_threshold())
```

#### 线程间通讯
- 通过队列通讯，queue.Queue，生产者-消费者模式
```
import queue
import threading
import time

def producer(q):
    for i in range(5):
        print(f'Producing item {i}')
        q.put(i)
        time.sleep(1)

def consumer(q, name):
    while True:
        item = q.get()
        if item is None:
            break
        print(f'Consumer-{name} processing item {item}')
        q.task_done()

q = queue.Queue()
producer_thread = threading.Thread(target=producer, args=(q,))
consumer_threads = [threading.Thread(target=consumer, args=(q, i)) for i in range(2)]

producer_thread.start()
for t in consumer_threads:
    t.start()

producer_thread.join()
# Stop consumers
for _ in consumer_threads:
    q.put(None)
for t in consumer_threads:
    t.join()
```
- 特定事件发生，threading.event
```
import threading

def worker(event):
    print('Worker is working...')
    # Simulate work with sleep
    threading.Event().wait(3)  # 模拟工作耗时
    event.set()  # 通知主线程任务已完成

event = threading.Event()
t = threading.Thread(target=worker, args=(event,))
t.start()

print('Main thread waiting for worker...')
event.wait()  # 等待worker线程设置事件
print('Worker finished, main thread continues.')
```
- threading.Condition的同步
```
import threading

class BoundedBuffer:
    def __init__(self, size=5):
        self.buffer = [None] * size
        self.size = size
        self.count = 0
        self.in_position = 0
        self.out_position = 0
        self.lock = threading.Condition()

    def produce(self, x):
        with self.lock:
            while self.count == self.size:
                self.lock.wait()  # 如果缓冲区满，则生产者等待
            self.buffer[self.in_position] = x
            self.in_position = (self.in_position + 1) % self.size
            self.count += 1
            self.lock.notify()  # 通知可能在等待的消费者

    def consume(self):
        with self.lock:
            while self.count == 0:
                self.lock.wait()  # 如果缓冲区空，则消费者等待
            x = self.buffer[self.out_position]
            self.out_position = (self.out_position + 1) % self.size
            self.count -= 1
            self.lock.notify()  # 通知可能在等待的生产者
            return x

# 示例使用
buffer = BoundedBuffer(2)

def producer():
    for i in range(4):
        buffer.produce(i)
        print(f"Produced {i}")

def consumer():
    for _ in range(4):
        item = buffer.consume()
        print(f"Consumed {item}")

t1 = threading.Thread(target=producer)
t2 = threading.Thread(target=consumer)

t1.start()
t2.start()

t1.join()
t2.join()
```

#### 进程，协程，线程区别
| 特性/类型       | 线程 (Threads)       | 进程 (Processes)        | 协程 (Coroutines) |
| ----------- | ------------------ | --------------------- | --------------- |
| **内存共享**    | 是（共享内存空间）          | 否（独立内存空间）             | 是（同一进程内）        |
| **通信难度**    | 较简单                | 复杂（需要IPC）             | 相对简单（在同一进程中）    |
| **资源消耗**    | 中等                 | 高                     | 低               |
| **上下文切换开销** | 低                  | 高                     | 极低              |
| **隔离性**     | 差                  | 好                     | 差               |
| **适用场景**    | 并发任务，快速响应          | 并行任务，安全要求高            | 高并发I/O密集型任务     |
| **优点**      | 数据共享效率高，适合I/O密集型任务 | 更好的稳定性和安全性，适合CPU密集型任务 | 高效，易于编写异步代码     |
| **缺点**      | 容易出现同步问题，调试困难      | 开销大，进程间通信复杂           | 无法直接利用多核优势      |
#### 协程
协程是一个由代码控制的异步程序，一个
```
import asyncio

async def producer(queue):
    for i in range(5):
        await asyncio.sleep(1)  # 模拟生产者生成数据的时间延迟
        print(f'Producer: 生产数据 {i}')
        await queue.put(i)
    await queue.put(None)  # 发送结束信号

async def consumer(queue):
    while True:
        item = await queue.get()
        if item is None:
            break  # 结束信号
        print(f'Consumer: 处理数据 {item}')
        await asyncio.sleep(0.5)  # 模拟消费者处理数据的时间延迟
    print('Consumer: 完成')

async def main():
    queue = asyncio.Queue()

    # 创建生产者和消费者的任务
    producer_task = asyncio.create_task(producer(queue))
    consumer_task = asyncio.create_task(consumer(queue))

    # 等待所有任务完成
    await asyncio.gather(producer_task, consumer_task)

# 运行异步主函数
asyncio.run(main())
```