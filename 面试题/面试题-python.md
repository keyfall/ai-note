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

#### docker
- **镜像操作**
  - `docker pull [OPTIONS] NAME[:TAG|@DIGEST]` 拉取一个镜像或仓库到本地
  - `docker images [OPTIONS]` 列出本地镜像
  - `docker rmi [OPTIONS] IMAGE [IMAGE...]` 移除一个或多个镜像
  - `docker build [OPTIONS] PATH | URL | -` 使用Dockerfile构建镜像
  - `docker tag SOURCE_IMAGE[:TAG] TARGET_IMAGE[:TAG]` 标记本地镜像，将其归入某一仓库

- **容器操作**
  - `docker run [OPTIONS] IMAGE [COMMAND] [ARG...]` 创建并启动一个容器
  - `docker ps [OPTIONS]` 列出当前运行的容器
  - `docker stop [OPTIONS] CONTAINER [CONTAINER...]` 停止运行中的容器
  - `docker start [OPTIONS] CONTAINER [CONTAINER...]` 启动已存在的容器
  - `docker restart [OPTIONS] CONTAINER [CONTAINER...]` 重启容器
  - `docker rm [OPTIONS] CONTAINER [CONTAINER...]` 移除一个或多个容器
  - `docker exec [OPTIONS] CONTAINER COMMAND [ARG...]` 在运行中的容器内执行命令
  
- **日志与信息查看**
  - `docker logs [OPTIONS] CONTAINER` 获取容器的日志
  - `docker inspect [OPTIONS] NAME|ID [NAME|ID...]` 查看容器详细信息
  
- **网络相关**
  - `docker network ls` 列出所有网络
  - `docker network create [OPTIONS] NETWORK` 创建网络
  - `docker network connect [OPTIONS] NETWORK CONTAINER` 将容器连接到网络
  - `docker network disconnect [OPTIONS] NETWORK CONTAINER` 断开容器与网络的连接
  
- **卷管理**
  - `docker volume ls` 列出卷
  - `docker volume create [OPTIONS] [VOLUME]` 创建卷
  - `docker volume rm [OPTIONS] VOLUME [VOLUME...]` 移除卷
  
- **系统范围的操作**
  - `docker system df` 显示Docker磁盘使用情况
  - `docker system prune [OPTIONS]` 清理未使

#### 下列linux常用命令：

| **命令**    | **功能**                           |
| --------- | -------------------------------- |
| `touch`   | 创建空文件或更新文件的时间戳                   |
| `cat`     | 显示文件内容或将多个文件合并后显示                |
| `less`    | 分页显示文件内容，支持前后翻页                  |
| `head`    | 显示文件开头部分，默认前10行                  |
| `tail`    | 显示文件结尾部分，默认最后10行                 |
| `grep`    | 在文件中搜索指定模式的文本行                   |
| `find`    | 在目录层次结构中查找文件                     |
| `chmod`   | 更改文件或目录的访问权限                     |
| `chown`   | 更改文件或目录的所有者和组                    |
| `ps`      | 显示当前系统中的进程状态                     |
| `top`     | 实时监控系统的资源使用情况                    |
| `kill`    | 终止进程                             |
| `df`      | 显示磁盘空间使用情况                       |
| `du`      | 显示文件或目录的磁盘使用情况                   |
| `man`     | 显示命令的手册页，提供详细的命令帮助信息             |
| `wc`      | 统计文件中的行数、单词数和字节数                 |
| `awk`     | 文本处理工具，用于数据提取和报告生成               |
| `sed`     | 流编辑器，通过命令对文本直接操作，用于对输入流进行基本的文本转换 |
| `crontab` | 设置定时任务                           |
| `uniq`    | 去除相邻重复行或统计重复次数                   |
| `ssh`     | 远程登录到另一台计算机                      |
| `scp`     | 在本地和远程计算机之间复制文件                  |
| `wget`    | 下载网络文件                           |
| `curl`    | 获取或发送URL数据                       |
| `netstat` | 显示网络连接、路由表、接口统计信息等               |
| `whoami`  | 显示当前登录用户的名字                      |

#### HTTP常用状态码
|           |     |                       |                              |
| --------- | --- | --------------------- | ---------------------------- |
| **信息响应**  | 100 | Continue              | 请求已接收，客户端应继续发送请求体。           |
|           | 101 | Switching Protocols   | 服务器已理解客户端的请求，并将通过升级协议进行响应。   |
| **成功响应**  | 200 | OK                    | 请求成功，服务器返回所请求的数据。            |
|           | 201 | Created               | 请求成功且服务器创建了新的资源。             |
|           | 204 | No Content            | 请求成功，但响应报文不含实体的主体部分。         |
| **重定向**   | 301 | Moved Permanently     | 永久重定向                        |
|           | 302 | Found                 | 临时重定向                        |
|           | 304 | Not Modified          | 资源未被修改，可以使用缓存版本。             |
| **客户端错误** | 400 | Bad Request           | 由于语法错误，服务器无法理解该请求。           |
|           | 401 | Unauthorized          | 当前请求需要用户验证。                  |
|           | 403 | Forbidden             | 服务器理解请求但拒绝执行。无权限             |
|           | 404 | Not Found             | 服务器找不到请求的网页或资源。              |
|           | 405 | Method Not Allowed    | 禁用请求中指定的方法。                  |
|           | 429 | Too Many Requests     | 用户在给定时间内发送了太多请求。             |
| **服务器错误** | 500 | Internal Server Error | 服务器遇到未知情况，无法完成请求。            |
|           | 501 | Not Implemented       | 服务器不支持请求的功能，无法完成请求。          |
|           | 502 | Bad Gateway           | 作为网关或代理角色的服务器，从上游服务器收到无效响应。  |
|           | 503 | Service Unavailable   | 服务器暂时不可用（过载或维护）。             |
|           | 504 | Gateway Timeout       | 作为网关或代理角色的服务器，未能及时从上游服务器获得响应 |

### TCP三次握手（Three-Way Handshake）

用于建立连接：

1. **第一次握手**：客户端发送一个带有SYN（同步序列编号，Synchronize Sequence Numbers）标志的数据包到服务器，并进入SYN_SENT状态。
2. **第二次握手**：服务器接收到客户端的SYN包后，必须对这个SYN包进行确认，同时自己也发送一个SYN包，即SYN+ACK包，此时服务器进入SYN_RECV状态。
3. **第三次握手**：客户端接收到服务器的SYN+ACK包后，还需向服务器发送确认信息ACK，此包发送完毕后，客户端和服务端都进入ESTABLISHED状态，完成三次握手。

这确保了双方都能发送和接收数据，并且可以同步双方的初始序列号。

### TCP四次挥手（Four-Way Handshake）

用于终止连接：

1. **第一次挥手**：主动关闭方发送一个FIN（结束标志，Finish），用来关闭自己的写通道。这时主动关闭方进入FIN_WAIT_1状态。
2. **第二次挥手**：被动关闭方收到FIN包后，向主动关闭方发送ACK确认信息，被动关闭方进入CLOSE_WAIT状态。主动关闭方收到这个ACK后进入FIN_WAIT_2状态。
3. **第三次挥手**：当被动关闭方准备好关闭时，它会发送自己的FIN包给主动关闭方以关闭从被动关闭方到主动关闭方的数据流，被动关闭方进入LAST_ACK状态。
4. **第四次挥手**：主动关闭方收到FIN包后，发送ACK确认信息给被动关闭方，然后进入TIME_WAIT状态，等待足够的时间确保被动关闭方收到了确认信息之后才进入CLOSED状态。被动关闭方收到ACK包后立即进入CLOSED状态。